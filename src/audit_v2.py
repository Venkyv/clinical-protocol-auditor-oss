"""
Sprint 2 (Redesigned) — Audit Engine v2
Sprint 7 — Structural Evidence Scoring with evidence_anchors

Architecture: LLM Extractor → Code Judge + Evidence Scorer → LLM Explainer

Step 1: LLM EXTRACTS  (llama-3.1-8b-instant)
Step 2: CODE JUDGES + EVIDENCE SCORER — zero LLM, fully deterministic

        Pre-gate: structural heading / ToC detector
          Any text matching a heading or ToC pattern scores 0/3 immediately.

        Gate 1 — Length:      evidence >= 80 characters           (1 pt)
        Gate 2 — Depth:       contains procedural language         (1 pt)
        Gate 3 — Specificity: matches a rule-specific anchor term  (1 pt)

        Score 3/3 → PASS
        Score < 3 → MANUAL_REVIEW  (GxP-safe, never a false PASS)

        Anchors live in evidence_anchors in each rule's JSON entry.
        New rules or protocols only need JSON updates — no code changes.

Step 3: LLM EXPLAINS (llama-3.3-70b-versatile — FAIL only)
"""

import os
import sys
import json
import re
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

BASE_DIR   = Path(__file__).parent.parent
CHROMA_DIR = str(BASE_DIR / "chroma_db")
RULES_DIR  = BASE_DIR / "rules"

load_dotenv(BASE_DIR / ".env")


# ── Prompts ────────────────────────────────────────────────────

EXTRACTOR_PROMPT = """You are a precise document analyst. Your only job is to extract relevant text from a clinical trial protocol.

You will be given:
1. An extraction instruction — what to look for
2. Protocol content — the document to search

Return ONLY a JSON object in this exact format:
{
  "found": true or false,
  "extracted_text": "the exact relevant text you found, or empty string if not found",
  "location": "section number or page reference where found, or empty string"
}

Rules:
- Return ONLY the JSON. No explanation, no preamble.
- If the text exists, copy it exactly — do not paraphrase.
- If not found, set found to false and extracted_text to empty string.
- Do not make judgements about whether the content is adequate."""


EXPLAINER_PROMPT = """You are an expert Clinical Protocol Auditor. A protocol has failed a specific regulatory rule.

You will be given the rule, the extracted text (or lack of it), and the specific failure reason.

Write a professional audit finding in this exact JSON format:
{
  "finding": "one clear sentence describing the specific gap found",
  "recommended_action": "one specific corrective action the sponsor should take"
}

Return ONLY the JSON. Be specific — cite section numbers if available. Be concise."""


# ── Evidence Scorer ────────────────────────────────────────────

# Pre-gate patterns — structural headings and ToC entries score 0 immediately
_TOC_LINE     = re.compile(r"^\d[\d\.]*\s+[A-Z][A-Z\s\(\)\/\-]{4,}\.{3,}\s*\d{1,3}\s*$")
_BARE_HEADING = re.compile(r"^\d[\d\.]*\s{1,4}[A-Z][A-Z\s\(\)\/\-]{4,}$")

# Gate 2: procedural depth vocabulary
_DEPTH_SIGNALS = [
    "shall", "must", "will", "required", "procedure", "process",
    "report", "monitor", "review", "assess", "document", "provide",
    "include", "ensure", "conduct", "maintain", "define", "specify",
    "describe", "establish", "designate", "confirm", "submit",
    # Scientific / statistical justification language
    "determined", "estimated", "calculated", "justified", "based on",
    "assuming", "powered to", "planned", "intended", "expected"
]

_MIN_LENGTH = 80


def score_evidence(extracted_text: str, rule: dict) -> dict:
    """
    Three-gate structural evidence scorer.

    Pre-gate: heading/ToC pattern → 0/3, no further checks.
    Gate 1 — Length:      text >= 80 chars
    Gate 2 — Depth:       contains at least one procedural depth signal
    Gate 3 — Specificity: matches at least one evidence_anchor term

    Returns score dict:
        total        int   0–3
        length       bool
        depth        bool
        specificity  bool
        anchor_hit   str   matching anchor term, or ""
        reason       str   human-readable summary for audit log
    """
    raw     = (extracted_text or "").strip()
    text    = raw.lower()
    anchors = rule.get("evidence_anchors", [])

    # Pre-gate: structural heading or ToC entry → 0/3 immediately
    if _TOC_LINE.match(raw) or _BARE_HEADING.match(raw):
        return {
            "total": 0, "length": False, "depth": False,
            "specificity": False, "anchor_hit": "",
            "reason": "Pre-gate: structural heading or ToC entry — score 0/3"
        }

    # Gate 1 — Length
    gate_length = len(text) >= _MIN_LENGTH

    # Gate 2 — Depth
    gate_depth = any(sig in text for sig in _DEPTH_SIGNALS)

    # Gate 3 — Specificity (rule-specific anchors from rules JSON)
    gate_specificity = False
    anchor_hit = ""
    for group in anchors:
        for term in group:
            if term.lower() in text:
                gate_specificity = True
                anchor_hit = term
                break
        if gate_specificity:
            break

    total = sum([gate_length, gate_depth, gate_specificity])

    passed = []
    failed = []
    if gate_length:      passed.append("length")
    else:                failed.append(f"length < {_MIN_LENGTH} chars")
    if gate_depth:       passed.append("depth")
    else:                failed.append("no procedural language")
    if gate_specificity: passed.append(f"anchor '{anchor_hit}'")
    else:                failed.append("no rule-specific anchor matched")

    parts = []
    if passed: parts.append("Passed: " + ", ".join(passed))
    if failed: parts.append("Failed: " + ", ".join(failed))

    return {
        "total": total, "length": gate_length,
        "depth": gate_depth, "specificity": gate_specificity,
        "anchor_hit": anchor_hit,
        "reason": " | ".join(parts)
    }


def scorer_verdict(score: dict) -> tuple:
    """
    Score 3/3 → PASS.
    Score < 3 → MANUAL_REVIEW (GxP-safe: never issue a false PASS).
    Returns (status: str, reason: str).
    """
    if score["total"] == 3:
        return "PASS", f"Evidence score 3/3 — {score['reason']}"

    missing = []
    if not score["length"]:
        missing.append("evidence too short")
    if not score["depth"]:
        missing.append("no procedural language")
    if not score["specificity"]:
        missing.append(
            "no rule-specific anchor matched — section may exist "
            "but content does not demonstrate compliance"
        )
    return (
        "MANUAL_REVIEW",
        f"Evidence score {score['total']}/3 — "
        + "; ".join(missing)
        + " — human review required"
    )


# ── Step 1: LLM Extractor ──────────────────────────────────────

def extract_evidence(rule: dict, protocol_content: str, llm) -> dict:
    """Ask LLM to extract relevant text from protocol for this rule."""
    instruction = rule.get("extract_instruction", rule.get("requirement", ""))
    prompt = f"""EXTRACTION INSTRUCTION:
{instruction}

PROTOCOL CONTENT:
{protocol_content[:2500]}

Extract the relevant text. Return JSON only."""

    try:
        response = llm.invoke([
            SystemMessage(content=EXTRACTOR_PROMPT),
            HumanMessage(content=prompt)
        ])
        text = response.content.strip()
        if "```" in text:
            text = re.sub(r'```(?:json)?', '', text).strip()
        brace_start = text.find('{')
        brace_end   = text.rfind('}')
        if brace_start != -1 and brace_end != -1:
            text = text[brace_start:brace_end + 1]
        return json.loads(text)
    except Exception as e:
        return {"found": False, "extracted_text": "", "location": "",
                "error": str(e)[:100]}


# ── Step 2: Code Judge + Evidence Scorer ──────────────────────

def judge_rule(rule: dict, extraction: dict) -> dict:
    """
    Deterministic PASS/FAIL/MANUAL_REVIEW — zero LLM.

    FAIL  → code alone (placeholder, explicit fail conditions, not found)
    PASS  → code candidate + Evidence Scorer 3/3
    MANUAL_REVIEW → everything else, including scorer < 3/3
    """
    rule_id    = rule.get("rule_id", "")
    severity   = rule.get("severity", "")
    fail_conds = rule.get("fail_conditions", [])
    pass_conds = rule.get("pass_conditions", [])

    found        = extraction.get("found", False)
    extracted    = (extraction.get("extracted_text") or "").lower().strip()
    raw_evidence = extraction.get("extracted_text", "")
    extract_err  = extraction.get("error")

    # Extraction error
    if extract_err:
        return {"rule_id": rule_id, "severity": severity,
                "status": "MANUAL_REVIEW",
                "reason": f"Extraction error: {extract_err}",
                "extracted_text": "", "location": "", "score": None}

    # Not found
    if not found or not extracted:
        not_found_fails = [c for c in fail_conds
                           if any(k in c for k in
                                  ("not found", "absent", "no mention", "missing"))]
        if not_found_fails:
            return {"rule_id": rule_id, "severity": severity,
                    "status": "FAIL", "reason": not_found_fails[0],
                    "extracted_text": "", "location": extraction.get("location", ""),
                    "score": None}
        return {"rule_id": rule_id, "severity": severity,
                "status": "MANUAL_REVIEW",
                "reason": "Content not found — human review required to confirm absence",
                "extracted_text": "", "location": "", "score": None}

    # Placeholder detection
    for signal in ["provide name", "tbd", "to be determined", "to be confirmed",
                   "insert name", "placeholder", "[name]", "n/a", "not applicable",
                   "error! bookmark", "bookmark not defined", "xxxx", "####"]:
        if signal in extracted:
            return {"rule_id": rule_id, "severity": severity,
                    "status": "FAIL",
                    "reason": f"Placeholder detected: '{signal}'",
                    "extracted_text": raw_evidence,
                    "location": extraction.get("location", ""), "score": None}

    # Explicit fail conditions
    for fail_cond in fail_conds:
        if any(k in fail_cond for k in ("not found", "absent", "no mention")):
            continue
        fail_signals = [w for w in fail_cond.lower().split() if len(w) > 4]
        if sum(1 for sig in fail_signals if sig in extracted) >= 2:
            return {"rule_id": rule_id, "severity": severity,
                    "status": "FAIL", "reason": fail_cond,
                    "extracted_text": raw_evidence,
                    "location": extraction.get("location", ""), "score": None}

    # Candidate PASS — run Evidence Scorer
    candidate = False

    for pass_cond in pass_conds:
        pass_signals = [w for w in pass_cond.lower().split() if len(w) > 5]
        if sum(1 for sig in pass_signals if sig in extracted) >= 1:
            candidate = True
            break

    if not candidate and found and len(extracted) > 50:
        if sum(1 for s in ["shall","must","will","required","procedure","report",
                            "monitor","review","assess","document","provide",
                            "include","ensure","conduct","maintain"]
               if s in extracted) >= 2:
            candidate = True

    if candidate:
        score  = score_evidence(raw_evidence, rule)
        status, reason = scorer_verdict(score)
        return {"rule_id": rule_id, "severity": severity,
                "status": status, "reason": reason,
                "extracted_text": raw_evidence,
                "location": extraction.get("location", ""),
                "score": score}

    return {"rule_id": rule_id, "severity": severity,
            "status": "MANUAL_REVIEW",
            "reason": "Content found but could not be deterministically assessed — human review required",
            "extracted_text": raw_evidence,
            "location": extraction.get("location", ""), "score": None}


# ── Step 3: LLM Explainer (FAIL only) ─────────────────────────

def explain_finding(rule: dict, judgment: dict, llm) -> dict:
    prompt = f"""RULE: {rule.get('requirement', '')}
REGULATORY REFERENCE: {rule.get('regulatory_reference', '')}
FAILURE REASON: {judgment.get('reason', '')}
EXTRACTED TEXT: {judgment.get('extracted_text', '(none found)')[:500]}
LOCATION: {judgment.get('location', 'Unknown')}

Write the audit finding. Return JSON only."""

    try:
        response = llm.invoke([
            SystemMessage(content=EXPLAINER_PROMPT),
            HumanMessage(content=prompt)
        ])
        text = response.content.strip()
        if "```" in text:
            text = re.sub(r'```(?:json)?', '', text).strip()
        brace_start = text.find('{')
        brace_end   = text.rfind('}')
        if brace_start != -1 and brace_end != -1:
            text = text[brace_start:brace_end + 1]
        explanation = json.loads(text)
    except Exception:
        explanation = {
            "finding": judgment.get("reason", "Protocol does not meet this requirement"),
            "recommended_action": f"Review protocol against {rule.get('regulatory_reference', '')}"
        }

    return {
        "rule_id":              rule.get("rule_id", ""),
        "status":               "FAIL",
        "severity":             rule.get("severity", ""),
        "category":             rule.get("category", ""),
        "finding":              explanation.get("finding", ""),
        "location":             judgment.get("location", ""),
        "evidence":             judgment.get("extracted_text", "")[:300],
        "failure_reason":       judgment.get("reason", ""),
        "regulatory_reference": rule.get("regulatory_reference", ""),
        "recommended_action":   explanation.get("recommended_action", "")
    }


# ── Rules loading ──────────────────────────────────────────────

def load_all_rules() -> list:
    rules = []
    for fname in ["ich_e6_r3_rules.json", "fda_ind_rules.json"]:
        path = RULES_DIR / fname
        if path.exists():
            data = json.loads(path.read_text())
            for r in data:
                if "evidence_anchors" not in r:
                    print(f"  WARNING: {r.get('rule_id')} missing evidence_anchors "
                          f"— PASS cannot be issued for this rule")
            rules.extend(data)
    rules.sort(key=lambda r: r.get("rule_id", ""))
    return rules


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


# ── Main audit function ────────────────────────────────────────

def run_audit_v2(api_key=None, rule_callback=None):
    print("=" * 64)
    print(" Clinical Protocol Auditor v2")
    print(" Extractor  →  Code Judge + Evidence Scorer  →  Explainer")
    print(" Sprint 7: Structural Evidence Scoring — fully deterministic")
    print("=" * 64)

    api_key = api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Set GROQ_API_KEY in .env file")

    print("\nLoading embeddings...")
    embeddings = get_embeddings()
    print("Connecting to ChromaDB...")
    protocol_store   = Chroma(collection_name="protocol",
                               embedding_function=embeddings,
                               persist_directory=CHROMA_DIR)
    regulatory_store = Chroma(collection_name="regulatory",
                               embedding_function=embeddings,
                               persist_directory=CHROMA_DIR)

    pc = protocol_store._collection.count()
    rc = regulatory_store._collection.count()
    print(f"  Protocol KB:   {pc} vectors")
    print(f"  Regulatory KB: {rc} vectors")
    if pc == 0 or rc == 0:
        raise ValueError("Run python src/ingest.py first")

    rules = load_all_rules()
    no_anchors = [r["rule_id"] for r in rules if not r.get("evidence_anchors")]
    print(f"  Rules loaded:  {len(rules)} | "
          f"Anchor coverage: {'all ✓' if not no_anchors else f'missing {no_anchors}'}\n")

    print("Retrieving protocol content upfront (deterministic)...")
    protocol_queries = [
        "study population eligibility inclusion exclusion criteria age",
        "primary endpoint secondary endpoint statistical analysis definition",
        "safety monitoring adverse events reporting DSMB oversight",
        "informed consent sponsor investigator responsibilities",
        "sample size randomization blinding study design",
        "protocol version title sponsor contact information",
        "data management delegation authority staff qualifications",
        "investigator brochure risk benefit monitoring plan stopping rules",
    ]
    seen_p, p_chunks = set(), []
    for q in protocol_queries:
        for doc in protocol_store.similarity_search(q, k=5):
            if doc.page_content not in seen_p:
                seen_p.add(doc.page_content)
                src = doc.metadata.get("source_file", "?")
                pg  = doc.metadata.get("page", "?")
                p_chunks.append(f"[{src} p.{pg}]\n{doc.page_content}")
    protocol_content = "\n\n---\n\n".join(p_chunks)
    print(f"  {len(p_chunks)} unique chunks — same content for all {len(rules)} rules\n")

    # Only two LLM instances — NO LLM in PASS decision path
    llm_extract = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=api_key,
                           temperature=0, max_tokens=300)
    llm_explain = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=api_key,
                           temperature=0, max_tokens=400)

    findings       = []
    pass_count     = fail_count = manual_count = scorer_demoted = 0

    print(f"  {'Rule':<22} {'Sev':<10} {'Score':<8} Result")
    print(f"  {'-'*22} {'-'*10} {'-'*8} {'-'*14}")

    for i, rule in enumerate(rules):
        rule_id  = rule.get("rule_id", "")
        severity = rule.get("severity", "")

        extraction = extract_evidence(rule, protocol_content, llm_extract)
        judgment   = judge_rule(rule, extraction)
        status     = judgment["status"]
        score      = judgment.get("score")

        score_col = f"{score['total']}/3" if score else "-"
        if score and status == "MANUAL_REVIEW":
            scorer_demoted += 1

        print(f"  {rule_id:<22} {severity:<10} {score_col:<8} {status}")

        if status == "FAIL":
            result = explain_finding(rule, judgment, llm_explain)
            fail_count += 1
        elif status == "PASS":
            result = {
                "rule_id": rule_id, "status": "PASS",
                "severity": severity, "category": rule.get("category", ""),
                "finding": "", "location": judgment.get("location", ""),
                "evidence": judgment.get("extracted_text", "")[:200],
                "failure_reason": "",
                "regulatory_reference": rule.get("regulatory_reference", ""),
                "recommended_action": "",
                "evidence_score": score
            }
            pass_count += 1
        else:
            result = {
                "rule_id": rule_id, "status": "MANUAL_REVIEW",
                "severity": severity, "category": rule.get("category", ""),
                "finding": judgment.get("reason", "Human review required"),
                "location": judgment.get("location", ""),
                "evidence": judgment.get("extracted_text", "")[:200],
                "failure_reason": judgment.get("reason", ""),
                "regulatory_reference": rule.get("regulatory_reference", ""),
                "recommended_action": "Regulatory Affairs team to review manually",
                **({"evidence_score": score} if score else {})
            }
            manual_count += 1

        findings.append(result)
        if rule_callback:
            rule_callback({"rule_id": rule_id, "severity": severity,
                           "status": status, "finding": result.get("finding", ""),
                           "index": i + 1, "total": len(rules)})

    print(f"\n{'='*64}")
    print(f"  {fail_count} FAIL  |  {pass_count} PASS  |  {manual_count} MANUAL_REVIEW")
    if scorer_demoted:
        print(f"  Scorer: {scorer_demoted} candidate PASS(es) scored < 3/3 → MANUAL_REVIEW")
    print(f"  Every PASS: score 3/3 — pre-gate ✓  length ✓  depth ✓  anchor ✓")
    print(f"{'='*64}\n")

    report = build_report(findings)
    return report, findings


def build_report(findings: list) -> str:
    fails   = [f for f in findings if f["status"] == "FAIL"]
    passes  = [f for f in findings if f["status"] == "PASS"]
    manuals = [f for f in findings if f["status"] == "MANUAL_REVIEW"]
    critical = len([f for f in fails if f.get("severity") == "Critical"])
    major    = len([f for f in fails if f.get("severity") == "Major"])
    minor    = len([f for f in fails if f.get("severity") == "Minor"])

    def get_cat(f):
        rid = f.get("rule_id", "")
        if "-A-" in rid: return "A"
        if "-B-" in rid: return "B"
        return "C"

    lines = [
        "CLINICAL PROTOCOL AUDIT REPORT",
        "=" * 40,
        "EXECUTIVE SUMMARY",
        f"Total Findings:   {len(fails)}",
        f"Critical:         {critical}",
        f"Major:            {major}",
        f"Minor:            {minor}",
        f"Manual Review:    {len(manuals)}",
        f"Verified PASSes:  {len(passes)} (each scored 3/3 on structural evidence)",
        ""
    ]

    for cat, label in [("A","INTERNAL INCONSISTENCIES"),
                        ("B","REGULATORY NON-COMPLIANCE"),
                        ("C","MISSING MANDATORY SECTIONS")]:
        cat_fails = [f for f in fails if get_cat(f) == cat]
        if not cat_fails: continue
        lines.append(f"CATEGORY {cat}: {label}")
        for i, f in enumerate(cat_fails, 1):
            lines += [f"Finding {cat}-{i:03d}",
                      f"  Severity:             {f.get('severity','')}",
                      f"  Location:             {f.get('location','Not specified')}",
                      f"  Issue:                {f.get('finding','')}",
                      f"  Regulatory Reference: {f.get('regulatory_reference','')}",
                      f"  Recommended Action:   {f.get('recommended_action','')}", ""]

    if manuals:
        lines += ["CATEGORY D: MANUAL REVIEW REQUIRED",
                  "Items below require Regulatory Affairs review.",
                  "Items marked [N/3] scored below 3/3 on structural evidence.", ""]
        for i, f in enumerate(manuals, 1):
            cat   = get_cat(f)
            label = {"A":"Internal Inconsistency","B":"Regulatory Non-Compliance",
                     "C":"Missing Section"}.get(cat, "")
            sc    = f.get("evidence_score")
            note  = f" [score {sc['total']}/3]" if sc else ""
            lines += [f"Manual Review {i:03d}{note}",
                      f"  Rule ID:              {f.get('rule_id','')}",
                      f"  Category:             {cat} — {label}",
                      f"  Severity:             {f.get('severity','')}",
                      f"  Reason:               {f.get('finding','')}",
                      f"  Regulatory Reference: {f.get('regulatory_reference','')}",
                      f"  Recommended Action:   {f.get('recommended_action','Regulatory Affairs team to review manually')}",
                      ""]
    return "\n".join(lines)


if __name__ == "__main__":
    api_key = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        report, findings = run_audit_v2(api_key)
        print("\n" + "=" * 64)
        print(report)
        (BASE_DIR / "audit_report_v2.txt").write_text(report)
        (BASE_DIR / "findings_log.json").write_text(json.dumps(findings, indent=2))
        print(f"Saved: audit_report_v2.txt + findings_log.json")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)
