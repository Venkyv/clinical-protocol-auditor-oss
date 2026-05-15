"""
Sprint 2 (Redesigned) — Audit Engine v2
Architecture: LLM as Extractor → Code as Judge → LLM as Explainer

Step 1: LLM EXTRACTS relevant text from protocol for each rule
Step 2: CODE JUDGES pass/fail deterministically against rule criteria
Step 3: LLM EXPLAINS (only on FAIL) — writes professional finding

This makes PASS/FAIL decisions deterministic and trustworthy.
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

BASE_DIR = Path(__file__).parent.parent
CHROMA_DIR = str(BASE_DIR / "chroma_db")
RULES_DIR = BASE_DIR / "rules"
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


# ── Step 1: LLM Extractor ──────────────────────────────────────

def extract_evidence(rule: dict, protocol_content: str, llm) -> dict:
    """Ask LLM to extract relevant text from protocol for this rule"""
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
        # Strip markdown code blocks if present
        if "```" in text:
            text = re.sub(r'```(?:json)?', '', text).strip()
        # Extract only the first JSON object if multiple present
        brace_start = text.find('{')
        brace_end = text.rfind('}')
        if brace_start != -1 and brace_end != -1:
            text = text[brace_start:brace_end+1]
        return json.loads(text)
    except Exception as e:
        return {"found": False, "extracted_text": "", "location": "", "error": str(e)[:100]}


# ── Step 2: Code Judge ─────────────────────────────────────────

def judge_rule(rule: dict, extraction: dict) -> dict:
    """
    Deterministic code-based PASS/FAIL judgment.
    Same input → same output every time.
    """
    rule_id   = rule.get("rule_id", "")
    severity  = rule.get("severity", "")
    fail_conds = rule.get("fail_conditions", [])
    pass_conds = rule.get("pass_conditions", [])

    found        = extraction.get("found", False)
    extracted    = (extraction.get("extracted_text") or "").lower().strip()
    extract_err  = extraction.get("error")

    # If extraction itself errored — flag for manual review
    if extract_err:
        return {
            "rule_id":  rule_id,
            "severity": severity,
            "status":   "MANUAL_REVIEW",
            "reason":   f"Extraction error: {extract_err}",
            "extracted_text": "",
            "location": ""
        }

    # ── Deterministic checks ───────────────────────────────────

    # Check 1: Not found at all → check if any fail condition matches "not found"
    if not found or not extracted:
        not_found_fails = [c for c in fail_conds if "not found" in c or "absent" in c or "no mention" in c or "missing" in c]
        if not_found_fails:
            return {
                "rule_id":  rule_id,
                "severity": severity,
                "status":   "FAIL",
                "reason":   not_found_fails[0],
                "extracted_text": "",
                "location": extraction.get("location", "")
            }
        # Found nothing but no explicit "not found" fail condition → manual review
        return {
            "rule_id":  rule_id,
            "severity": severity,
            "status":   "MANUAL_REVIEW",
            "reason":   "Content not found — human review required to confirm absence",
            "extracted_text": "",
            "location": ""
        }

    # Check 2: Placeholder / incomplete content detection
    placeholder_signals = [
        "provide name", "tbd", "to be determined", "to be confirmed",
        "insert name", "placeholder", "[name]", "n/a", "not applicable",
        "error! bookmark", "bookmark not defined", "xxxx", "####"
    ]
    for signal in placeholder_signals:
        if signal in extracted:
            return {
                "rule_id":  rule_id,
                "severity": severity,
                "status":   "FAIL",
                "reason":   f"Section contains placeholder or incomplete content: '{signal}' detected",
                "extracted_text": extraction.get("extracted_text", ""),
                "location": extraction.get("location", "")
            }

    # Check 3: Explicit fail conditions — check extracted text
    for fail_cond in fail_conds:
        # Skip "not found" conditions — already handled above
        if "not found" in fail_cond or "absent" in fail_cond or "no mention" in fail_cond:
            continue
        # Check for specific fail signals in extracted text
        fail_signals = [w for w in fail_cond.lower().split() if len(w) > 4]
        matches = sum(1 for sig in fail_signals if sig in extracted)
        if matches >= 2:  # At least 2 significant words match
            return {
                "rule_id":  rule_id,
                "severity": severity,
                "status":   "FAIL",
                "reason":   fail_cond,
                "extracted_text": extraction.get("extracted_text", ""),
                "location": extraction.get("location", "")
            }

    # Check 4: Pass conditions — positive confirmation
    # Use single strong domain keyword match (more lenient than fail check)
    for pass_cond in pass_conds:
        # Extract meaningful domain keywords (longer words = more specific)
        pass_signals = [w for w in pass_cond.lower().split() if len(w) > 5]
        matches = sum(1 for sig in pass_signals if sig in extracted)
        if matches >= 1:
            return {
                "rule_id":  rule_id,
                "severity": severity,
                "status":   "PASS",
                "reason":   pass_cond,
                "extracted_text": extraction.get("extracted_text", ""),
                "location": extraction.get("location", "")
            }

    # Check 5: Content found, no placeholders, no explicit fail signals detected
    # Lean toward PASS for content-found cases where we have good extracted text
    if found and len(extracted) > 50:
        # Check if extracted text has substantive content (not just a section header)
        substantive_signals = ["shall", "must", "will", "required", "procedure",
                               "report", "monitor", "review", "assess", "document",
                               "provide", "include", "ensure", "conduct", "maintain"]
        substantive_matches = sum(1 for s in substantive_signals if s in extracted)
        if substantive_matches >= 2:
            return {
                "rule_id":  rule_id,
                "severity": severity,
                "status":   "PASS",
                "reason":   "Substantive content found addressing this requirement",
                "extracted_text": extraction.get("extracted_text", ""),
                "location": extraction.get("location", "")
            }

    # Check 6: Content found but genuinely ambiguous → manual review
    return {
        "rule_id":  rule_id,
        "severity": severity,
        "status":   "MANUAL_REVIEW",
        "reason":   "Content found but could not be deterministically assessed — human review required",
        "extracted_text": extraction.get("extracted_text", ""),
        "location": extraction.get("location", "")
    }


# ── Step 3: LLM Explainer (FAIL only) ─────────────────────────

def explain_finding(rule: dict, judgment: dict, llm) -> dict:
    """Generate professional finding text only for FAILs"""
    prompt = f"""RULE: {rule.get('requirement','')}
REGULATORY REFERENCE: {rule.get('regulatory_reference','')}
FAILURE REASON: {judgment.get('reason','')}
EXTRACTED TEXT: {judgment.get('extracted_text','(none found)')[:500]}
LOCATION: {judgment.get('location','Unknown')}

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
        brace_end = text.rfind('}')
        if brace_start != -1 and brace_end != -1:
            text = text[brace_start:brace_end+1]
        explanation = json.loads(text)
    except Exception:
        explanation = {
            "finding": judgment.get("reason", "Protocol does not meet this requirement"),
            "recommended_action": f"Review protocol against {rule.get('regulatory_reference','')}"
        }

    return {
        "rule_id":               rule.get("rule_id", ""),
        "status":                "FAIL",
        "severity":              rule.get("severity", ""),
        "category":              rule.get("category", ""),
        "finding":               explanation.get("finding", ""),
        "location":              judgment.get("location", ""),
        "evidence":              judgment.get("extracted_text", "")[:300],
        "failure_reason":        judgment.get("reason", ""),
        "regulatory_reference":  rule.get("regulatory_reference", ""),
        "recommended_action":    explanation.get("recommended_action", "")
    }


# ── Rules loading ──────────────────────────────────────────────

def load_all_rules() -> list:
    """Load rules from JSON files — sorted by rule_id for deterministic order"""
    rules = []
    for fname in ["ich_e6_r3_rules.json", "fda_ind_rules.json"]:
        path = RULES_DIR / fname
        if path.exists():
            rules.extend(json.loads(path.read_text()))
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
    print("=" * 55)
    print("  Clinical Protocol Auditor v2 — Redesigned")
    print("  Extractor → Code Judge → Explainer")
    print("=" * 55)

    api_key = api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Set GROQ_API_KEY in .env file")

    print("\nLoading embeddings...")
    embeddings = get_embeddings()

    print("Connecting to ChromaDB...")
    protocol_store  = Chroma(collection_name="protocol",   embedding_function=embeddings, persist_directory=CHROMA_DIR)
    regulatory_store = Chroma(collection_name="regulatory", embedding_function=embeddings, persist_directory=CHROMA_DIR)

    pc = protocol_store._collection.count()
    rc = regulatory_store._collection.count()
    print(f"  Protocol KB:   {pc} vectors")
    print(f"  Regulatory KB: {rc} vectors")

    if pc == 0 or rc == 0:
        raise ValueError("Run python src/ingest.py first")

    # Load rules
    rules = load_all_rules()
    print(f"  Rules loaded:  {len(rules)} rules\n")

    # Retrieve ALL protocol content once upfront — deterministic
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
    print(f"  Protocol chunks: {len(p_chunks)} unique")
    print(f"  Same content used for ALL {len(rules)} rule checks\n")

    # LLM — use 70b for extraction, same for explanation
    # Two models — 8b for extraction (high volume, low tokens)
    #              70b for explanation (low volume, quality matters)
    llm_extract = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=api_key,
        temperature=0,
        max_tokens=300
    )
    llm_explain = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=api_key,
        temperature=0,
        max_tokens=400
    )

    findings = []
    pass_count = 0
    fail_count = 0
    manual_count = 0

    print(f"Checking {len(rules)} rules — Extractor → Code Judge → Explainer\n")

    for i, rule in enumerate(rules):
        rule_id  = rule.get("rule_id", "")
        severity = rule.get("severity", "")
        print(f"  [{i+1}/{len(rules)}] {rule_id} [{severity}]", end=" ")

        # ── Step 1: Extract ──────────────────────────────
        extraction = extract_evidence(rule, protocol_content, llm_extract)

        # ── Step 2: Code Judge ───────────────────────────
        judgment = judge_rule(rule, extraction)
        status   = judgment["status"]

        # ── Step 3: Explain (FAIL only) ──────────────────
        if status == "FAIL":
            result = explain_finding(rule, judgment, llm_explain)
            fail_count += 1
        elif status == "PASS":
            result = {
                "rule_id":              rule_id,
                "status":               "PASS",
                "severity":             severity,
                "category":             rule.get("category", ""),
                "finding":              "",
                "location":             judgment.get("location", ""),
                "evidence":             judgment.get("extracted_text", "")[:200],
                "failure_reason":       "",
                "regulatory_reference": rule.get("regulatory_reference", ""),
                "recommended_action":   ""
            }
            pass_count += 1
        else:  # MANUAL_REVIEW
            result = {
                "rule_id":              rule_id,
                "status":               "MANUAL_REVIEW",
                "severity":             severity,
                "category":             rule.get("category", ""),
                "finding":              judgment.get("reason", "Human review required"),
                "location":             judgment.get("location", ""),
                "evidence":             judgment.get("extracted_text", "")[:200],
                "failure_reason":       judgment.get("reason", ""),
                "regulatory_reference": rule.get("regulatory_reference", ""),
                "recommended_action":   "Regulatory Affairs team to review manually"
            }
            manual_count += 1

        print(f"→ {status}")
        findings.append(result)

        if rule_callback:
            rule_callback({
                "rule_id":  rule_id,
                "severity": severity,
                "status":   status,
                "finding":  result.get("finding", ""),
                "index":    i + 1,
                "total":    len(rules)
            })

    print(f"\nResults: {fail_count} FAIL | {pass_count} PASS | {manual_count} MANUAL_REVIEW")
    print("Generating structured report...")

    report = build_report(findings)
    return report, findings


def build_report(findings: list) -> str:
    """Build structured report from findings — no LLM, pure code"""
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

    cat_names = {
        "A": "INTERNAL INCONSISTENCIES",
        "B": "REGULATORY NON-COMPLIANCE",
        "C": "MISSING MANDATORY SECTIONS"
    }

    lines = [
        "CLINICAL PROTOCOL AUDIT REPORT",
        "=" * 32,
        "EXECUTIVE SUMMARY",
        f"Total Findings: {len(fails)}",
        f"Critical: {critical} findings",
        f"Major: {major} findings",
        f"Minor: {minor} findings",
        f"Manual Review Required: {len(manuals)}",
        ""
    ]

    for cat in ["A", "B", "C"]:
        cat_fails = [f for f in fails if get_cat(f) == cat]
        if not cat_fails:
            continue
        lines.append(f"CATEGORY {cat}: {cat_names[cat]}")
        for i, f in enumerate(cat_fails, 1):
            lines += [
                f"Finding {cat}-{i:03d}",
                f"Severity: {f.get('severity','')}",
                f"Location: {f.get('location','Not specified')}",
                f"Issue: {f.get('finding','')}",
                f"Regulatory Reference: {f.get('regulatory_reference','')}",
                f"Recommended Action: {f.get('recommended_action','')}",
                ""
            ]

    # MANUAL REVIEW section — grouped by category with full details
    if manuals:
        lines += ["CATEGORY D: MANUAL REVIEW REQUIRED",
                  "These rules require human assessment — content found but could not be",
                  "deterministically assessed by automated code. A Regulatory Affairs specialist",
                  "must review each item below.", ""]

        def get_cat_m(f):
            rid = f.get("rule_id","")
            if "-A-" in rid: return "A"
            if "-B-" in rid: return "B"
            return "C"

        for i, f in enumerate(manuals, 1):
            cat = get_cat_m(f)
            cat_label = {"A":"Internal Inconsistency","B":"Regulatory Non-Compliance","C":"Missing Section"}.get(cat,"")
            lines += [
                f"Manual Review {i:03d}",
                f"Rule ID: {f.get('rule_id','')}",
                f"Category: {cat} — {cat_label}",
                f"Severity: {f.get('severity','')}",
                f"Reason: {f.get('finding','')}",
                f"Regulatory Reference: {f.get('regulatory_reference','')}",
                f"Recommended Action: {f.get('recommended_action','Regulatory Affairs team to review manually')}",
                ""
            ]

    return "\n".join(lines)


if __name__ == "__main__":
    api_key = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        report, findings = run_audit_v2(api_key)
        print("\n" + "=" * 55)
        print("AUDIT REPORT")
        print("=" * 55)
        print(report)
        (BASE_DIR / "audit_report_v2.txt").write_text(report)
        (BASE_DIR / "findings_log.json").write_text(json.dumps(findings, indent=2))
        print(f"\nSaved: audit_report_v2.txt + findings_log.json")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)
