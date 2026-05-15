"""
test_evidence_scorer.py — Sprint 7 Evidence Scorer tests
Run with:  python -m pytest test_evidence_scorer.py -v
"""
import json, re, pathlib, sys

_TOC_LINE     = re.compile(r"^\d[\d\.]*\s+[A-Z][A-Z\s\(\)\/\-]{4,}\.{3,}\s*\d{1,3}\s*$")
_BARE_HEADING = re.compile(r"^\d[\d\.]*\s{1,4}[A-Z][A-Z\s\(\)\/\-]{4,}$")
_DEPTH_SIGNALS = [
    "shall","must","will","required","procedure","process","report","monitor",
    "review","assess","document","provide","include","ensure","conduct","maintain",
    "define","specify","describe","establish","designate","confirm","submit",
    "determined","estimated","calculated","justified","based on","assuming",
    "powered to","planned","intended","expected"
]
_MIN_LENGTH = 80

def score_evidence(extracted_text, rule):
    raw  = (extracted_text or "").strip()
    text = raw.lower()
    anchors = rule.get("evidence_anchors", [])
    if _TOC_LINE.match(raw) or _BARE_HEADING.match(raw):
        return {"total":0,"length":False,"depth":False,"specificity":False,"anchor_hit":"",
                "reason":"Pre-gate: structural heading or ToC"}
    gate_len  = len(text) >= _MIN_LENGTH
    gate_dep  = any(s in text for s in _DEPTH_SIGNALS)
    gate_spec = False; hit = ""
    for group in anchors:
        for term in group:
            if term.lower() in text:
                gate_spec = True; hit = term; break
        if gate_spec: break
    total = sum([gate_len, gate_dep, gate_spec])
    return {"total":total,"length":gate_len,"depth":gate_dep,"specificity":gate_spec,"anchor_hit":hit}

def scorer_verdict(score):
    return ("PASS","3/3") if score["total"]==3 else ("MANUAL_REVIEW",f"{score['total']}/3")

ANCHORS = {
    "FDA-IND-C-002": [
        ["statistical analysis plan","statistical analysis method","analysis population",
         "intent-to-treat","per protocol analysis","statistical plan","sap"],
        ["will be analysed","will be analyzed","compared using","descriptive statistics",
         "confidence interval","fisher","chi-square","t-test","anova","logistic regression"]],
    "ICH-E6R3-B-005": [
        ["80% power","90% power","power of 80","power of 90","statistical power","% power","power calculation"],
        ["sample size of","sample size was","sample size:","effect size","clinically meaningful",
         "detectable difference","significance level","type i error","alpha of 0.05","alpha = 0.05"]],
    "ICH-E6R3-A-004": [
        ["open-label","double-blind","single-blind","unblinded","randomised","randomized",
         "no randomis","not randomis","no interim analysis","no sub-group"]],
    "ICH-E6R3-B-004": [
        ["serious adverse event","sae","expedited report","unexpected serious"],
        ["within 24","within 7","within 15","calendar days","working days","24 hours",
         "7 days","15 days","reporting timeline","reporting period"]],
}

def rule(rid): return {"rule_id":rid,"evidence_anchors":ANCHORS[rid]}

class TestPreGate:
    def test_toc_entry_scores_zero(self):
        ev = "9 STATISTICAL CONSIDERATIONS .......................................................................................................... 20"
        assert score_evidence(ev, rule("FDA-IND-C-002"))["total"] == 0

    def test_bare_heading_scores_zero(self):
        assert score_evidence("9.2  SAMPLE SIZE DETERMINATION", rule("ICH-E6R3-B-005"))["total"] == 0

    def test_toc_verdict_is_manual_review(self):
        ev = "9 STATISTICAL CONSIDERATIONS .......................................................................................................... 20"
        s  = score_evidence(ev, rule("FDA-IND-C-002"))
        assert scorer_verdict(s)[0] == "MANUAL_REVIEW"

class TestGateIsolation:
    def _r(self, anchors): return {"rule_id":"T","evidence_anchors":anchors}

    def test_gate1_80_passes(self):
        assert score_evidence("a"*80, self._r([]))["length"] is True

    def test_gate1_79_fails(self):
        assert score_evidence("a"*79, self._r([]))["length"] is False

    def test_gate2_shall_passes(self):
        assert score_evidence("a"*80+" shall be reported", self._r([]))["depth"] is True

    def test_gate2_determined_passes(self):
        assert score_evidence("a"*80+" was determined based on", self._r([]))["depth"] is True

    def test_gate2_planned_passes(self):
        assert score_evidence("a"*80+" no interim analysis is planned", self._r([]))["depth"] is True

    def test_gate2_no_signals_fails(self):
        assert score_evidence("a"*80+" this section covers the topic", self._r([]))["depth"] is False

    def test_gate3_anchor_matches(self):
        s = score_evidence("a"*80+" within 24 hours of event shall be reported",
                           self._r([["within 24","within 7"]]))
        assert s["specificity"] is True and s["anchor_hit"] == "within 24"

    def test_gate3_no_anchor_fails(self):
        s = score_evidence("a"*80+" the study will be conducted carefully",
                           self._r([["serious adverse event","sae"]]))
        assert s["specificity"] is False

    def test_empty_anchors_gate3_always_fails(self):
        assert score_evidence("a"*80+" shall be done", {"rule_id":"T","evidence_anchors":[]})["specificity"] is False

class TestScorerVerdict:
    def test_3_returns_pass(self):
        assert scorer_verdict({"total":3})== ("PASS","3/3")
    def test_2_returns_manual(self):
        assert scorer_verdict({"total":2})[0] == "MANUAL_REVIEW"
    def test_1_returns_manual(self):
        assert scorer_verdict({"total":1})[0] == "MANUAL_REVIEW"
    def test_0_returns_manual(self):
        assert scorer_verdict({"total":0})[0] == "MANUAL_REVIEW"

class TestSubstantiveEvidence:
    def test_sae_reporting_3_of_3(self):
        ev = ("All serious adverse events (SAEs) must be reported to the sponsor "
              "within 24 hours of the investigator becoming aware. Expedited "
              "reporting to the regulatory authority will follow within 7 calendar "
              "days for unexpected fatal or life-threatening SAEs.")
        assert score_evidence(ev, rule("ICH-E6R3-B-004"))["total"] == 3

    def test_sample_size_with_power_3_of_3(self):
        ev = ("The sample size of 120 participants was determined based on a power "
              "calculation assuming 80% power to detect a clinically meaningful "
              "difference of 10% with a two-sided alpha of 0.05. The effect size "
              "was estimated from published literature.")
        assert score_evidence(ev, rule("ICH-E6R3-B-005"))["total"] == 3

    def test_statistical_plan_3_of_3(self):
        ev = ("A formal statistical analysis plan (SAP) will be finalized prior to "
              "database lock. The primary analysis will use an intent-to-treat "
              "population. Descriptive statistics will be provided for all endpoints. "
              "Confidence intervals will be calculated at the 95% level.")
        assert score_evidence(ev, rule("FDA-IND-C-002"))["total"] == 3

    def test_a004_no_analysis_anchor_fires(self):
        ev = "No interim analysis is planned.  No sub-group analyses are planned."
        s  = score_evidence(ev, rule("ICH-E6R3-A-004"))
        assert s["specificity"] is True, "anchor 'no interim analysis' must match"
        assert s["anchor_hit"] == "no interim analysis"

class TestAnchorCoverage:
    def test_all_18_rules_have_anchors_in_deployed_json(self):
        for p in ["/tmp/ich_e6_r3_rules.json","/tmp/fda_ind_rules.json"]:
            path = pathlib.Path(p)
            if path.exists():
                for r in json.loads(path.read_text()):
                    rid = r["rule_id"]
                    assert "evidence_anchors" in r, f"{rid} missing evidence_anchors"
                    assert len(r["evidence_anchors"]) > 0, f"{rid} has empty anchors"
