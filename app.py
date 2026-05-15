"""
Sprint 3 — Flask Dashboard
Clinical Protocol Auditor — Open Source Stack
One-click audit with live streaming + PDF export
"""

import os
import sys
import json
import hashlib
import uuid
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, send_file, Response, stream_with_context

# Add src to path
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / "src"))
load_dotenv(BASE_DIR / ".env")

from audit_v2 import run_audit_v2, get_embeddings, load_all_rules

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB max upload

UPLOAD_DIR = BASE_DIR / "uploads"
REPORTS_DIR = BASE_DIR / "reports"
UPLOAD_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)


def get_file_checksum(filepath: Path) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()[:16]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/audit", methods=["POST"])
def run_audit():
    """Run full audit and stream progress back to browser"""

    def generate():
        try:
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                yield f"data: {json.dumps({'type': 'error', 'message': 'GROQ_API_KEY not set in .env'})}\n\n"
                return

            # Handle file upload or use default
            protocol_file = None
            if "protocol" in request.files and request.files["protocol"].filename:
                f = request.files["protocol"]
                protocol_file = UPLOAD_DIR / f.filename
                f.save(str(protocol_file))
                yield f"data: {json.dumps({'type': 'progress', 'step': 1, 'message': f'Protocol uploaded: {f.filename}'})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'progress', 'step': 1, 'message': 'Using pre-loaded protocol: clinical_protocol_hepatitis_b.pdf'})}\n\n"

            yield f"data: {json.dumps({'type': 'progress', 'step': 2, 'message': 'Loading ChromaDB knowledge bases...'})}\n\n"
            yield f"data: {json.dumps({'type': 'progress', 'step': 3, 'message': 'Retrieving 18 audit rules from rules store...'})}\n\n"
            yield f"data: {json.dumps({'type': 'progress', 'step': 4, 'message': 'Agent cross-referencing protocol against ICH E6(R3) and FDA guidance...'})}\n\n"

            # Run audit — rule results streamed immediately via callback
            import queue, threading

            result_queue = queue.Queue()
            audit_result = {}

            def stream_rule(r):
                result_queue.put(('rule', r))

            def run_in_thread():
                try:
                    report, findings = run_audit_v2(api_key, rule_callback=stream_rule)
                    audit_result['report'] = report
                    audit_result['findings'] = findings
                except Exception as e:
                    audit_result['error'] = str(e)
                finally:
                    result_queue.put(('done', None))

            thread = threading.Thread(target=run_in_thread)
            thread.start()

            # Stream rule results as they arrive
            while True:
                try:
                    msg_type, data = result_queue.get(timeout=120)
                    if msg_type == 'rule':
                        yield f"data: {json.dumps({'type': 'rule_result', **data})}\n\n"
                    elif msg_type == 'done':
                        break
                except queue.Empty:
                    break

            thread.join()

            if 'error' in audit_result:
                yield f"data: {json.dumps({'type': 'error', 'message': audit_result['error']})}\n\n"
                return

            report = audit_result.get('report', '')
            findings = audit_result.get('findings', [])

            # Build audit metadata
            fails   = [f for f in findings if f.get("status") == "FAIL"]
            passes  = [f for f in findings if f.get("status") == "PASS"]
            manuals = [f for f in findings if f.get("status") == "MANUAL_REVIEW"]
            critical = len([f for f in fails if f.get("severity") == "Critical"])
            major = len([f for f in fails if f.get("severity") == "Major"])
            minor = len([f for f in fails if f.get("severity") == "Minor"])

            audit_id = f"AUD-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            timestamp = datetime.now().strftime("%d %B %Y %H:%M UTC")

            # Save GxP audit log
            audit_log = {
                "audit_id": audit_id,
                "timestamp": datetime.now().isoformat(),
                "operator": "venkatesh.reddy@apexon.com",
                "tool": "Apexon Clinical Protocol Auditor — Open Source Stack",
                "stack": "LangChain + ChromaDB + HuggingFace + Groq llama-3.3-70b",
                "protocol": {
                    "filename": protocol_file.name if protocol_file else "clinical_protocol_hepatitis_b.pdf",
                    "checksum": "sha256:" + (get_file_checksum(protocol_file) if protocol_file else "preloaded")
                },
                "rules_version": {
                    "ich_file": "ich_e6_r3_rules.json",
                    "fda_file": "fda_ind_rules.json",
                    "total_rules": 18
                },
                "retrieval": {
                    "method": "upfront_bulk — all chunks retrieved once, same for all rule checks",
                    "protocol_chunks": 33,
                    "regulatory_chunks": 19
                },
                "summary": {
                    "total_rules_checked": len(findings),
                    "fail": len(fails),
                    "pass": len(passes),
                    "manual_review": len(manuals),
                    "critical": critical,
                    "major": major,
                    "minor": minor
                },
                "findings": findings
            }

            log_path = REPORTS_DIR / f"{audit_id}_log.json"
            log_path.write_text(json.dumps(audit_log, indent=2))

            # Also save to root findings_log.json for easy access
            (BASE_DIR / "findings_log.json").write_text(json.dumps(audit_log, indent=2))

            # Save report text
            report_path = REPORTS_DIR / f"{audit_id}_report.txt"
            report_path.write_text(report)
            (BASE_DIR / "audit_report_v2.txt").write_text(report)

            yield f"data: {json.dumps({'type': 'progress', 'step': 5, 'message': f'Audit complete — {len(fails)} findings identified'})}\n\n"

            yield f"data: {json.dumps({'type': 'done', 'report': report, 'findings': findings, 'audit_id': audit_id, 'timestamp': timestamp, 'summary': {'total': len(fails), 'critical': critical, 'major': major, 'minor': minor, 'manual': len(manuals)}, 'rules_checked': len(findings), 'passes': len(passes)})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.route("/export/<audit_id>")
def export_pdf(audit_id):
    """Export PDF built entirely from findings_log.json — structured data only"""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.colors import HexColor
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        from reportlab.lib.units import mm
        import io

        log_path = REPORTS_DIR / f"{audit_id}_log.json"
        if not log_path.exists():
            return jsonify({"error": "Audit log not found. Run audit first."}), 404

        log_data = json.loads(log_path.read_text())
        all_findings = log_data.get("findings", [])
        fails  = [f for f in all_findings if f.get("status") == "FAIL"]
        passes = [f for f in all_findings if f.get("status") == "PASS"]

        def get_cat(f):
            rid = f.get("rule_id", "")
            if "-A-" in rid: return "A"
            if "-B-" in rid: return "B"
            return "C"

        cat_a = [f for f in fails if get_cat(f) == "A"]
        cat_b = [f for f in fails if get_cat(f) == "B"]
        cat_c = [f for f in fails if get_cat(f) == "C"]
        critical = [f for f in fails if f.get("severity") == "Critical"]
        major_f  = [f for f in fails if f.get("severity") == "Major"]
        minor_f  = [f for f in fails if f.get("severity") == "Minor"]

        # Colours
        NAVY  = HexColor("#0D2240"); TEAL  = HexColor("#0E7490")
        CRIT  = HexColor("#DC2626"); MAJ   = HexColor("#D97706")
        MIN   = HexColor("#059669"); GRAY  = HexColor("#F1F5F9")
        LGRAY = HexColor("#F8FAFC"); DARK  = HexColor("#374151")
        MID   = HexColor("#6B7280"); WHITE = HexColor("#FFFFFF")

        def sev_color(s):
            s = (s or "").lower()
            if "critical" in s: return CRIT
            if "major"    in s: return MAJ
            return MIN

        # Styles
        def ps(name, **kw): return ParagraphStyle(name, **kw)
        meta_s  = ps("m",  fontSize=8,  textColor=MID,  fontName="Helvetica", spaceAfter=2, leading=11)
        title_s = ps("t",  fontSize=22, textColor=NAVY, fontName="Helvetica-Bold", spaceAfter=4, leading=26)
        sub_s   = ps("s",  fontSize=9,  textColor=MID,  fontName="Helvetica", spaceAfter=8)
        h1_s    = ps("h1", fontSize=13, textColor=TEAL, fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=5)
        body_s  = ps("b",  fontSize=10, textColor=DARK, fontName="Helvetica", spaceAfter=3, leading=14)
        ref_s   = ps("r",  fontSize=9,  textColor=MID,  fontName="Helvetica-Oblique", spaceAfter=2, leftIndent=10, leading=12)
        ev_s    = ps("e",  fontSize=9,  textColor=MID,  fontName="Helvetica-Oblique", spaceAfter=3, leftIndent=10, leading=12)
        h2_s    = ps("h2", fontSize=11, textColor=NAVY, fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=3)

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4,
            rightMargin=20*mm, leftMargin=20*mm,
            topMargin=18*mm, bottomMargin=18*mm)
        story = []

        # Header
        story.append(Paragraph(
            "APEXON — AI Solution Architecture  |  Venkatesh Reddy Valluri, AI Solution Architect",
            meta_s))
        story.append(Paragraph("Clinical Protocol Audit Report", title_s))
        ts_raw = log_data.get("timestamp", "")[:19].replace("T", " ")
        story.append(Paragraph(
            f"Audit ID: {audit_id}  ·  Generated: {ts_raw}  ·  "
            f"Protocol: {log_data.get('protocol', {}).get('filename', 'N/A')}",
            sub_s))
        story.append(HRFlowable(width="100%", thickness=2, color=TEAL, spaceAfter=6))

        # Summary table — all counts from structured data
        summary_data = [
            ["Rules Checked", "Total Findings", "Critical", "Major", "Minor", "Pass"],
            [str(len(all_findings)), str(len(fails)),
             str(len(critical)), str(len(major_f)), str(len(minor_f)), str(len(passes))]
        ]
        t = Table(summary_data, colWidths=[32*mm]*6)
        t.setStyle(TableStyle([
            ("BACKGROUND",   (0,0),(-1,0), NAVY),
            ("TEXTCOLOR",    (0,0),(-1,0), WHITE),
            ("FONTNAME",     (0,0),(-1,0), "Helvetica-Bold"),
            ("FONTSIZE",     (0,0),(-1,-1), 10),
            ("ALIGN",        (0,0),(-1,-1), "CENTER"),
            ("VALIGN",       (0,0),(-1,-1), "MIDDLE"),
            ("ROWHEIGHT",    (0,0),(-1,-1), 7*mm),
            ("BACKGROUND",   (0,1),(-1,1), LGRAY),
            ("BACKGROUND",   (2,1),(2,1),  CRIT),
            ("TEXTCOLOR",    (2,1),(2,1),  WHITE),
            ("FONTNAME",     (2,1),(2,1),  "Helvetica-Bold"),
            ("GRID",         (0,0),(-1,-1), 0.5, HexColor("#E5E7EB")),
        ]))
        story.append(t)
        story.append(Spacer(1, 6*mm))

        # Render each category from JSON
        for cat_id, cat_name, cat_findings, cat_color in [
            ("A", "Internal Inconsistencies",   cat_a, CRIT),
            ("B", "Regulatory Non-Compliance",  cat_b, MAJ),
            ("C", "Missing Mandatory Sections", cat_c, MIN),
        ]:
            if not cat_findings:
                story.append(Paragraph(
                    f"Category {cat_id}: {cat_name} — No findings", h1_s))
                continue

            story.append(Paragraph(
                f"Category {cat_id}: {cat_name} ({len(cat_findings)} findings)", h1_s))
            story.append(HRFlowable(width="100%", thickness=1,
                color=cat_color, spaceAfter=4))

            for i, f in enumerate(cat_findings, 1):
                sev = f.get("severity", "Minor")
                rid = f.get("rule_id", "?")
                loc = (f.get("location") or "").strip()
                finding = (f.get("finding") or "").strip()
                ref  = (f.get("regulatory_reference") or "").strip()
                ev   = (f.get("evidence") or "").strip()

                # Coloured header bar
                sc = sev_color(sev)
                hdr = Table([[
                    Paragraph(f"<b>{rid}</b>",
                        ps("fh", fontSize=9, textColor=WHITE, fontName="Helvetica-Bold")),
                    Paragraph(f"<b>{sev.upper()}</b>",
                        ps("sh", fontSize=9, textColor=WHITE, fontName="Helvetica-Bold")),
                    Paragraph(f"Finding {cat_id}-{i:03d}",
                        ps("fn", fontSize=9, textColor=WHITE, fontName="Helvetica")),
                ]], colWidths=[58*mm, 28*mm, None])
                hdr.setStyle(TableStyle([
                    ("BACKGROUND",  (0,0),(-1,0), sc),
                    ("VALIGN",      (0,0),(-1,0), "MIDDLE"),
                    ("ROWHEIGHT",   (0,0),(-1,0), 7*mm),
                    ("LEFTPADDING", (0,0),(-1,0), 6),
                ]))
                story.append(hdr)

                if loc:     story.append(Paragraph(f"<b>Location:</b> {loc}", body_s))
                if finding: story.append(Paragraph(f"<b>Issue:</b> {finding}", body_s))
                if ev:      story.append(Paragraph(f'Evidence: "{ev[:250]}"', ev_s))
                if ref:     story.append(Paragraph(f"Regulatory Reference: {ref}", ref_s))
                story.append(Spacer(1, 3*mm))

        # MANUAL REVIEW section
        manuals_pdf = [f for f in all_findings if f.get("status") == "MANUAL_REVIEW"]
        if manuals_pdf:
            story.append(Paragraph("Manual Review Required", h1_s))
            story.append(HRFlowable(width="100%", thickness=1, color=MAJ, spaceAfter=4))
            story.append(Paragraph(
                "The following rules require human assessment by a Regulatory Affairs specialist.",
                body_s))
            story.append(Spacer(1, 3*mm))

            def get_cat_pdf(f):
                rid = f.get("rule_id","")
                if "-A-" in rid: return "A — Internal Inconsistency"
                if "-B-" in rid: return "B — Regulatory Non-Compliance"
                return "C — Missing Section"

            for i, f in enumerate(manuals_pdf, 1):
                sev  = f.get("severity","")
                rid  = f.get("rule_id","?")
                reason = (f.get("finding") or "").strip()
                ref  = (f.get("regulatory_reference") or "").strip()

                hdr = Table([[
                    Paragraph(f"<b>{rid}</b>",
                        ps("fh", fontSize=9, textColor=WHITE, fontName="Helvetica-Bold")),
                    Paragraph(f"<b>{sev.upper()}</b>",
                        ps("sh", fontSize=9, textColor=WHITE, fontName="Helvetica-Bold")),
                    Paragraph(f"Manual Review {i:03d} · {get_cat_pdf(f)}",
                        ps("fn", fontSize=9, textColor=WHITE, fontName="Helvetica")),
                ]], colWidths=[58*mm, 28*mm, None])
                hdr.setStyle(TableStyle([
                    ("BACKGROUND",  (0,0),(-1,0), MAJ),
                    ("VALIGN",      (0,0),(-1,0), "MIDDLE"),
                    ("ROWHEIGHT",   (0,0),(-1,0), 7*mm),
                    ("LEFTPADDING", (0,0),(-1,0), 6),
                ]))
                story.append(hdr)
                if reason: story.append(Paragraph(f"<b>Reason:</b> {reason}", body_s))
                if ref:    story.append(Paragraph(f"Regulatory Reference: {ref}", ref_s))
                story.append(Paragraph("Recommended Action: Regulatory Affairs specialist to review manually", ref_s))
                story.append(Spacer(1, 3*mm))

        # GxP Audit Trail
        story.append(Spacer(1, 6*mm))
        story.append(HRFlowable(width="100%", thickness=1,
            color=HexColor("#E5E7EB"), spaceAfter=4))
        story.append(Paragraph("GxP Audit Trail", h2_s))
        rv = log_data.get("rules_version", {})
        audit_meta = [
            ["Audit ID",       audit_id],
            ["Timestamp",      log_data.get("timestamp", "")],
            ["Protocol",       log_data.get("protocol", {}).get("filename", "N/A")],
            ["Checksum",       log_data.get("protocol", {}).get("checksum", "N/A")],
            ["ICH Rules",      rv.get("ich_file", "N/A")],
            ["FDA Rules",      rv.get("fda_file", "N/A")],
            ["Rules Checked",  str(len(all_findings))],
            ["Data Source",    "findings_log.json (structured JSON — not LLM narrative)"],
        ]
        at = Table(audit_meta, colWidths=[45*mm, None])
        at.setStyle(TableStyle([
            ("FONTNAME",      (0,0),(0,-1), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0),(-1,-1), 9),
            ("TEXTCOLOR",     (0,0),(0,-1), NAVY),
            ("TEXTCOLOR",     (1,0),(1,-1), DARK),
            ("ROWBACKGROUNDS",(0,0),(-1,-1), [LGRAY, WHITE]),
            ("GRID",          (0,0),(-1,-1), 0.3, HexColor("#E5E7EB")),
            ("LEFTPADDING",   (0,0),(-1,-1), 6),
            ("TOPPADDING",    (0,0),(-1,-1), 4),
            ("BOTTOMPADDING", (0,0),(-1,-1), 4),
        ]))
        story.append(at)

        # Footer
        story.append(Spacer(1, 8*mm))
        story.append(HRFlowable(width="100%", thickness=1,
            color=HexColor("#E5E7EB"), spaceAfter=4))
        story.append(Paragraph(
            f"Generated by Venkatesh Reddy Valluri · Apexon Clinical Protocol Auditor "
            f"(Open Source Stack) | LangChain + ChromaDB + Groq | {audit_id}",
            meta_s))

        doc.build(story)
        buffer.seek(0)
        return send_file(buffer, mimetype="application/pdf",
            as_attachment=True,
            download_name=f"{audit_id}_audit_report.pdf")

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    print("\n" + "="*55)
    print("  Apexon Clinical Protocol Auditor")
    print("  Open Source Stack — Flask Dashboard")
    print("  LangChain + ChromaDB + Groq + HuggingFace")
    print("="*55)
    print("  Open: http://localhost:5000")
    print("="*55 + "\n")
    app.run(debug=False, port=5000, threaded=True)
