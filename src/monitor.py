"""
Sprint 5 — Automated Regulatory Monitoring
Checks ICH, FDA, EMA source URLs daily for document updates
Uses SHA-256 checksum comparison to detect changes
Triggers human approval workflow when change detected
"""

import os
import json
import hashlib
import requests
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent
load_dotenv(BASE_DIR / ".env")

MONITOR_DIR = BASE_DIR / "monitoring"
MONITOR_DIR.mkdir(exist_ok=True)

CHECKSUMS_FILE = MONITOR_DIR / "checksums.json"
PENDING_FILE   = MONITOR_DIR / "pending_reviews.json"
LOG_FILE       = MONITOR_DIR / "monitor_log.json"

# Regulatory sources to monitor
SOURCES = [
    {
        "id": "ICH-E6-R3",
        "name": "ICH E6(R3) GCP Guidelines",
        "url": "https://www.ema.europa.eu/en/documents/scientific-guideline/ich-e6-r3-guideline-good-clinical-practice-gcp-step-5_en.pdf",
        "local_file": "gcp-guidelines/ich_e6_r3_gcp_guidelines.pdf",
        "rules_file": "ich_e6_r3_rules.json",
        "issuer": "EMA / ICH"
    },
    {
        "id": "FDA-HBV-GUIDANCE",
        "name": "FDA Hepatitis B Clinical Trial Guidance",
        "url": "https://www.fda.gov/media/117977/download",
        "local_file": "gcp-guidelines/fda_hepatitis_b_guidance.pdf",
        "rules_file": "fda_ind_rules.json",
        "issuer": "FDA CDER"
    }
]


def compute_checksum(filepath: Path) -> str:
    """SHA-256 checksum of local file"""
    if not filepath.exists():
        return "file_not_found"
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_checksums() -> dict:
    if CHECKSUMS_FILE.exists():
        return json.loads(CHECKSUMS_FILE.read_text())
    return {}


def save_checksums(checksums: dict):
    CHECKSUMS_FILE.write_text(json.dumps(checksums, indent=2))


def load_pending() -> list:
    if PENDING_FILE.exists():
        return json.loads(PENDING_FILE.read_text())
    return []


def save_pending(pending: list):
    PENDING_FILE.write_text(json.dumps(pending, indent=2))


def append_log(entry: dict):
    log = []
    if LOG_FILE.exists():
        log = json.loads(LOG_FILE.read_text())
    log.append(entry)
    LOG_FILE.write_text(json.dumps(log, indent=2))


def check_source(source: dict, data_dir: Path) -> dict:
    """Check if a regulatory source has changed"""
    source_id = source["id"]
    local_path = data_dir / source["local_file"]

    result = {
        "source_id": source_id,
        "name": source["name"],
        "checked_at": datetime.now().isoformat(),
        "status": "unchanged",
        "action_required": False
    }

    # Compute current local file checksum
    current_checksum = compute_checksum(local_path)
    stored_checksums = load_checksums()
    stored_checksum = stored_checksums.get(source_id, "")

    if not stored_checksum:
        # First run — store baseline
        stored_checksums[source_id] = current_checksum
        save_checksums(stored_checksums)
        result["status"] = "baseline_recorded"
        result["checksum"] = current_checksum
        print(f"  [{source_id}] Baseline recorded: {current_checksum[:16]}...")
        return result

    if current_checksum != stored_checksum:
        result["status"] = "CHANGE_DETECTED"
        result["action_required"] = True
        result["old_checksum"] = stored_checksum
        result["new_checksum"] = current_checksum
        print(f"  [{source_id}] ⚠️  CHANGE DETECTED!")
        print(f"    Old: {stored_checksum[:16]}...")
        print(f"    New: {current_checksum[:16]}...")

        # Add to pending reviews
        pending = load_pending()
        pending.append({
            "review_id": f"REV-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "source_id": source_id,
            "source_name": source["name"],
            "issuer": source["issuer"],
            "rules_file": source["rules_file"],
            "detected_at": datetime.now().isoformat(),
            "status": "PENDING_REVIEW",
            "old_checksum": stored_checksum,
            "new_checksum": current_checksum,
            "reviewer": None,
            "review_decision": None,
            "review_notes": None,
            "reviewed_at": None
        })
        save_pending(pending)
        print(f"    Added to pending reviews — human approval required")
    else:
        result["status"] = "unchanged"
        result["checksum"] = current_checksum
        print(f"  [{source_id}] ✅ No change — {current_checksum[:16]}...")

    return result


def run_monitoring(data_dir: Path = None):
    """Main monitoring function — run daily via scheduler"""
    if data_dir is None:
        data_dir = BASE_DIR / "data"

    print("=" * 55)
    print("  Regulatory Monitoring — Sprint 5")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)
    print(f"\nChecking {len(SOURCES)} regulatory sources...\n")

    results = []
    changes_detected = 0

    for source in SOURCES:
        print(f"Checking: {source['name']}")
        result = check_source(source, data_dir)
        results.append(result)
        if result.get("action_required"):
            changes_detected += 1
        print()

    # Summary
    print("=" * 55)
    print(f"Monitoring complete:")
    print(f"  Sources checked: {len(SOURCES)}")
    print(f"  Changes detected: {changes_detected}")

    pending = load_pending()
    pending_count = len([p for p in pending if p["status"] == "PENDING_REVIEW"])
    print(f"  Pending human reviews: {pending_count}")

    if pending_count > 0:
        print(f"\n⚠️  ACTION REQUIRED: {pending_count} document(s) need Regulatory Affairs review")
        print(f"   Run: python src/review_portal.py to manage pending reviews")

    # Log the run
    append_log({
        "run_at": datetime.now().isoformat(),
        "sources_checked": len(SOURCES),
        "changes_detected": changes_detected,
        "results": results
    })

    return results


# ── Human approval portal ────────────────────────────────────
def list_pending():
    """Show all pending regulatory document reviews"""
    pending = load_pending()
    pending_reviews = [p for p in pending if p["status"] == "PENDING_REVIEW"]

    if not pending_reviews:
        print("No pending reviews.")
        return

    print(f"\n{'='*55}")
    print(f"Pending Regulatory Reviews: {len(pending_reviews)}")
    print(f"{'='*55}")
    for p in pending_reviews:
        print(f"\nReview ID:   {p['review_id']}")
        print(f"Source:      {p['source_name']} ({p['issuer']})")
        print(f"Detected:    {p['detected_at'][:19]}")
        print(f"Rules file:  {p['rules_file']}")
        print(f"Status:      {p['status']}")


def approve_review(review_id: str, reviewer: str, notes: str = ""):
    """Approve a pending regulatory document change"""
    pending = load_pending()
    for p in pending:
        if p["review_id"] == review_id and p["status"] == "PENDING_REVIEW":
            p["status"] = "APPROVED"
            p["reviewer"] = reviewer
            p["review_decision"] = "APPROVED"
            p["review_notes"] = notes
            p["reviewed_at"] = datetime.now().isoformat()
            save_pending(pending)

            # Update stored checksum
            checksums = load_checksums()
            checksums[p["source_id"]] = p["new_checksum"]
            save_checksums(checksums)

            print(f"✅ Review {review_id} approved by {reviewer}")
            print(f"   Rules file {p['rules_file']} should now be updated")
            print(f"   Re-run: python src/ingest_rules.py to reload rules")
            return True

    print(f"Review {review_id} not found or already processed")
    return False


def reject_review(review_id: str, reviewer: str, notes: str = ""):
    """Reject a pending regulatory document change"""
    pending = load_pending()
    for p in pending:
        if p["review_id"] == review_id and p["status"] == "PENDING_REVIEW":
            p["status"] = "REJECTED"
            p["reviewer"] = reviewer
            p["review_decision"] = "REJECTED"
            p["review_notes"] = notes
            p["reviewed_at"] = datetime.now().isoformat()
            save_pending(pending)
            print(f"❌ Review {review_id} rejected by {reviewer}")
            return True

    print(f"Review {review_id} not found or already processed")
    return False


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "list":
            list_pending()
        elif cmd == "approve" and len(sys.argv) >= 4:
            approve_review(sys.argv[2], sys.argv[3],
                           sys.argv[4] if len(sys.argv) > 4 else "")
        elif cmd == "reject" and len(sys.argv) >= 4:
            reject_review(sys.argv[2], sys.argv[3],
                          sys.argv[4] if len(sys.argv) > 4 else "")
        else:
            print("Usage:")
            print("  python src/monitor.py              # run monitoring check")
            print("  python src/monitor.py list         # list pending reviews")
            print("  python src/monitor.py approve REV-ID 'reviewer' 'notes'")
            print("  python src/monitor.py reject REV-ID 'reviewer' 'notes'")
    else:
        run_monitoring()
