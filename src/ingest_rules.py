"""
Sprint 2 — Rules store ingestion
Loads structured audit rules from JSON into ChromaDB
One rule per chunk — deterministic retrieval
"""

import json
import sys
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

BASE_DIR = Path(__file__).parent.parent
CHROMA_DIR = str(BASE_DIR / "chroma_db")
RULES_DIR = BASE_DIR / "rules"

RULES_FILES = {
    "ich_e6_r3_rules.json": "ICH E6(R3) GCP Guidelines",
    "fda_ind_rules.json": "FDA IND Protocol Guidance"
}


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def rule_to_document(rule: dict, source_file: str) -> Document:
    """Convert a rule dict to a LangChain Document for embedding"""
    # Rich text representation for semantic search
    content = f"""Rule ID: {rule['rule_id']}
Category: {rule['category']}
Severity: {rule['severity']}
Requirement: {rule['requirement']}
Extract Instruction: {rule.get('extract_instruction', rule.get('check_instruction', ''))}
Fail Conditions: {'; '.join(rule.get('fail_conditions', []))}
Pass Conditions: {'; '.join(rule.get('pass_conditions', []))}
Regulatory Reference: {rule['regulatory_reference']}"""

    metadata = {
        "rule_id": rule["rule_id"],
        "category": rule["category"],
        "severity": rule["severity"],
        "regulatory_reference": rule["regulatory_reference"],
        "effective_date": rule.get("effective_date", ""),
        "source_file": source_file,
        "tags": ",".join(rule.get("tags", []))
    }

    return Document(page_content=content, metadata=metadata)


def ingest_rules(embeddings):
    all_docs = []

    for filename, source_name in RULES_FILES.items():
        rules_path = RULES_DIR / filename
        if not rules_path.exists():
            print(f"  Warning: {filename} not found — skipping")
            continue

        with open(rules_path, "r") as f:
            rules = json.load(f)

        print(f"  Loading {len(rules)} rules from {filename}")
        for rule in rules:
            doc = rule_to_document(rule, source_name)
            all_docs.append(doc)

    if not all_docs:
        print("No rules found. Check rules/ directory.")
        sys.exit(1)

    print(f"\n  Total rules to index: {len(all_docs)}")

    # Store in ChromaDB as separate collection
    vectorstore = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        collection_name="audit_rules",
        persist_directory=CHROMA_DIR
    )

    count = vectorstore._collection.count()
    print(f"  Rules collection ready — {count} rules indexed")
    return count


def verify_rules_retrieval(embeddings):
    """Test that rules are retrievable by category and severity"""
    print("\nVerifying rules retrieval...")

    store = Chroma(
        collection_name="audit_rules",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )

    total = store._collection.count()
    print(f"  Total rules in store: {total}")

    # Test retrieval by concept
    test_queries = [
        "medical monitor designation sponsor responsibilities",
        "investigator qualifications training records",
        "primary endpoint consistency protocol sections"
    ]

    for query in test_queries:
        results = store.similarity_search(query, k=2)
        print(f"\n  Query: '{query[:50]}'")
        for r in results:
            print(f"    → {r.metadata['rule_id']} [{r.metadata['severity']}] — {r.metadata['regulatory_reference'][:50]}")


def main():
    print("=" * 55)
    print("  Clinical Protocol Auditor — Sprint 2")
    print("  Rules Store Ingestion")
    print("=" * 55)

    embeddings = get_embeddings()

    print("\nIngesting audit rules into ChromaDB...")
    count = ingest_rules(embeddings)

    verify_rules_retrieval(embeddings)

    print(f"\nSprint 2 rules store complete — {count} rules ready")
    print("Run next: python src/audit_v2.py")


if __name__ == "__main__":
    main()
