"""
Sprint 1 — Audit engine (LangChain modern approach)
Uses ChromaDB retrieval + Claude API for reasoning
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

BASE_DIR = Path(__file__).parent.parent
CHROMA_DIR = str(BASE_DIR / "chroma_db")

# Load environment variables from .env file
load_dotenv(BASE_DIR / ".env")

SYSTEM_PROMPT = """You are an expert Clinical Protocol Auditor with deep knowledge of ICH E6(R3) GCP guidelines and FDA regulatory requirements.

You will be given content retrieved from two knowledge bases:
1. PROTOCOL KB — the clinical trial protocol document
2. REGULATORY KB — ICH E6(R3) GCP guidelines and FDA regulatory guidance

Cross-reference the protocol against regulatory requirements and produce a structured audit report.

Your report MUST follow this exact format:

CLINICAL PROTOCOL AUDIT REPORT
================================
EXECUTIVE SUMMARY
Total Findings: [N]
Critical: [N] findings
Major: [N] findings
Minor: [N] findings

CATEGORY A: INTERNAL INCONSISTENCIES
Finding A-001
Severity: [Critical/Major/Minor]
Location: [Section reference]
Issue: [Description]
Regulatory Reference: [ICH/FDA citation]
Recommended Action: [Action]

CATEGORY B: REGULATORY NON-COMPLIANCE
Finding B-001
Severity: [Critical/Major/Minor]
Location: [Section reference]
Issue: [Description]
Regulatory Reference: [ICH/FDA citation]
Recommended Action: [Action]

CATEGORY C: MISSING MANDATORY SECTIONS
Finding C-001
Severity: [Critical/Major/Minor]
Location: [Protocol structure]
Issue: [Description]
Regulatory Reference: [ICH/FDA citation]
Recommended Action: [Action]

Generate 7-9 total findings. Cite specific section numbers. Never fabricate findings."""


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def retrieve(store, queries, k=4):
    results, seen = [], set()
    for q in queries:
        for doc in store.similarity_search(q, k=k):
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                src = doc.metadata.get("source_file", "unknown")
                pg = doc.metadata.get("page", "?")
                results.append(f"[{src} p.{pg}]\n{doc.page_content}")
    return "\n\n---\n\n".join(results)


def run_audit(api_key=None):
    print("="*55)
    print("  Clinical Protocol Auditor — Open Source Stack")
    print("  ChromaDB + HuggingFace + Claude API")
    print("="*55)

    api_key = api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Set GROQ_API_KEY environment variable")

    print("\nLoading embeddings...")
    embeddings = get_embeddings()

    print("Connecting to ChromaDB...")
    p_store = Chroma(collection_name="protocol", embedding_function=embeddings, persist_directory=CHROMA_DIR)
    r_store = Chroma(collection_name="regulatory", embedding_function=embeddings, persist_directory=CHROMA_DIR)

    pc = p_store._collection.count()
    rc = r_store._collection.count()
    print(f"  Protocol KB: {pc} vectors")
    print(f"  Regulatory KB: {rc} vectors")

    if pc == 0 or rc == 0:
        raise ValueError("Empty collections. Run: python src/ingest.py first")

    print("\nQuerying knowledge bases...")
    p_content = retrieve(p_store, [
        "study population age range inclusion exclusion criteria",
        "primary endpoint secondary endpoint statistical analysis",
        "safety monitoring adverse events reporting DSMB",
        "informed consent process procedures",
        "sample size justification randomization blinding",
        "investigator qualifications sponsor responsibilities",
        "protocol version data management procedures",
    ])

    r_content = retrieve(r_store, [
        "ICH E6 R3 protocol mandatory sections requirements",
        "sponsor medical monitor responsibilities oversight",
        "investigator qualifications training documentation",
        "safety reporting adverse events requirements",
        "informed consent ICH E6 requirements",
        "delegation of authority staff documentation",
        "monitoring plan procedures requirements",
    ])

    print(f"  Protocol content: {len(p_content)} chars")
    print(f"  Regulatory content: {len(r_content)} chars")

    prompt = f"""Perform a complete clinical protocol audit using the content below.

PROTOCOL CONTENT:
{p_content[:8000]}

REGULATORY REQUIREMENTS (ICH E6(R3) + FDA):
{r_content[:8000]}

Produce the complete structured audit report with Category A, B, and C findings."""

    print("\nClaude reasoning across both knowledge bases...")
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=api_key,  # loaded from .env or environment
        temperature=0,
        max_tokens=4000
    )
    response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)])
    return response.content


if __name__ == "__main__":
    api_key = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        report = run_audit(api_key)
        print("\n" + "="*55)
        print("AUDIT REPORT")
        print("="*55)
        print(report)
        out = BASE_DIR / "audit_report.txt"
        out.write_text(report)
        print(f"\nSaved: {out}")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
