"""
Sprint 1 — Document ingestion pipeline
Loads protocol + regulatory PDFs into ChromaDB
using HuggingFace all-MiniLM-L6-v2 embeddings
"""

import os
import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = str(BASE_DIR / "chroma_db")

COLLECTIONS = {
    "protocol": {
        "path": DATA_DIR / "protocol",
        "description": "Clinical trial protocol documents"
    },
    "regulatory": {
        "path": DATA_DIR / "guidelines",
        "description": "ICH E6(R3) GCP guidelines and FDA regulatory guidance"
    }
}

def get_embeddings():
    print("Loading HuggingFace embeddings (all-MiniLM-L6-v2)...")
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

def load_pdfs(folder: Path) -> list:
    docs = []
    pdf_files = list(folder.glob("*.pdf"))
    if not pdf_files:
        print(f"  No PDFs found in {folder}")
        return docs
    for pdf in pdf_files:
        print(f"  Loading: {pdf.name}")
        loader = PyPDFLoader(str(pdf))
        pages = loader.load()
        # Add source filename to metadata
        for page in pages:
            page.metadata["source_file"] = pdf.name
            page.metadata["collection"] = folder.name
        docs.extend(pages)
        print(f"  Loaded {len(pages)} pages from {pdf.name}")
    return docs

def chunk_documents(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    print(f"  Split into {len(chunks)} chunks")
    return chunks

def ingest_collection(name: str, config: dict, embeddings) -> int:
    print(f"\nIngesting collection: {name}")
    print(f"  Source: {config['path']}")

    docs = load_pdfs(config["path"])
    if not docs:
        print(f"  Skipping — no documents found")
        return 0

    chunks = chunk_documents(docs)

    print(f"  Creating ChromaDB collection: {name}")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=name,
        persist_directory=CHROMA_DIR
    )

    count = vectorstore._collection.count()
    print(f"  Collection '{name}' ready — {count} vectors stored")
    return count

def verify_retrieval(embeddings):
    """Quick retrieval test to confirm KB is working"""
    print("\nVerifying retrieval quality...")

    for collection_name in ["protocol", "regulatory"]:
        try:
            store = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=CHROMA_DIR
            )
            count = store._collection.count()
            if count == 0:
                print(f"  [{collection_name}] Empty — run ingest first")
                continue

            # Test query
            query = "study population age range inclusion criteria"
            results = store.similarity_search(query, k=3)
            print(f"\n  [{collection_name}] {count} vectors — test query: '{query}'")
            for i, doc in enumerate(results):
                source = doc.metadata.get("source_file", "unknown")
                page = doc.metadata.get("page", "?")
                preview = doc.page_content[:120].replace("\n", " ")
                print(f"    Result {i+1} | {source} p.{page} | {preview}...")

        except Exception as e:
            print(f"  [{collection_name}] Error: {e}")

def main():
    print("=" * 55)
    print("  Clinical Protocol Auditor — Open Source Stack")
    print("  Sprint 1: Document Ingestion Pipeline")
    print("=" * 55)

    # Check data directories have PDFs
    missing = []
    for name, config in COLLECTIONS.items():
        pdfs = list(config["path"].glob("*.pdf"))
        if not pdfs:
            missing.append(f"{name}: {config['path']}")

    if missing:
        print("\nMissing PDFs — please copy documents to:")
        for m in missing:
            print(f"  {m}")
        print("\nExpected files:")
        print("  data/protocol/clinical_protocol_hepatitis_b.pdf")
        print("  data/guidelines/ich_e6_r3_gcp_guidelines.pdf")
        print("  data/guidelines/fda_hepatitis_b_guidance.pdf")
        print("\nThen re-run: python src/ingest.py")
        sys.exit(1)

    embeddings = get_embeddings()
    total = 0
    for name, config in COLLECTIONS.items():
        total += ingest_collection(name, config, embeddings)

    print(f"\nIngestion complete — {total} total vectors across all collections")

    if "--verify" in sys.argv or True:
        verify_retrieval(embeddings)

    print("\nSprint 1 complete. Run next: python src/audit.py")

if __name__ == "__main__":
    main()
