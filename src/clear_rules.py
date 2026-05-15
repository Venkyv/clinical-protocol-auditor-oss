"""
Utility script — clears the audit_rules ChromaDB collection
Run this before re-ingesting rules to avoid duplicates
"""
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
CHROMA_DIR = str(BASE_DIR / "chroma_db")

print("Loading embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

print("Deleting audit_rules collection...")
store = Chroma(
    collection_name="audit_rules",
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR
)
count_before = store._collection.count()
print(f"  Rules before: {count_before}")
store.delete_collection()
print("  Collection deleted ✅")
print("\nNow run: python src/ingest_rules.py")
