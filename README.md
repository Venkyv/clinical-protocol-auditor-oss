# Clinical Protocol Auditor — Open Source Stack
## Apexon AI Solution Architecture Practice

Agentic clinical protocol auditor built on LangChain + ChromaDB + HuggingFace + Claude API.
Mirrors the AWS Bedrock POC but runs entirely on open-source infrastructure.

---

## Project Structure

```
clinical_auditor_oss/
├── data/
│   ├── protocol/          ← clinical_protocol_hepatitis_b.pdf
│   └── guidelines/        ← ich_e6_r3_gcp_guidelines.pdf
│                            fda_hepatitis_b_guidance.pdf
├── chroma_db/             ← auto-created on ingest
├── src/
│   ├── ingest.py          ← Sprint 1: document ingestion
│   └── audit.py           ← Sprint 1: LangChain audit agent
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Copy your PDF documents
```bash
# Copy from your S3 downloads or local files
cp clinical_protocol_hepatitis_b.pdf data/protocol/
cp ich_e6_r3_gcp_guidelines.pdf data/guidelines/
cp fda_hepatitis_b_guidance.pdf data/guidelines/
```

### 3. Set your API key
```bash
export ANTHROPIC_API_KEY=your_key_here
```
Or create a `.env` file:
```
ANTHROPIC_API_KEY=your_key_here
```

---

## Usage

### Step 1 — Ingest documents into ChromaDB
```bash
python src/ingest.py
```
Expected output:
- Protocol collection: ~1,500 vectors
- Regulatory collection: ~2,000 vectors
- Retrieval verification test passes

### Step 2 — Run the audit
```bash
python src/audit.py
```
The agent will:
1. Query the protocol KB for key sections
2. Query the regulatory KB for applicable requirements
3. Cross-reference findings
4. Output a structured audit report (Category A / B / C)

Report is also saved to `audit_report.txt`

---

## Tech Stack

| Component       | Technology                        |
|----------------|-----------------------------------|
| Orchestration  | LangChain (ReAct agent)           |
| LLM            | Claude 3.5 Sonnet (Anthropic API) |
| Embeddings     | HuggingFace all-MiniLM-L6-v2     |
| Vector store   | ChromaDB (local)                  |
| PDF parsing    | PyPDF                             |

---

## Sprint Roadmap

- **Sprint 1** (current): Ingestion + audit engine ← you are here
- **Sprint 2**: Dynamic rules store (JSON rules per regulation)
- **Sprint 3**: Flask dashboard + PDF export
- **Sprint 4**: GxP audit trail (PostgreSQL retrieval log)
- **Sprint 5**: Automated regulatory monitoring
- **Sprint 6**: Multi-region skill packs (UK/EU/US/APAC)
