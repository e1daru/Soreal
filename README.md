# Soreal — Legal Document Analysis for SMBs

> **Knowledge Graph RAG** powered by SurrealDB + LangGraph + Ollama.
> Upload any legal document, ask plain-English questions, and get grounded answers with **source references** and a **confidence score** — all running locally on your laptop.

---

## ✨ Features

- **Document ingestion** — paste text or upload a `.txt` file; the pipeline chunks, embeds and stores everything in a local SurrealDB knowledge graph
- **Entity & relation extraction** — parties, clauses, obligations, rights, risks, payment terms, key dates… extracted by a local LLM (Ollama)
- **Graph-traversal RAG** — questions are answered via vector similarity search + one-hop graph walk
- **Source references** — every answer shows which knowledge-graph nodes it was grounded on
- **Confidence score** — a groundedness score (0–100 %) tells you how much the answer follows the document vs. the LLM's own knowledge
- **100 % local** — no data leaves your laptop; no cloud API required

---

## 🖥 Quick Start

### Prerequisites

| Dependency | Install |
|---|---|
| Python ≥ 3.11 | [python.org](https://www.python.org/downloads/) |
| [SurrealDB](https://surrealdb.com/install) | `brew install surrealdb/tap/surreal` (macOS) |
| [Ollama](https://ollama.com) | `brew install ollama` (macOS) |

### 1 — Start the services

```bash
# Terminal 1 — SurrealDB (in-memory, no persistence needed for demo)
surreal start --user root --pass root memory

# Terminal 2 — Ollama
ollama serve
ollama pull llama3.1:8b   # ~5 GB, one-time download
```

### 2 — Install Python dependencies

```bash
# Clone the repo and enter the directory
cd Soreal

# Create a virtual environment (recommended)
python -m venv .venv && source .venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### 3 — Configure environment (optional)

```bash
cp .env.example .env
# Edit .env if your SurrealDB / Ollama are on non-default ports
```

### 4 — Launch the web app

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

---

## 🗺 Architecture

```
Browser (index.html)
    │
    ▼
Flask app (app.py)
    │
    ▼
agent.py
    ├── SurrealDB (knowledge graph + vector store)
    ├── Ollama / llama3.1:8b (NER, relation extraction, Q&A)
    ├── HuggingFace BGE embeddings (384-dim, local)
    └── LangGraph agent (ingest → extract → triplets → load → query)
```

### API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/init` | Connect to SurrealDB + Ollama, load embeddings |
| `POST` | `/ingest` | Ingest a document (JSON `{text, session_id}`) |
| `POST` | `/ask` | Ask a question (JSON `{question, session_id}`) |
| `GET` | `/summary` | Knowledge graph table counts |
| `GET` | `/health` | Liveness check |

---

## 📓 Jupyter Notebook

The original research notebook (`knowledge_graph_rag.ipynb`) contains the full pipeline with step-by-step cells for exploration and debugging.
See `plan-runNotebookLocally.prompt.md` for instructions on running it.
