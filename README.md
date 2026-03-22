# Soreal

Soreal is a legal-document knowledge-graph RAG application. The maintained runtime is a Docker-first FastAPI service that ingests one legal document at a time, extracts entities and relationships with Ollama, stores them in SurrealDB, and answers grounded questions through a small web UI.

## What is maintained here

- Primary maintained runtime: `server.py`, `soreal_engine.py`, `static/index.html`, `docker-compose.yml`, `Dockerfile`, and `docker/wait_for_services.py`
- Primary docs: `README.md`, `AGENT.md`, `docs/architecture.md`, and `docs/file-guide.md`
- Companion artifacts: notebooks, presentation HTML files, sample inputs, and saved external documentation snapshots under `documentations/`

If a notebook or presentation file disagrees with the Python app or Docker setup, treat the Python app and root docs as the source of truth.

## Quickstart

### Prerequisites

- Docker Desktop or Docker Engine with Compose support
- Enough disk space for the Ollama model and Hugging Face embedding cache
- Patience on first startup; the image, Python dependencies, embeddings, and model may need to download

### 1. Optional: create a local `.env`

```bash
cp .env.example .env
```

For the Dockerized app, Compose already overrides the app container's internal service URLs. `.env` is mainly useful for local Python runs, notebooks, or LangSmith configuration.

### 2. Start the stack

```bash
docker compose up --build -d
docker compose ps
```

The app waits for SurrealDB and the Ollama HTTP API before starting Uvicorn.

### 3. Pull the default Ollama model

```bash
docker compose exec ollama ollama pull llama3.2:3b
```

If you change `OLLAMA_MODEL`, pull that model instead.

### 4. Open the app

- UI: `http://127.0.0.1:8001/`
- Health check: `curl http://127.0.0.1:8001/`
- Graph summary: `curl http://127.0.0.1:8001/api/status`

### 5. Stop or reset

```bash
docker compose down
docker compose down -v
curl -X POST http://127.0.0.1:8001/api/reset
```

`/api/reset` clears the current graph and in-memory session state without deleting Docker volumes.

## Runtime behavior at a glance

- `POST /api/analyze` resets the database, recreates the schema, ingests the submitted legal text, and streams progress back as SSE.
- `POST /api/ask` answers questions against the current graph and returns groundedness metadata plus source matches.
- The maintained runtime is effectively single-document. Each new analyze call replaces the existing graph.
- The active LangGraph `thread_id` is stored in memory inside `server.py`, so conversational state is process-local and disappears on restart.

For a deeper architecture walkthrough, see [docs/architecture.md](docs/architecture.md).

## Services and ports

| Service | Port | Role |
| --- | --- | --- |
| FastAPI app | `8001` | Serves the UI and API |
| SurrealDB | `8000` | Stores graph data and vector search state |
| Ollama | `11434` | Provides extraction and answer-generation LLM access |

## HTTP API overview

| Endpoint | Request | Response | Notes |
| --- | --- | --- | --- |
| `POST /api/analyze` | JSON body with `text` (`1..100000` chars) | `text/event-stream` | Emits `status`, `progress`, `complete`, `error`, then `[DONE]`. Each analyze run rebuilds the graph from scratch. |
| `POST /api/ask` | JSON body with `question` (`1..5000` chars) | JSON object | Returns `answer`, `groundedness`, `sources`, `tool_steps`, and `graph_facts_count`. |
| `GET /api/status` | No body | JSON object | Returns record counts for node and edge tables in the graph. |
| `POST /api/reset` | No body | JSON object | Clears graph data, reinitializes the schema, and resets the in-memory session. |

### Analyze stream contract

`POST /api/analyze` emits SSE payloads with these `type` values:

- `status`
- `progress`
- `complete`
- `error`
- `[DONE]`

The `complete` event includes `thread_id`, `entities`, `triplets`, `entity_details`, and `triplet_details`.

### Ask response shape

`POST /api/ask` returns:

- `answer`: final model answer
- `groundedness`: score and match metadata derived from retrieved graph facts
- `sources`: retrieved graph records used for traceability
- `tool_steps`: tool calls the agent attempted during answering
- `graph_facts_count`: number of graph facts considered for groundedness

Example:

```bash
curl -X POST http://127.0.0.1:8001/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What obligations does the Vendor have?"}'
```

## Configuration

Most host-facing defaults live in `.env.example`. The app container overrides service URLs to use Compose hostnames internally.

| Variable | Default | Notes |
| --- | --- | --- |
| `SURREAL_URL` | `ws://localhost:8000/rpc` | Host-facing default for local runs. The app container overrides this to `ws://surrealdb:8000/rpc`. |
| `SURREAL_USERNAME` | `root` | SurrealDB username |
| `SURREAL_PASSWORD` | `root` | SurrealDB password |
| `SURREAL_NAMESPACE` | `soreal` | SurrealDB namespace |
| `SURREAL_DATABASE` | `kg` | SurrealDB database |
| `OLLAMA_MODEL` | `llama3.2:3b` | Default model used for extraction and answering |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Host-facing default. The app container overrides this to `http://ollama:11434`. |
| `LANGCHAIN_TRACING_V2` | `false` | Enables LangSmith tracing when configured |
| `LANGCHAIN_PROJECT` | `soreal-kg-agent` | LangSmith project name |
| `LANGCHAIN_API_KEY` | empty | Optional LangSmith API key |
| `INGEST_CHUNK_SIZE` | `260` | Chunk size used by `ingest_text()` |
| `INGEST_CHUNK_OVERLAP` | `80` | Chunk overlap used by `ingest_text()` |
| `MAX_EXTRACTED_ENTITIES` | `24` | Cap on extracted entities across the ingest run |
| `MAX_TRIPLETS_PER_CHUNK` | `24` | Cap on extracted triplets |
| `OLLAMA_EXTRACTION_NUM_PREDICT` | `256` | Extraction-time generation limit for the Ollama client |
| `SERVICE_WAIT_TIMEOUT` | `180` | Optional timeout for `docker/wait_for_services.py` during container startup |

Fixed in code rather than env vars:

- Embedding model: `BAAI/bge-small-en-v1.5`
- Embedding dimension: `384`

## Optional local Python workflow

If you need to run the FastAPI app outside Docker:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
uvicorn server:app --reload --host 0.0.0.0 --port 8001
```

For this mode, you must also run SurrealDB and Ollama yourself and keep `.env` aligned with the services you started.

## Repo map

- [AGENT.md](AGENT.md): quick orientation for agents and contributors
- [docs/architecture.md](docs/architecture.md): runtime design, ingest/query flow, schema groups, and limitations
- [docs/file-guide.md](docs/file-guide.md): annotated repo and file-role guide
- `server.py`: FastAPI entrypoint and session handling
- `soreal_engine.py`: KG-RAG engine, schema, tools, and groundedness logic
- `static/index.html`: frontend UI
- `docker-compose.yml`: Docker-first local runtime
- `Dockerfile`: app image build
- `docker/wait_for_services.py`: startup readiness checks

## Companion materials

These files are useful, but they are not the primary runtime:

- `knowledge_graph_rag.ipynb`
- `vectorstores(1).ipynb`
- `architecture.html`
- `presentation.html`
- `documentations/`

Some companion files use alternative setup paths or older defaults. For example, notebook content may mention different model names such as `llama3.1:8b`. Prefer the maintained runtime defaults in the Python app and root docs unless a task explicitly targets those companion artifacts.

## Troubleshooting

### First startup is slow

This is expected. The first successful run may need to:

- build the app image
- install Python dependencies
- download the embedding model
- pull the Ollama model

### The app container keeps restarting

Inspect logs:

```bash
docker compose logs app --tail=200
```

Common causes:

- SurrealDB failed to start
- Ollama is still booting
- the configured Ollama model has not been pulled yet
- dependencies changed and the app image needs a rebuild

### Analyze or ask fails because the model is missing

Pull the configured model explicitly:

```bash
docker compose exec ollama ollama pull llama3.2:3b
```

If you changed `OLLAMA_MODEL`, use that value instead.

### Dockerized Ollama is slow on macOS

On macOS Docker Desktop, the Ollama container usually runs CPU-only. The stack still works, but latency can be noticeably higher than a native host Ollama setup.

### Notebook instructions do not match the app

Treat notebooks as companion materials. For app setup, use this README plus:

- [AGENT.md](AGENT.md)
- [docs/architecture.md](docs/architecture.md)
- [docs/file-guide.md](docs/file-guide.md)

### Start from scratch

```bash
docker compose down -v
docker compose up --build -d
docker compose exec ollama ollama pull llama3.2:3b
```
