# AGENT.md

## Project identity

Soreal is a Docker-first legal-document knowledge-graph RAG application. The maintained runtime is a FastAPI service backed by SurrealDB, Ollama, LangGraph, and Hugging Face embeddings. The app ingests one legal document at a time, rebuilds a graph from that document, and answers grounded questions through a single static web UI.

Primary runtime, in order of importance:

- `server.py`: FastAPI entrypoint, request validation, SSE streaming, process-local session state, and static file serving
- `soreal_engine.py`: schema definitions, embeddings, Ollama clients, LangGraph workflow, ingest/query tools, groundedness scoring, and high-level runtime API
- `static/index.html`: single-page frontend with embedded CSS and JavaScript
- `docker-compose.yml`: local multi-container runtime
- `Dockerfile`: app image build
- `docker/wait_for_services.py`: startup readiness checks for SurrealDB and Ollama

Companion artifacts exist in the repo, but they are not the primary deployment path:

- `knowledge_graph_rag.ipynb`
- `vectorstores(1).ipynb`
- `architecture.html`
- `presentation.html`
- files under `documentations/`

## Runtime shape

The architecture is intentionally simple:

1. The browser loads `static/index.html` from `/`.
2. The UI calls `POST /api/analyze`, `POST /api/ask`, `GET /api/status`, and `POST /api/reset`.
3. `server.py` passes the real work into `soreal_engine.py`.
4. `soreal_engine.py` talks to SurrealDB for storage/search and Ollama for extraction/answering.
5. Ingest progress is streamed back as SSE events; question answers return JSON.

The HTTP layer is thin. Most behavior changes belong in `soreal_engine.py`, but endpoint shape changes must stay synchronized with `server.py`, `static/index.html`, and the repo docs.

## Operational constraints and gotchas

- `POST /api/analyze` is destructive at the graph level. `ingest_document()` calls `reset_database()` and `init_schema()` before rebuilding the graph from the submitted text.
- The maintained runtime is effectively single-document. Each new analyze run replaces prior graph contents.
- Session state is process-local. `server.py` stores a single `thread_id` in a module-level `_session` dictionary, so restarts lose conversational continuity.
- The frontend is a single static HTML file. There is no separate frontend build step, bundler, or component tree.
- There are no automated tests in the repo today. Validation is manual.
- Ollama model availability is an operational dependency. The default model is `llama3.2:3b`, and it must be pulled before analyze/ask flows work.
- Notebook and presentation files are companion materials. They may use different defaults or older setup flows and should not be treated as the source of truth for the app runtime.

## Source of truth

- API contract: `server.py` and the client expectations in `static/index.html`
- Runtime defaults and ingest/query behavior: `soreal_engine.py`
- Container/runtime wiring: `docker-compose.yml`, `Dockerfile`, `docker/wait_for_services.py`
- Host-facing environment template: `.env.example`
- Human docs: `README.md`, `docs/architecture.md`, `docs/file-guide.md`
- Companion/reference material: notebooks, HTML presentation files, `documentations/`, `history.txt`

## Safe modification guidance

- If you change an endpoint, request payload, response payload, or SSE event shape, update all of:
  - `server.py`
  - `static/index.html`
  - `README.md`
  - `docs/architecture.md` if the runtime flow changed
- If you change environment variables or defaults, update all of:
  - `soreal_engine.py`
  - `.env.example`
  - `docker-compose.yml` if container overrides changed
  - `docker/wait_for_services.py` if startup behavior changed
  - `README.md`
- If you change the graph schema, tool behavior, or query pipeline, update `docs/architecture.md` so the grouped schema/runtime descriptions stay accurate.
- Avoid editing notebooks, `architecture.html`, `presentation.html`, or `documentations/` unless the task explicitly targets companion materials. They are not part of the maintained runtime.
- Do not assume the notebooks match current runtime defaults. The main code currently defaults to `llama3.2:3b`; notebook examples may differ.

## Quick validation checklist

- `docker compose config` still resolves cleanly after runtime config changes
- `README.md` still matches the actual commands, ports, and env vars
- The frontend still calls the endpoints documented in `server.py`
- Any changed API shape is reflected in both the web UI and repo docs
