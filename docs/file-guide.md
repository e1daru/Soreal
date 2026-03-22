# Soreal File Guide

## Summary

This guide covers the files a maintainer or agent is most likely to inspect. It distinguishes between the maintained runtime, supporting configuration, and companion/reference artifacts so future edits land in the right place.

## Maintained runtime files

| Path | Purpose | Status | When to edit |
| --- | --- | --- | --- |
| `server.py` | FastAPI entrypoint. Validates requests, exposes the HTTP API, streams ingest progress, stores the current `thread_id`, and serves the frontend. | Runtime source of truth | Edit when endpoint shapes, request validation, session handling, or routing change. |
| `soreal_engine.py` | Core KG-RAG engine. Defines config defaults, schema, embeddings, Ollama clients, LangGraph workflow, ingest/query tools, and groundedness scoring. | Runtime source of truth | Edit when schema, ingest logic, retrieval, tool behavior, or model/config defaults change. |
| `static/index.html` | Single-page frontend with embedded HTML, CSS, and JavaScript for analyze, ask, status, and reset flows. | Runtime source of truth | Edit when the UI, client-side API usage, or displayed response fields change. |

## Configuration and container files

| Path | Purpose | Status | When to edit |
| --- | --- | --- | --- |
| `docker-compose.yml` | Defines the three-service local runtime: app, SurrealDB, and Ollama, plus named volumes and host ports. | Runtime configuration | Edit when container wiring, service images, ports, or environment overrides change. |
| `Dockerfile` | Builds the app image, installs Python dependencies, copies runtime files, adds a health check, and starts Uvicorn after readiness checks. | Runtime configuration | Edit when the app image, dependency installation, or startup command changes. |
| `docker/wait_for_services.py` | Waits for SurrealDB TCP reachability and the Ollama API before the app starts. | Runtime helper | Edit when startup readiness behavior or timeout handling changes. |
| `requirements.txt` | Python package list for the maintained app container and local Python runtime. | Runtime dependency manifest | Edit when the app gains, removes, or upgrades direct Python dependencies. |
| `.env.example` | Host-facing environment template for local runs, notebooks, and optional overrides. | Runtime configuration template | Edit when committed env vars or default values change. |
| `.dockerignore` | Excludes notebooks, reference assets, caches, and local config from the app build context. | Build hygiene | Edit when build inputs change and new files should be included or excluded. |
| `.gitignore` | Ignores local environment files and notebook checkpoints. | Repo hygiene | Edit when new local-only artifacts should stay untracked. |

## Documentation files

| Path | Purpose | Status | When to edit |
| --- | --- | --- | --- |
| `README.md` | Main human-facing onboarding doc for the maintained runtime. | Documentation source of truth | Edit when setup, commands, API contracts, or runtime defaults change. |
| `AGENT.md` | Fast orientation for AI agents and future contributors. | Documentation source of truth | Edit when repo boundaries, source-of-truth files, or operational constraints change. |
| `docs/architecture.md` | Runtime design, ingest/query flow, schema groupings, and system limitations. | Documentation source of truth | Edit when architecture or schema behavior changes. |
| `docs/file-guide.md` | Repo map and file-role guide. | Documentation source of truth | Edit when important repo files are added, removed, or reclassified. |

## Companion notebooks and HTML presentations

| Path | Purpose | Status | When to edit |
| --- | --- | --- | --- |
| `knowledge_graph_rag.ipynb` | Notebook version of the legal KG-RAG workflow that the Python runtime was derived from. | Companion demo/prototype | Edit only when the task explicitly targets notebook workflows or research demos. |
| `vectorstores(1).ipynb` | Separate notebook focused on SurrealDB vector-store usage. | Companion example | Edit only when the vector-store notebook itself needs maintenance. |
| `architecture.html` | Standalone HTML visualization of the system workflow. | Companion presentation asset | Edit only when presentation collateral needs to change. |
| `presentation.html` | Standalone presentation/demo deck for the project. | Companion presentation asset | Edit only when presentation collateral needs to change. |
| `plan-runNotebookLocally.prompt.md` | Saved planning note for restructuring notebook setup. | Companion planning artifact | Read for notebook context; edit only if notebook-planning guidance is intentionally being updated. |

Companion files may use alternative setup flows, older instructions, or different model defaults. Do not treat them as the deployment source of truth for the app.

## Sample and input text files

| Path | Purpose | Status | When to edit |
| --- | --- | --- | --- |
| `legal_doc.txt` | Example legal document text used to test ingestion and graph extraction behavior. | Sample input | Edit when you need a different example legal document for demos or manual testing. |
| `story.txt` | Non-legal sample text. Useful for quick experiments, but not representative of the main use case. | Sample/reference input | Edit only if demo inputs are being refreshed intentionally. |

## External reference snapshots

| Path | Purpose | Status | When to edit |
| --- | --- | --- | --- |
| `documentations/` | Saved HTML snapshots from external docs and articles related to SurrealDB and LangChain. | Reference/archive material | Usually read-only. Update only if you intentionally refresh the saved reference set. |

## Other reference material

| Path | Purpose | Status | When to edit |
| --- | --- | --- | --- |
| `history.txt` | Scratch SurrealQL and indexing notes. | Historical/reference note | Treat as background context, not runtime configuration. Edit only if curating those notes on purpose. |

## Practical editing rules

- If the user asks to change runtime behavior, start with `server.py`, `soreal_engine.py`, `static/index.html`, and the container/config files.
- If the user asks to change setup instructions or repo orientation, start with `README.md`, `AGENT.md`, and the docs under `docs/`.
- If the user asks about notebooks, presentations, or saved HTML snapshots, treat them as companion artifacts unless they explicitly want those files changed.
