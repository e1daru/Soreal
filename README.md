# Soreal

Legal document KG-RAG web app powered by FastAPI, SurrealDB, and Ollama.

## 1. Prerequisites

- Docker Desktop or Docker Engine with Compose support.
- About 10 GB of free disk space for Ollama and embedding caches.
- More patience than usual on first boot: the app image, embedding model, and Ollama model all need to be downloaded once.

This repo now has a Docker-first local runtime. The stack includes:

- FastAPI app on port `8001`
- SurrealDB on port `8000`
- Ollama on port `11434`

The notebook can still run locally from VS Code or Jupyter, but it can point at the same Dockerized services through `localhost`.

## 2. Configure environment

Create `.env` from `.env.example` if you want to override defaults for local notebook runs, LangSmith tracing, or credentials:

```bash
cp .env.example .env
```

The Compose app service already uses container-internal hostnames, so you usually do not need to edit `.env` for the web app itself.

## 3. Start the stack

```bash
docker compose up --build -d
```

Check container status:

```bash
docker compose ps
```

The app waits for SurrealDB and the Ollama API before starting Uvicorn.

## 4. Pull the Ollama model

The Ollama container starts empty on first boot. Pull the configured model once:

```bash
docker compose exec ollama ollama pull llama3.2:3b
```

If you change `OLLAMA_MODEL` in `.env`, pull that model name instead.

## 5. Open the app

Open `http://127.0.0.1:8001`.

Health checks:

```bash
curl http://127.0.0.1:8001/
curl http://127.0.0.1:8001/api/status
```

## 6. Stop or reset

Stop the containers:

```bash
docker compose down
```

Stop the containers and remove persisted data:

```bash
docker compose down -v
```

Reset the graph without deleting Docker volumes:

```bash
curl -X POST http://127.0.0.1:8001/api/reset
```

## 7. Notebook usage with Docker

Run the notebook locally after the Compose stack is up. Because Compose publishes SurrealDB and Ollama to `localhost:8000` and `localhost:11434`, the notebook's existing defaults continue to work.

If you want local environment variables for the notebook, keep `.env` aligned with `.env.example`.

## 8. Persistence

Compose creates named volumes for:

- SurrealDB data
- Ollama model cache
- HuggingFace embedding cache

That means the database, pulled Ollama models, and embedding downloads survive container restarts.

## 9. Troubleshooting

### First run is slow

That is expected. The first successful startup may need to:

- build the Python image
- download Python dependencies
- download the HuggingFace embedding model
- pull the Ollama model

### The app container keeps restarting

Inspect logs:

```bash
docker compose logs app --tail=200
```

The most common causes are:

- SurrealDB failed to start
- Ollama is still booting
- required Python dependencies changed and the image needs a rebuild

### The model is missing

If analyze or ask fails on a clean machine, pull the model explicitly:

```bash
docker compose exec ollama ollama pull llama3.2:3b
```

### Dockerized Ollama is slow on macOS

On macOS Docker Desktop, the Ollama container usually runs CPU-only. It is functional, but slower than a native host Ollama setup.

### Start from scratch

```bash
docker compose down -v
docker compose up --build -d
docker compose exec ollama ollama pull llama3.2:3b
```

## 10. Optional local Python workflow

If you need to run the FastAPI app outside Docker for debugging, the previous local path still works:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
uvicorn server:app --reload --host 0.0.0.0 --port 8001
```

For that mode, you must also run SurrealDB and Ollama yourself.
