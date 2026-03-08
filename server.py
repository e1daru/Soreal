"""
Soreal Web Server — FastAPI backend for the KG-RAG web app.

Endpoints:
  POST /api/analyze       Stream SSE progress while ingesting a document
  POST /api/ask           Ask a question, get answer + sources + confidence
  GET  /api/status        Current graph summary
  POST /api/reset         Clear the database
  GET  /                  Serve frontend
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import soreal_engine as engine

app = FastAPI(title="Soreal KG-RAG", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"

# Global thread_id for the current session
_session: dict = {"thread_id": None}


# ── Models ──────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=100_000)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=5_000)


# ── Routes ──────────────────────────────────────────────────────────────────

@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest):
    """Ingest a document and stream progress via SSE."""

    def event_stream():
        for event in engine.ingest_document(req.text):
            if event["type"] == "complete":
                _session["thread_id"] = event["thread_id"]
            yield f"data: {json.dumps(event)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/ask")
async def ask(req: AskRequest):
    """Ask a question against the current knowledge graph."""
    tid = _session.get("thread_id")
    if not tid:
        # Create a fresh query-only thread
        tid = str(uuid.uuid4())
        _session["thread_id"] = tid
    result = engine.ask_question(req.question, tid)
    return result


@app.get("/api/status")
async def status():
    """Return current graph summary."""
    raw = engine.get_graph_summary.invoke({})
    return json.loads(raw)


@app.post("/api/reset")
async def reset():
    """Clear database and reset session."""
    engine.reset_database()
    engine.init_schema()
    _session["thread_id"] = None
    return {"status": "ok", "message": "Database cleared."}


# ── Static files ────────────────────────────────────────────────────────────

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    index = STATIC_DIR / "index.html"
    return index.read_text()
