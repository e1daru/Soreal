"""
app.py — Soreal Legal Document Analysis Web App
Run locally with:  python app.py
"""

from __future__ import annotations

import uuid

from flask import Flask, jsonify, render_template, request

import agent as ag

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB upload limit

# ── Routes ────────────────────────────────────────────────────────────────────


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    """Simple liveness check — also reports whether the agent is ready."""
    return jsonify({"status": "ok", "agent_ready": ag.is_ready()})


@app.route("/init", methods=["POST"])
def init_agent():
    """
    Initialise the agent (connect to SurrealDB + Ollama, load embeddings).
    Called once on startup from the UI.
    """
    try:
        ag.init_agent()
        return jsonify({"status": "ok", "message": "Agent initialised successfully."})
    except ConnectionError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 503
    except Exception as exc:  # noqa: BLE001
        return jsonify({"status": "error", "message": f"Unexpected error: {exc}"}), 500


@app.route("/ingest", methods=["POST"])
def ingest():
    """
    Ingest a legal document (text or uploaded file) into the knowledge graph.

    JSON body:  { "text": "...", "session_id": "..." }  (session_id optional)
    Multipart:  file field named "file", optional "session_id" form field
    """
    if not ag.is_ready():
        return jsonify({"status": "error", "message": "Agent not initialised. POST /init first."}), 503

    # Accept either JSON or multipart/form-data
    text = ""
    session_id = ""

    if request.content_type and "multipart/form-data" in request.content_type:
        uploaded = request.files.get("file")
        if uploaded:
            text = uploaded.read().decode("utf-8", errors="replace")
        session_id = request.form.get("session_id", "")
    else:
        data = request.get_json(silent=True) or {}
        text = data.get("text", "")
        session_id = data.get("session_id", "")

    if not text.strip():
        return jsonify({"status": "error", "message": "No document text provided."}), 400

    if not session_id:
        session_id = str(uuid.uuid4())

    thread_config = {"configurable": {"thread_id": session_id}}

    try:
        result = ag.run_ingest_pipeline(text, thread_config)
        result["session_id"] = session_id
        return jsonify(result)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/ask", methods=["POST"])
def ask():
    """
    Ask a question about the ingested document.

    JSON body:
      { "question": "...", "session_id": "..." }
    """
    if not ag.is_ready():
        return jsonify({"status": "error", "message": "Agent not initialised. POST /init first."}), 503

    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    session_id = (data.get("session_id") or "").strip()

    if not question:
        return jsonify({"status": "error", "message": "No question provided."}), 400
    if not session_id:
        return jsonify({"status": "error", "message": "No session_id provided."}), 400

    thread_config = {"configurable": {"thread_id": session_id}}

    try:
        result = ag.ask_question(question, thread_config)
        return jsonify({"status": "ok", **result})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/summary", methods=["GET"])
def summary():
    """Return record counts for all KG tables."""
    if not ag.is_ready():
        return jsonify({"status": "error", "message": "Agent not initialised."}), 503
    try:
        import json

        raw = ag.get_graph_summary.invoke({})
        counts = json.loads(raw)
        non_zero = {k: v for k, v in sorted(counts.items()) if v > 0}
        return jsonify({"status": "ok", "counts": non_zero})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"status": "error", "message": str(exc)}), 500


if __name__ == "__main__":
    import os as _os
    debug_mode = _os.getenv("FLASK_DEBUG", "false").lower() == "true"
    print("=" * 60)
    print("  Soreal — Legal Document Analysis")
    print("  Open http://localhost:5000 in your browser")
    print("=" * 60)
    app.run(debug=debug_mode, host="0.0.0.0", port=5000)
