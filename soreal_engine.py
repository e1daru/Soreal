"""
Soreal KG-RAG Engine — extracted from knowledge_graph_rag.ipynb (legal domain).

Provides:
  - SurrealDB legal document schema (18 node tables, 30 edge tables)
  - LangChain tools (ingest, extract, triplets, load, search, mutate, summary)
  - LangGraph agent with auto-approve pipeline + Q&A loop
  - Groundedness scoring with source references
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
import uuid
from typing import Annotated, Generator, TypedDict

import numpy as np
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from surrealdb import Surreal

load_dotenv(override=True)

# ── Config ──────────────────────────────────────────────────────────────────
SURREAL_URL = os.getenv("SURREAL_URL", "ws://localhost:8000/rpc")
SURREAL_USERNAME = os.getenv("SURREAL_USERNAME", "root")
SURREAL_PASSWORD = os.getenv("SURREAL_PASSWORD", "root")
SURREAL_NAMESPACE = os.getenv("SURREAL_NAMESPACE", "soreal")
SURREAL_DATABASE = os.getenv("SURREAL_DATABASE", "kg")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
INGEST_CHUNK_SIZE = int(os.getenv("INGEST_CHUNK_SIZE", "260"))
INGEST_CHUNK_OVERLAP = int(os.getenv("INGEST_CHUNK_OVERLAP", "80"))
MAX_EXTRACTED_ENTITIES = int(os.getenv("MAX_EXTRACTED_ENTITIES", "24"))
MAX_TRIPLETS_PER_CHUNK = int(os.getenv("MAX_TRIPLETS_PER_CHUNK", "24"))
OLLAMA_EXTRACTION_NUM_PREDICT = int(os.getenv("OLLAMA_EXTRACTION_NUM_PREDICT", "256"))

# ── Connections ─────────────────────────────────────────────────────────────
import threading

_conn_lock = threading.Lock()


def _new_connection() -> Surreal:
    """Create and authenticate a fresh SurrealDB connection."""
    c = Surreal(SURREAL_URL)
    c.signin({"username": SURREAL_USERNAME, "password": SURREAL_PASSWORD})
    c.use(SURREAL_NAMESPACE, SURREAL_DATABASE)
    return c


conn = _new_connection()


def _ensure_conn() -> Surreal:
    """Return the module-level conn, reconnecting if the WebSocket is dead."""
    global conn
    try:
        conn.query("RETURN true;")
        return conn
    except Exception:
        with _conn_lock:
            # Double-check after acquiring lock
            try:
                conn.query("RETURN true;")
                return conn
            except Exception:
                conn = _new_connection()
                return conn

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
extraction_llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0,
    num_predict=OLLAMA_EXTRACTION_NUM_PREDICT,
)
print(f"[soreal] Using Ollama model: {OLLAMA_MODEL}")


def ensure_ollama_model_ready() -> None:
    """Fail fast with a clear message if the configured Ollama model is missing."""
    tags_url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags"
    try:
        with urllib.request.urlopen(tags_url, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Unable to reach Ollama at {OLLAMA_BASE_URL}. Check that the Ollama service is running."
        ) from exc

    models = payload.get("models", []) if isinstance(payload, dict) else []
    available_models = [model.get("name", "") for model in models if isinstance(model, dict)]
    if OLLAMA_MODEL not in available_models:
        available = ", ".join(sorted(filter(None, available_models))) or "none"
        raise RuntimeError(
            f"Ollama model '{OLLAMA_MODEL}' is not installed. Available models: {available}. "
            f"Run 'docker compose exec ollama ollama pull {OLLAMA_MODEL}' and retry."
        )

# ── Entity & predicate catalogs ─────────────────────────────────────────────

VALID_ENTITY_TYPES = [
    "party", "representative", "clause", "obligation", "right", "restriction",
    "condition", "definition", "key_date", "payment_term", "fee",
    "liability_cap", "risk", "compliance_requirement", "section", "flag",
    "amendment", "document", "term",
]

ENTITY_TYPE_TO_TABLE = {t: t for t in VALID_ENTITY_TYPES}
ENTITY_TYPE_TO_TABLE["right"] = "legal_right"  # avoid SurrealQL keyword

VECTORIZED_TABLES = [
    "party", "clause", "obligation", "legal_right", "restriction",
    "risk", "condition", "definition", "compliance_requirement",
    "section", "chunk",
]

VALID_PREDICATES = [
    "contains_section", "contains_clause", "contains", "has_child",
    "creates", "grants", "imposes", "depends_on", "references_term",
    "cross_references", "involves", "binds", "benefits", "signs",
    "represented_by", "has_term", "has_key_date", "due_by", "amended_by",
    "has_payment", "sets_liability", "has_fee", "has_risk",
    "requires_compliance", "flagged_as", "derived_from", "deviates_from",
    "related_to", "similar_to", "sourced_from", "has_version",
]

NODE_TABLES = list(set(ENTITY_TYPE_TO_TABLE.values())) + ["chunk", "document", "paragraph"]
EDGE_TABLES = VALID_PREDICATES
EXTRA_NODE_TABLES = ["document_version", "template", "playbook", "precedent"]
RESETTABLE_TABLES = sorted(set(NODE_TABLES + EXTRA_NODE_TABLES + EDGE_TABLES))

# ── Schema ──────────────────────────────────────────────────────────────────
SCHEMA_STATEMENTS = [
    # Document structure
    "DEFINE TABLE IF NOT EXISTS document SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS label    ON document TYPE string",
    "DEFINE FIELD IF NOT EXISTS doc_type ON document TYPE option<string>",
    "DEFINE FIELD IF NOT EXISTS vector   ON document TYPE option<array<float>>",
    "DEFINE TABLE IF NOT EXISTS document_version SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS label          ON document_version TYPE string",
    "DEFINE FIELD IF NOT EXISTS version_number ON document_version TYPE option<string>",
    "DEFINE TABLE IF NOT EXISTS section SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS label          ON section TYPE string",
    "DEFINE FIELD IF NOT EXISTS section_number ON section TYPE option<string>",
    "DEFINE FIELD IF NOT EXISTS vector         ON section TYPE option<array<float>>",
    "DEFINE TABLE IF NOT EXISTS paragraph SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS content         ON paragraph TYPE string",
    "DEFINE FIELD IF NOT EXISTS paragraph_index ON paragraph TYPE option<int>",
    "DEFINE FIELD IF NOT EXISTS vector          ON paragraph TYPE option<array<float>>",
    "DEFINE TABLE IF NOT EXISTS chunk SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS content     ON chunk TYPE string",
    "DEFINE FIELD IF NOT EXISTS source      ON chunk TYPE option<string>",
    "DEFINE FIELD IF NOT EXISTS chunk_index ON chunk TYPE int",
    "DEFINE FIELD IF NOT EXISTS vector      ON chunk TYPE option<array<float>>",
    # Legal semantics
    "DEFINE TABLE IF NOT EXISTS clause SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS label   ON clause TYPE string",
    "DEFINE FIELD IF NOT EXISTS content ON clause TYPE option<string>",
    "DEFINE FIELD IF NOT EXISTS vector  ON clause TYPE option<array<float>>",
    "DEFINE TABLE IF NOT EXISTS obligation SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS label       ON obligation TYPE string",
    "DEFINE FIELD IF NOT EXISTS description ON obligation TYPE option<string>",
    "DEFINE FIELD IF NOT EXISTS vector      ON obligation TYPE option<array<float>>",
    "DEFINE TABLE IF NOT EXISTS legal_right SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS label       ON legal_right TYPE string",
    "DEFINE FIELD IF NOT EXISTS description ON legal_right TYPE option<string>",
    "DEFINE FIELD IF NOT EXISTS vector      ON legal_right TYPE option<array<float>>",
    "DEFINE TABLE IF NOT EXISTS restriction SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS label       ON restriction TYPE string",
    "DEFINE FIELD IF NOT EXISTS description ON restriction TYPE option<string>",
    "DEFINE FIELD IF NOT EXISTS vector      ON restriction TYPE option<array<float>>",
    "DEFINE TABLE IF NOT EXISTS condition SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS label       ON condition TYPE string",
    "DEFINE FIELD IF NOT EXISTS description ON condition TYPE option<string>",
    "DEFINE FIELD IF NOT EXISTS vector      ON condition TYPE option<array<float>>",
    "DEFINE TABLE IF NOT EXISTS definition SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS label   ON definition TYPE string",
    "DEFINE FIELD IF NOT EXISTS meaning ON definition TYPE option<string>",
    "DEFINE FIELD IF NOT EXISTS vector  ON definition TYPE option<array<float>>",
    # Parties
    "DEFINE TABLE IF NOT EXISTS party SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS label      ON party TYPE string",
    "DEFINE FIELD IF NOT EXISTS party_type ON party TYPE option<string>",
    "DEFINE FIELD IF NOT EXISTS vector     ON party TYPE option<array<float>>",
    "DEFINE TABLE IF NOT EXISTS representative SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS label ON representative TYPE string",
    "DEFINE FIELD IF NOT EXISTS role  ON representative TYPE option<string>",
    # Temporal
    "DEFINE TABLE IF NOT EXISTS term SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS label      ON term TYPE string",
    "DEFINE FIELD IF NOT EXISTS start_date ON term TYPE option<string>",
    "DEFINE FIELD IF NOT EXISTS end_date   ON term TYPE option<string>",
    "DEFINE TABLE IF NOT EXISTS key_date SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS label      ON key_date TYPE string",
    "DEFINE FIELD IF NOT EXISTS date_value ON key_date TYPE option<string>",
    "DEFINE FIELD IF NOT EXISTS date_type  ON key_date TYPE option<string>",
    "DEFINE TABLE IF NOT EXISTS amendment SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS label       ON amendment TYPE string",
    "DEFINE FIELD IF NOT EXISTS description ON amendment TYPE option<string>",
    # Financial
    "DEFINE TABLE IF NOT EXISTS payment_term SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS label       ON payment_term TYPE string",
    "DEFINE FIELD IF NOT EXISTS description ON payment_term TYPE option<string>",
    "DEFINE FIELD IF NOT EXISTS amount      ON payment_term TYPE option<string>",
    "DEFINE TABLE IF NOT EXISTS liability_cap SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS label       ON liability_cap TYPE string",
    "DEFINE FIELD IF NOT EXISTS cap_amount  ON liability_cap TYPE option<string>",
    "DEFINE FIELD IF NOT EXISTS description ON liability_cap TYPE option<string>",
    "DEFINE TABLE IF NOT EXISTS fee SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS label       ON fee TYPE string",
    "DEFINE FIELD IF NOT EXISTS amount      ON fee TYPE option<string>",
    "DEFINE FIELD IF NOT EXISTS description ON fee TYPE option<string>",
    # Risk & Analysis
    "DEFINE TABLE IF NOT EXISTS risk SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS label       ON risk TYPE string",
    "DEFINE FIELD IF NOT EXISTS severity    ON risk TYPE option<string>",
    "DEFINE FIELD IF NOT EXISTS description ON risk TYPE option<string>",
    "DEFINE FIELD IF NOT EXISTS vector      ON risk TYPE option<array<float>>",
    "DEFINE TABLE IF NOT EXISTS compliance_requirement SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS label       ON compliance_requirement TYPE string",
    "DEFINE FIELD IF NOT EXISTS description ON compliance_requirement TYPE option<string>",
    "DEFINE FIELD IF NOT EXISTS vector      ON compliance_requirement TYPE option<array<float>>",
    "DEFINE TABLE IF NOT EXISTS flag SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS label       ON flag TYPE string",
    "DEFINE FIELD IF NOT EXISTS flag_type   ON flag TYPE option<string>",
    "DEFINE FIELD IF NOT EXISTS description ON flag TYPE option<string>",
    # Comparison & Templates
    "DEFINE TABLE IF NOT EXISTS template SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS label       ON template TYPE string",
    "DEFINE FIELD IF NOT EXISTS description ON template TYPE option<string>",
    "DEFINE TABLE IF NOT EXISTS playbook SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS label       ON playbook TYPE string",
    "DEFINE FIELD IF NOT EXISTS description ON playbook TYPE option<string>",
    "DEFINE TABLE IF NOT EXISTS precedent SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS label  ON precedent TYPE string",
    "DEFINE FIELD IF NOT EXISTS source ON precedent TYPE option<string>",
    # Edge tables
    "DEFINE TABLE IF NOT EXISTS has_version       TYPE RELATION SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS source_chunk ON has_version TYPE option<string>",
    "DEFINE TABLE IF NOT EXISTS contains_section  TYPE RELATION SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS source_chunk ON contains_section TYPE option<string>",
    "DEFINE TABLE IF NOT EXISTS contains          TYPE RELATION SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS source_chunk ON contains TYPE option<string>",
    "DEFINE TABLE IF NOT EXISTS has_child         TYPE RELATION SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS source_chunk ON has_child TYPE option<string>",
    "DEFINE TABLE IF NOT EXISTS contains_clause   TYPE RELATION SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS source_chunk ON contains_clause TYPE option<string>",
    "DEFINE TABLE IF NOT EXISTS creates           TYPE RELATION SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS source_chunk ON creates TYPE option<string>",
    "DEFINE TABLE IF NOT EXISTS grants            TYPE RELATION SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS source_chunk ON grants TYPE option<string>",
    "DEFINE TABLE IF NOT EXISTS imposes           TYPE RELATION SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS source_chunk ON imposes TYPE option<string>",
    "DEFINE TABLE IF NOT EXISTS depends_on        TYPE RELATION SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS source_chunk ON depends_on TYPE option<string>",
    "DEFINE TABLE IF NOT EXISTS references_term   TYPE RELATION SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS source_chunk ON references_term TYPE option<string>",
    "DEFINE TABLE IF NOT EXISTS cross_references  TYPE RELATION SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS source_chunk ON cross_references TYPE option<string>",
    "DEFINE TABLE IF NOT EXISTS involves          TYPE RELATION SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS source_chunk ON involves TYPE option<string>",
    "DEFINE TABLE IF NOT EXISTS binds             TYPE RELATION SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS source_chunk ON binds TYPE option<string>",
    "DEFINE TABLE IF NOT EXISTS benefits          TYPE RELATION SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS source_chunk ON benefits TYPE option<string>",
    "DEFINE TABLE IF NOT EXISTS signs             TYPE RELATION SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS source_chunk ON signs TYPE option<string>",
    "DEFINE TABLE IF NOT EXISTS represented_by    TYPE RELATION SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS source_chunk ON represented_by TYPE option<string>",
    "DEFINE TABLE IF NOT EXISTS has_term          TYPE RELATION SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS source_chunk ON has_term TYPE option<string>",
    "DEFINE TABLE IF NOT EXISTS has_key_date      TYPE RELATION SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS source_chunk ON has_key_date TYPE option<string>",
    "DEFINE TABLE IF NOT EXISTS due_by            TYPE RELATION SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS source_chunk ON due_by TYPE option<string>",
    "DEFINE TABLE IF NOT EXISTS amended_by        TYPE RELATION SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS source_chunk ON amended_by TYPE option<string>",
    "DEFINE TABLE IF NOT EXISTS has_payment       TYPE RELATION SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS source_chunk ON has_payment TYPE option<string>",
    "DEFINE TABLE IF NOT EXISTS sets_liability    TYPE RELATION SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS source_chunk ON sets_liability TYPE option<string>",
    "DEFINE TABLE IF NOT EXISTS has_fee           TYPE RELATION SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS source_chunk ON has_fee TYPE option<string>",
    "DEFINE TABLE IF NOT EXISTS has_risk          TYPE RELATION SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS source_chunk ON has_risk TYPE option<string>",
    "DEFINE TABLE IF NOT EXISTS requires_compliance TYPE RELATION SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS source_chunk ON requires_compliance TYPE option<string>",
    "DEFINE TABLE IF NOT EXISTS flagged_as        TYPE RELATION SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS source_chunk ON flagged_as TYPE option<string>",
    "DEFINE TABLE IF NOT EXISTS derived_from      TYPE RELATION SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS source_chunk ON derived_from TYPE option<string>",
    "DEFINE TABLE IF NOT EXISTS deviates_from     TYPE RELATION SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS source_chunk ON deviates_from TYPE option<string>",
    "DEFINE TABLE IF NOT EXISTS related_to        TYPE RELATION SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS source_chunk ON related_to TYPE option<string>",
    "DEFINE FIELD IF NOT EXISTS type         ON related_to TYPE option<string>",
    "DEFINE TABLE IF NOT EXISTS similar_to        TYPE RELATION SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS source_chunk ON similar_to TYPE option<string>",
    "DEFINE TABLE IF NOT EXISTS sourced_from      TYPE RELATION SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS source_chunk ON sourced_from TYPE option<string>",
    # Vector indexes
    "DEFINE INDEX IF NOT EXISTS idx_party_vec       ON party       FIELDS vector HNSW DIMENSION 384 DIST COSINE",
    "DEFINE INDEX IF NOT EXISTS idx_clause_vec      ON clause      FIELDS vector HNSW DIMENSION 384 DIST COSINE",
    "DEFINE INDEX IF NOT EXISTS idx_obligation_vec  ON obligation  FIELDS vector HNSW DIMENSION 384 DIST COSINE",
    "DEFINE INDEX IF NOT EXISTS idx_right_vec       ON legal_right FIELDS vector HNSW DIMENSION 384 DIST COSINE",
    "DEFINE INDEX IF NOT EXISTS idx_restriction_vec ON restriction FIELDS vector HNSW DIMENSION 384 DIST COSINE",
    "DEFINE INDEX IF NOT EXISTS idx_risk_vec        ON risk        FIELDS vector HNSW DIMENSION 384 DIST COSINE",
    "DEFINE INDEX IF NOT EXISTS idx_condition_vec   ON condition   FIELDS vector HNSW DIMENSION 384 DIST COSINE",
    "DEFINE INDEX IF NOT EXISTS idx_definition_vec  ON definition  FIELDS vector HNSW DIMENSION 384 DIST COSINE",
    "DEFINE INDEX IF NOT EXISTS idx_compliance_vec  ON compliance_requirement FIELDS vector HNSW DIMENSION 384 DIST COSINE",
    "DEFINE INDEX IF NOT EXISTS idx_section_vec     ON section     FIELDS vector HNSW DIMENSION 384 DIST COSINE",
    "DEFINE INDEX IF NOT EXISTS idx_chunk_vec       ON chunk       FIELDS vector HNSW DIMENSION 384 DIST COSINE",
]


def init_schema():
    conn = _ensure_conn()
    for stmt in SCHEMA_STATEMENTS:
        conn.query(stmt + ";")


def reset_database():
    """Drop all data for a fresh document analysis."""
    conn = _ensure_conn()
    for tbl in RESETTABLE_TABLES:
        try:
            conn.query(f"DELETE {tbl};")
        except Exception:
            pass


# ── Prompts ─────────────────────────────────────────────────────────────────

LEGAL_NER_PROMPT = """\
You are a legal-document NER system.
Given a text chunk from a contract or legal document, extract only the most salient legal entities.

Allowed entity types: {entity_types}
Maximum entities to return: {max_entities}

Return **only** valid JSON – a list of objects:
[
  {{"entity_type": "...", "name": "...", "properties": {{...}} }}
]

For each entity:
- `name` is a short, descriptive label.
- `properties` may include fields like amount, date_value, or description when they are explicit in the text.
- Prefer parties, obligations, payment terms, fees, dates, rights, restrictions, conditions, and sections.
- Skip generic legal filler and near-duplicate entities.
- Keep the list concise and deduplicated.

If a chunk has no entities, return an empty list: []

Text chunk:
\"\"\"
{text}
\"\"\"
"""

LEGAL_TRIPLET_PROMPT = """\
You are a legal-document relation extraction system.
Given extracted entities and the original text chunk, produce a list of
(subject, predicate, object) triplets that capture relationships.

Entities found so far:
{entities}

Allowed predicates: {predicates}
Maximum triplets to return: {max_triplets}

Return **only** valid JSON – a list of triplet objects:
[
  {{"subject": "entity_name", "predicate": "...", "object": "entity_name",
    "source_chunk": "chunk_id"}}
]

Rules:
- Only use entities from the list above as subject / object.
- Only use predicates from the allowed list.
- Each triplet must include the source_chunk id for traceability.
- Only return direct, explicit, high-confidence relationships.
- Skip weak, implied, or duplicate triplets.

Text chunk (id = {chunk_id}):
\"\"\"
{text}
\"\"\"
"""


# ── Helpers ─────────────────────────────────────────────────────────────────

def _safe_json_array(raw: str) -> list:
    text = str(raw or "").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        pass
    try:
        start = text.index("[")
        end = text.rindex("]") + 1
        data = json.loads(text[start:end])
        return data if isinstance(data, list) else []
    except (ValueError, json.JSONDecodeError):
        return []


def _make_id(name: str) -> str:
    return re.sub(r'[^a-z0-9_]', '_', name.lower().strip())[:60]


def _normalize_predicate(raw: str) -> str | None:
    """Match a raw predicate string to VALID_PREDICATES (case-insensitive, fuzzy)."""
    cleaned = raw.strip().lower().replace(" ", "_")
    if cleaned in VALID_PREDICATES:
        return cleaned
    # Try partial / startswith match
    for vp in VALID_PREDICATES:
        if vp.startswith(cleaned) or cleaned.startswith(vp):
            return vp
    return None


def _extract_query_rows(raw_result) -> list[dict]:
    """Normalize SurrealDB query responses into a flat list of record dicts."""
    if raw_result is None or isinstance(raw_result, str):
        return []

    if isinstance(raw_result, dict):
        if isinstance(raw_result.get("result"), list):
            return [r for r in raw_result["result"] if isinstance(r, dict)]
        return [raw_result] if isinstance(raw_result.get("id"), str) else []

    if not isinstance(raw_result, list):
        return []

    rows: list[dict] = []
    for item in raw_result:
        if isinstance(item, list):
            rows.extend(r for r in item if isinstance(r, dict))
            continue
        if isinstance(item, dict):
            result = item.get("result")
            if isinstance(result, list):
                rows.extend(r for r in result if isinstance(r, dict))
                continue
            rows.append(item)
    return rows


# ── Tools ───────────────────────────────────────────────────────────────────

@tool
def ingest_text(text: str, source: str = "legal_doc") -> str:
    """Split a legal document into overlapping chunks, embed and store them."""
    conn = _ensure_conn()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=INGEST_CHUNK_SIZE, chunk_overlap=INGEST_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = splitter.split_text(text)
    records = []
    for i, c in enumerate(chunks):
        vec = embeddings.embed_query(c)
        chunk_id = f"chunk:{source}_{i}"
        conn.query(
            "CREATE type::record($tb, $id) SET content=$content, "
            "source=$source, chunk_index=$idx, vector=$vec;",
            {"tb": "chunk", "id": f"{source}_{i}",
             "content": c, "source": source, "idx": i, "vec": vec}
        )
        records.append({"chunk_id": chunk_id, "length": len(c)})
    return json.dumps({"status": "ok", "chunks_created": len(records), "details": records})


@tool
def extract_legal_entities(chunks_json: str) -> str:
    """Run legal NER over every stored chunk and return merged entities JSON."""
    conn = _ensure_conn()
    chunks_info = json.loads(chunks_json)
    all_entities = []
    seen = set()
    for info in chunks_info.get("details", chunks_info):
        cid = info["chunk_id"] if isinstance(info, dict) else info
        short = cid.split(":")[-1] if ":" in cid else cid
        rows = conn.query(
            "SELECT content FROM type::record($tb, $id);",
            {"tb": "chunk", "id": short}
        )
        if not rows or not rows[0]:
            continue
        text = rows[0][0]["content"] if isinstance(rows[0], list) else rows[0].get("content", "")
        prompt = LEGAL_NER_PROMPT.format(
            entity_types=", ".join(VALID_ENTITY_TYPES),
            max_entities=MAX_EXTRACTED_ENTITIES,
            text=text,
        )
        resp = extraction_llm.invoke(prompt)
        raw = resp.content if hasattr(resp, "content") else str(resp)
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            continue
        try:
            ents = json.loads(match.group())
        except json.JSONDecodeError:
            continue
        for e in ents:
            key = (e.get("entity_type", ""), e.get("name", ""))
            if key not in seen:
                seen.add(key)
                all_entities.append(e)
            if len(all_entities) >= MAX_EXTRACTED_ENTITIES:
                break
        if len(all_entities) >= MAX_EXTRACTED_ENTITIES:
            break
    return json.dumps(all_entities)


@tool
def form_legal_triplets(entities_json: str, chunks_json: str) -> str:
    """Extract relation triplets between the given entities for every chunk."""
    conn = _ensure_conn()
    entities = json.loads(entities_json)
    chunks_info = json.loads(chunks_json)
    all_triplets = []
    for info in chunks_info.get("details", chunks_info):
        cid = info["chunk_id"] if isinstance(info, dict) else info
        short = cid.split(":")[-1] if ":" in cid else cid
        rows = conn.query(
            "SELECT content FROM type::record($tb, $id);",
            {"tb": "chunk", "id": short}
        )
        if not rows or not rows[0]:
            continue
        text = rows[0][0]["content"] if isinstance(rows[0], list) else rows[0].get("content", "")
        prompt = LEGAL_TRIPLET_PROMPT.format(
            entities=json.dumps(entities[:MAX_EXTRACTED_ENTITIES]),
            predicates=", ".join(VALID_PREDICATES),
            max_triplets=MAX_TRIPLETS_PER_CHUNK,
            chunk_id=cid, text=text,
        )
        resp = extraction_llm.invoke(prompt)
        raw = resp.content if hasattr(resp, "content") else str(resp)
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            continue
        try:
            trips = json.loads(match.group())
        except json.JSONDecodeError:
            continue
        for t in trips:
            norm = _normalize_predicate(t.get("predicate", ""))
            if norm:
                t["predicate"] = norm
                all_triplets.append(t)
            if len(all_triplets) >= MAX_TRIPLETS_PER_CHUNK:
                break
        if len(all_triplets) >= MAX_TRIPLETS_PER_CHUNK:
            break
    return json.dumps(all_triplets)


@tool
def load_entities_and_triplets(entities_json: str, triplets_json: str) -> str:
    """Upsert legal entities (with vectors) and create relation edges."""
    conn = _ensure_conn()
    entities = json.loads(entities_json)
    triplets = json.loads(triplets_json)

    ent_count = 0
    for e in entities:
        etype = e.get("entity_type", "unknown")
        table = ENTITY_TYPE_TO_TABLE.get(etype)
        if not table:
            continue
        rid = _make_id(e["name"])
        props = e.get("properties", {})
        label = e.get("name", rid)
        vec = embeddings.embed_query(label) if table in VECTORIZED_TABLES else None
        fields = "label=$label, vector=$vec" if vec else "label=$label"
        for k, v in props.items():
            fields += f", {k}=$prop_{k}"
        params = {"tb": table, "id": rid, "label": label, "vec": vec}
        for k, v in props.items():
            params[f"prop_{k}"] = v
        conn.query(f"CREATE type::record($tb, $id) SET {fields};", params)
        ent_count += 1

    # Build a case-insensitive lookup from entity name -> entity_type
    ent_lookup = {e["name"].lower(): e["entity_type"] for e in entities if "name" in e}

    edge_count = 0
    for t in triplets:
        pred = t.get("predicate")
        if pred not in VALID_PREDICATES:
            continue
        subj_name = t.get("subject", "")
        obj_name = t.get("object", "")
        subj_type = ent_lookup.get(subj_name.lower())
        obj_type = ent_lookup.get(obj_name.lower())
        if not subj_type or not obj_type:
            continue
        subj_table = ENTITY_TYPE_TO_TABLE.get(subj_type)
        obj_table = ENTITY_TYPE_TO_TABLE.get(obj_type)
        if not subj_table or not obj_table:
            continue
        s_id = _make_id(subj_name)
        o_id = _make_id(obj_name)
        sc = t.get("source_chunk", "")
        try:
            conn.query(
                f"RELATE {subj_table}:`{s_id}` ->{pred}-> {obj_table}:`{o_id}` "
                f"SET source_chunk=$sc;",
                {"sc": sc}
            )
            edge_count += 1
        except Exception:
            pass  # skip individual bad edges

    return json.dumps({"entities_loaded": ent_count, "edges_created": edge_count})


@tool
def search_graph(query: str, top_k: int = 5) -> str:
    """Semantic vector search across all vectorised KG tables + graph walk."""
    conn = _ensure_conn()
    qvec = embeddings.embed_query(query)
    results = []
    for tbl in VECTORIZED_TABLES:
        rows = conn.query(
            f"SELECT *, vector::similarity::cosine(vector, $qvec) AS score "
            f"FROM {tbl} WHERE vector != NONE "
            f"ORDER BY score DESC LIMIT $k;",
            {"qvec": qvec, "k": top_k}
        )
        for r in _extract_query_rows(rows):
            results.append({"table": tbl, "id": str(r.get("id", "")),
                            "label": r.get("label", r.get("content", "")),
                            "score": r.get("score", 0)})
    results.sort(key=lambda x: x["score"], reverse=True)
    top = results[:top_k]
    enriched = []
    for r in top:
        rid = r["id"]
        out_rows = _extract_query_rows(conn.query(f"SELECT ->? AS rels FROM {rid};"))
        in_rows = _extract_query_rows(conn.query(f"SELECT <-? AS rels FROM {rid};"))
        r["outgoing"] = out_rows[0].get("rels", []) if out_rows else []
        r["incoming"] = in_rows[0].get("rels", []) if in_rows else []
        enriched.append(r)
    return json.dumps(enriched, default=str)


@tool
def mutate_graph(surql: str) -> str:
    """Execute a SurrealQL mutation (CREATE / RELATE / UPDATE / DELETE)."""
    conn = _ensure_conn()
    blocked = ["REMOVE", "DROP", "DEFINE", "INFO"]
    upper = surql.upper()
    for b in blocked:
        if b in upper:
            return json.dumps({"error": f"Blocked keyword: {b}"})
    res = conn.query(surql)
    return json.dumps(res, default=str)


@tool
def get_graph_summary() -> str:
    """Return record counts for every node and edge table in the KG."""
    conn = _ensure_conn()
    tables = (
        [t for t in ENTITY_TYPE_TO_TABLE.values()]
        + ["chunk", "document", "paragraph"]
        + VALID_PREDICATES
    )
    counts = {}
    for t in set(tables):
        try:
            rows = _extract_query_rows(conn.query(f"SELECT count() FROM {t} GROUP ALL;"))
            counts[t] = rows[0].get("count", 0) if rows else 0
        except Exception:
            counts[t] = 0
    return json.dumps(counts)


ALL_TOOLS = [
    ingest_text, extract_legal_entities, form_legal_triplets,
    load_entities_and_triplets, search_graph, mutate_graph, get_graph_summary,
]


# ── Agent Graph ─────────────────────────────────────────────────────────────

class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    chunks_json: str
    entities_json: str
    triplets_json: str
    phase: str


QUERY_TOOLS = [search_graph, mutate_graph, get_graph_summary]


def _latest_human(state):
    return next((m for m in reversed(state.get("messages", [])) if isinstance(m, HumanMessage)), None)


def entry_node(state):
    return {}


def route_from_entry(state):
    h = _latest_human(state)
    content = h.content if h else ""
    if "Analyze this legal document:" in content:
        return "ingest"
    return "query"


def ingest_node(state):
    h = _latest_human(state)
    if h is None:
        return {"messages": [AIMessage(content="No input text found.")], "phase": "ingest"}
    content = h.content
    marker = "Analyze this legal document:"
    text = content.split(marker, 1)[1].strip() if marker in content else content.strip()
    chunks_json = ingest_text.invoke({"text": text, "source": "legal_doc"})
    info = json.loads(chunks_json)
    return {
        "chunks_json": chunks_json,
        "messages": [AIMessage(content=f"Stored {info.get('chunks_created', 0)} chunks in SurrealDB.")],
        "phase": "ingest_completed",
    }


def extract_entities_node(state):
    chunks_json = state.get("chunks_json", "")
    if not chunks_json:
        return {
            "entities_json": "[]",
            "messages": [AIMessage(content="No chunks available for entity extraction.")],
            "phase": "extract_completed",
        }
    entities_json = extract_legal_entities.invoke({"chunks_json": chunks_json})
    count = len(json.loads(entities_json))
    return {
        "entities_json": entities_json,
        "messages": [AIMessage(content=f"Extracted {count} legal entities.")],
        "phase": "extract_completed",
    }


def form_triplets_node(state):
    entities_json = state.get("entities_json", "[]")
    chunks_json = state.get("chunks_json", "")
    triplets_json = form_legal_triplets.invoke(
        {"entities_json": entities_json, "chunks_json": chunks_json}
    )
    count = len(json.loads(triplets_json))
    return {
        "triplets_json": triplets_json,
        "messages": [AIMessage(content=f"Formed {count} relation triplets.")],
        "phase": "triplets_completed",
    }


def load_node(state):
    entities_json = state.get("entities_json", "[]")
    triplets_json = state.get("triplets_json", "[]")
    result = load_entities_and_triplets.invoke(
        {"entities_json": entities_json, "triplets_json": triplets_json}
    )
    info = json.loads(result)
    return {
        "messages": [AIMessage(
            content=f"Knowledge graph updated: {info.get('entities_loaded', 0)} entities, "
                    f"{info.get('edges_created', 0)} edges."
        )],
        "phase": "loaded",
    }


def query_node(state):
    system = SystemMessage(content=(
        "You are a legal-document analysis assistant backed by a knowledge graph in SurrealDB.\n"
        "RULES:\n"
        "1. For counting or overview questions ('how many ...', 'list all ...'), "
        "call get_graph_summary FIRST.\n"
        "2. For specific questions about parties, clauses, obligations, etc., use search_graph.\n"
        "3. If the user provides corrections or new facts, use mutate_graph.\n"
        "4. ALWAYS answer based on data returned by tools — do NOT guess or give generic definitions.\n"
        "5. Cite the source chunk ID when possible for traceability."
    ))
    try:
        bound = llm.bind_tools(QUERY_TOOLS)
        response = bound.invoke([system] + state.get("messages", []))
        return {"messages": [response], "phase": "query"}
    except Exception:
        # Fallback: model doesn't support tool calling — do manual search + answer
        h = _latest_human(state)
        question = h.content if h else ""
        search_result = search_graph.invoke({"query": question, "top_k": 5})
        summary = get_graph_summary.invoke({})
        context_prompt = (
            f"Based ONLY on the following knowledge graph data, answer the user's question.\n\n"
            f"Search results:\n{search_result}\n\nGraph summary:\n{summary}\n\n"
            f"Question: {question}"
        )
        response = llm.invoke([system, HumanMessage(content=context_prompt)])
        return {"messages": [response], "phase": "query"}


query_tools_executor = ToolNode(QUERY_TOOLS)


def should_continue_query(state):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "query_tools"
    return END


def build_graph():
    """Build and compile the LangGraph agent (no HITL)."""
    wf = StateGraph(AgentState)
    wf.add_node("entry", entry_node)
    wf.add_node("ingest", ingest_node)
    wf.add_node("extract_entities", extract_entities_node)
    wf.add_node("form_triplets", form_triplets_node)
    wf.add_node("load", load_node)
    wf.add_node("query", query_node)
    wf.add_node("query_tools", query_tools_executor)
    wf.add_edge(START, "entry")
    wf.add_conditional_edges("entry", route_from_entry, {"ingest": "ingest", "query": "query"})
    wf.add_edge("ingest", "extract_entities")
    wf.add_edge("extract_entities", "form_triplets")
    wf.add_edge("form_triplets", "load")
    wf.add_edge("load", END)
    wf.add_conditional_edges("query", should_continue_query, {"query_tools": "query_tools", END: END})
    wf.add_edge("query_tools", "query")
    checkpointer = MemorySaver()
    return wf.compile(checkpointer=checkpointer)


# ── Groundedness ────────────────────────────────────────────────────────────

def _extract_graph_facts(tool_data) -> list[dict]:
    facts = []
    items = tool_data if isinstance(tool_data, list) else []
    for item in items:
        label = item.get("label", item.get("content", ""))
        table = item.get("table", "")
        score = item.get("score", 0)
        if label:
            facts.append({"type": "entity", "text": label, "table": table, "score": score})
        for direction in ("outgoing", "incoming"):
            rels = item.get(direction, [])
            if isinstance(rels, dict):
                rels = [rels]
            for r in (rels or []):
                if isinstance(r, dict):
                    for k, v in r.items():
                        if v:
                            facts.append({"type": "relationship", "text": f"{label} --{k}--> {v}"})
    return facts


def _summary_facts(summary: dict) -> list[dict]:
    """Convert table counts into simple fact strings for groundedness scoring."""
    facts = []
    if not isinstance(summary, dict):
        return facts
    for table, count in summary.items():
        if isinstance(count, int):
            label = table.replace("_", " ")
            facts.append({"type": "summary", "text": f"{label} count is {count}"})
    return facts


def _sources_from_search_results(search_results: list[dict]) -> list[dict]:
    sources = []
    for item in search_results:
        if not isinstance(item, dict):
            continue
        label = item.get("label", item.get("content", ""))
        sources.append({
            "entity": label,
            "table": item.get("table", ""),
            "score": item.get("score", 0),
            "id": item.get("id", ""),
        })
    return sources


def _answer_looks_like_tool_call(answer: str) -> bool:
    text = answer.strip().lower()
    if not text:
        return True
    suspicious_markers = [
        "corrected tool call",
        '"name": "search_graph"',
        '"name": "get_graph_summary"',
        '"parameters":',
        '"surql":',
    ]
    return any(marker in text for marker in suspicious_markers)


def _run_query_fallback(question: str) -> dict:
    """Answer from deterministic retrieval when model-side tool use is unreliable."""
    summary_raw = get_graph_summary.invoke({})
    try:
        summary = json.loads(summary_raw)
    except json.JSONDecodeError:
        summary = {}

    search_raw = search_graph.invoke({"query": question, "top_k": 5})
    try:
        search_results = json.loads(search_raw)
        if not isinstance(search_results, list):
            search_results = []
    except json.JSONDecodeError:
        search_results = []

    graph_facts = _extract_graph_facts(search_results) + _summary_facts(summary)
    sources = _sources_from_search_results(search_results)
    tool_steps = [
        {"tool": "get_graph_summary", "args": {}},
        {"tool": "search_graph", "args": {"query": question, "top_k": 5}},
    ]

    context_prompt = (
        "Answer the user's question using ONLY the knowledge graph data below.\n"
        "Do not describe tools or tool calls.\n"
        "If the data is insufficient, say that clearly.\n"
        "Prefer a concise answer.\n\n"
        f"Graph summary:\n{json.dumps(summary, indent=2, sort_keys=True)}\n\n"
        f"Search results:\n{json.dumps(search_results, indent=2)}\n\n"
        f"Question: {question}"
    )

    try:
        response = llm.invoke([
            SystemMessage(content=(
                "You are a legal-document analysis assistant. "
                "Answer only from the provided knowledge graph data."
            )),
            HumanMessage(content=context_prompt),
        ])
        final_answer = str(getattr(response, "content", "")).strip()
    except Exception:
        if sources:
            top = sources[0]
            final_answer = (
                f"Top matching graph record: {top.get('entity', 'Unknown')} "
                f"({top.get('table', 'unknown table')}, id={top.get('id', 'unknown')})."
            )
        else:
            non_zero = {k: v for k, v in summary.items() if v}
            final_answer = (
                "No specific matching records were retrieved. "
                f"Current graph summary: {json.dumps(non_zero or summary, sort_keys=True)}"
            )

    groundedness = compute_groundedness(final_answer, graph_facts) if graph_facts else {
        "groundedness_score": 0, "detail": "No graph data retrieved",
        "matched_facts": [], "token_coverage": 0, "avg_semantic_similarity": 0,
    }

    return {
        "answer": final_answer,
        "groundedness": groundedness,
        "sources": sources,
        "tool_steps": tool_steps,
        "graph_facts_count": len(graph_facts),
    }


def compute_groundedness(answer: str, graph_facts: list[dict]) -> dict:
    if not graph_facts or not answer.strip():
        return {"groundedness_score": 0.0, "detail": "No graph context available",
                "matched_facts": [], "token_coverage": 0.0, "avg_semantic_similarity": 0.0}
    answer_lower = answer.lower()
    answer_tokens = set(re.findall(r"\b[a-z]{2,}\b", answer_lower))
    stop = {"the", "is", "was", "are", "in", "on", "at", "to", "of", "and", "or", "an",
            "it", "he", "she", "his", "her", "by", "for", "with", "that", "this", "from",
            "not", "but", "had", "has", "have", "been", "did", "does", "do"}
    answer_tokens -= stop
    matched_facts = []
    graph_tokens: set[str] = set()
    for fact in graph_facts:
        fact_text = fact["text"].lower()
        fact_words = set(re.findall(r"\b[a-z]{2,}\b", fact_text)) - stop
        graph_tokens |= fact_words
        overlap = fact_words & answer_tokens
        if overlap:
            matched_facts.append({**fact, "matched_words": sorted(overlap)})
    answer_vec = np.array(embeddings.embed_query(answer))
    for mf in matched_facts:
        fv = np.array(embeddings.embed_query(mf["text"]))
        sim = float(np.dot(answer_vec, fv) / (np.linalg.norm(answer_vec) * np.linalg.norm(fv) + 1e-9))
        mf["semantic_similarity"] = round(sim, 3)
    token_overlap = answer_tokens & graph_tokens
    coverage = len(token_overlap) / max(len(answer_tokens), 1)
    avg_sem = (sum(f.get("semantic_similarity", 0) for f in matched_facts) / len(matched_facts)) if matched_facts else 0
    score = round(0.6 * coverage + 0.4 * avg_sem, 3)
    return {
        "groundedness_score": score, "token_coverage": round(coverage, 3),
        "avg_semantic_similarity": round(avg_sem, 3),
        "matched_facts": matched_facts,
        "answer_tokens_matched": sorted(token_overlap),
        "answer_tokens_unmatched": sorted(answer_tokens - graph_tokens),
        "total_graph_facts": len(graph_facts),
    }


# ── High-level API for the web server ──────────────────────────────────────

graph_agent = build_graph()


def ingest_document(text: str) -> Generator[dict, None, None]:
    """Run the full ingest pipeline, yielding progress events."""
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    try:
        ensure_ollama_model_ready()
        reset_database()
        init_schema()

        yield {"type": "status", "message": "Starting ingestion pipeline..."}

        for event in graph_agent.stream(
            {"messages": [HumanMessage(content=f"Analyze this legal document:\n\n{text}")]},
            config=config,
            stream_mode="values",
        ):
            phase = event.get("phase", "")
            last_msg = event["messages"][-1]
            content = str(getattr(last_msg, "content", "")).strip()
            if content:
                yield {"type": "progress", "phase": phase, "message": content}

        state = graph_agent.get_state(config)
        entities_raw = state.values.get("entities_json", "[]")
        triplets_raw = state.values.get("triplets_json", "[]")
        entities = _safe_json_array(entities_raw)
        triplets = _safe_json_array(triplets_raw)

        yield {
            "type": "complete",
            "thread_id": thread_id,
            "entities": len(entities),
            "triplets": len(triplets),
            "entity_details": entities,
            "triplet_details": triplets,
        }
    except Exception as exc:
        yield {"type": "error", "message": str(exc)}


def ask_question(question: str, thread_id: str) -> dict:
    """Ask a question and return answer + groundedness + sources."""
    config = {"configurable": {"thread_id": thread_id}}
    graph_facts: list[dict] = []
    final_answer = ""
    sources: list[dict] = []
    tool_steps: list[dict] = []

    for event in graph_agent.stream(
        {"messages": [HumanMessage(content=question)]},
        config=config,
        stream_mode="values",
    ):
        last_msg = event["messages"][-1]
        role = getattr(last_msg, "type", "unknown")
        tool_calls = getattr(last_msg, "tool_calls", [])

        if tool_calls:
            for tc in tool_calls:
                tool_steps.append({"tool": tc["name"], "args": tc.get("args", {})})

        elif role == "tool":
            raw = str(getattr(last_msg, "content", ""))
            tool_name = getattr(last_msg, "name", "")
            try:
                parsed = json.loads(raw)
                if tool_name == "search_graph":
                    graph_facts = _extract_graph_facts(parsed)
                    for item in (parsed if isinstance(parsed, list) else []):
                        label = item.get("label", item.get("content", ""))
                        sources.append({
                            "entity": label,
                            "table": item.get("table", ""),
                            "score": item.get("score", 0),
                            "id": item.get("id", ""),
                        })
            except json.JSONDecodeError:
                pass

        elif role == "ai" and not tool_calls:
            content = str(getattr(last_msg, "content", "")).strip()
            if content:
                final_answer = content

    used_mutation_tool = any(step.get("tool") == "mutate_graph" for step in tool_steps)
    if not used_mutation_tool and (not graph_facts or _answer_looks_like_tool_call(final_answer)):
        return _run_query_fallback(question)

    groundedness = compute_groundedness(final_answer, graph_facts) if graph_facts else {
        "groundedness_score": 0, "detail": "No graph data retrieved",
        "matched_facts": [], "token_coverage": 0, "avg_semantic_similarity": 0,
    }

    return {
        "answer": final_answer,
        "groundedness": groundedness,
        "sources": sources,
        "tool_steps": tool_steps,
        "graph_facts_count": len(graph_facts),
    }
