"""
agent.py — Soreal Legal Knowledge Graph RAG Agent
Extracted from knowledge_graph_rag.ipynb for use by the Flask web app.
"""

from __future__ import annotations

import json
import os
import re
import socket
import time
import urllib.request
from collections import defaultdict
from typing import Annotated, TypedDict
from urllib.parse import urlparse

import numpy as np
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from surrealdb import Surreal

# ── Load environment ─────────────────────────────────────────────────────────

load_dotenv()

os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "soreal-kg-agent")

SURREAL_URL = os.getenv("SURREAL_URL", "ws://localhost:8000/rpc")
SURREAL_USERNAME = os.getenv("SURREAL_USERNAME", "root")
SURREAL_PASSWORD = os.getenv("SURREAL_PASSWORD", "root")
SURREAL_NAMESPACE = os.getenv("SURREAL_NAMESPACE", "soreal")
SURREAL_DATABASE = os.getenv("SURREAL_DATABASE", "kg")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

# ── Entity / predicate catalogues ────────────────────────────────────────────

VALID_ENTITY_TYPES = [
    "party", "representative", "clause", "obligation", "right", "restriction",
    "condition", "definition", "key_date", "payment_term", "fee",
    "liability_cap", "risk", "compliance_requirement", "section", "flag",
    "amendment", "document", "term",
]

ENTITY_TYPE_TO_TABLE: dict[str, str] = {t: t for t in VALID_ENTITY_TYPES}
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

# ── SurrealDB schema ─────────────────────────────────────────────────────────

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
    "DEFINE TABLE IF NOT EXISTS fee SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS label       ON fee TYPE string",
    "DEFINE FIELD IF NOT EXISTS amount      ON fee TYPE option<string>",
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
    # Edge tables (relations)
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
    "DEFINE TABLE IF NOT EXISTS similar_to        TYPE RELATION SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS source_chunk ON similar_to TYPE option<string>",
    "DEFINE TABLE IF NOT EXISTS sourced_from      TYPE RELATION SCHEMAFULL",
    "DEFINE FIELD IF NOT EXISTS source_chunk ON sourced_from TYPE option<string>",
    # Vector indexes (HNSW)
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

# ── Prompts ───────────────────────────────────────────────────────────────────

LEGAL_NER_PROMPT = """\
You are a legal-document NER system.
Given a text chunk from a contract or legal document, extract every entity.

Allowed entity types: {entity_types}

Return **only** valid JSON – a list of objects:
[
  {{"entity_type": "...", "name": "...", "properties": {{...}} }}
]

For each entity:
- `name` is a short, descriptive label.
- `properties` may include fields like description, amount, severity, date_value, etc.

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

Return **only** valid JSON – a list of triplet objects:
[
  {{"subject": "entity_name", "predicate": "...", "object": "entity_name",
    "source_chunk": "chunk_id"}}
]

Rules:
- Only use entities from the list above as subject / object.
- Only use predicates from the allowed list.
- Each triplet must include the source_chunk id for traceability.

Text chunk (id = {chunk_id}):
\"\"\"
{text}
\"\"\"
"""

# ── Module-level singletons (initialised lazily by init_agent) ────────────────

conn: Surreal | None = None
embeddings: HuggingFaceEmbeddings | None = None
llm: ChatOllama | None = None
graph = None  # LangGraph compiled graph
_initialised = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _wait_for_surreal(url: str, timeout: float = 15.0) -> None:
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 8000
    deadline = time.time() + timeout
    last_err = None
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                return
        except OSError as e:
            last_err = e
            time.sleep(0.5)
    raise ConnectionError(
        f"SurrealDB not reachable at {url}. "
        "Start it with: surreal start --user root --pass root memory"
    ) from last_err


def _check_ollama(base_url: str, model: str) -> None:
    try:
        req = urllib.request.Request(f"{base_url}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            names = [m.get("name", "") for m in data.get("models", [])]
            if not any(model in n for n in names):
                raise ConnectionError(
                    f"Model '{model}' not found in Ollama. "
                    f"Run: ollama pull {model}. Available: {names}"
                )
    except urllib.error.URLError as exc:
        raise ConnectionError(
            f"Ollama not reachable at {base_url}. Start it with: ollama serve"
        ) from exc


def _make_id(name: str) -> str:
    """Sanitise an entity name into a valid SurrealDB record-id fragment."""
    return re.sub(r"[^a-z0-9_]", "_", name.lower())[:60]


# ── LangChain tools ───────────────────────────────────────────────────────────

@tool
def ingest_text(text: str, source: str = "legal_doc") -> str:
    """Split a legal document into overlapping chunks, embed and store them."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_text(text)
    records = []
    for i, c in enumerate(chunks):
        vec = embeddings.embed_query(c)
        conn.query(
            "CREATE type::thing($tb, $id) SET content=$content, "
            "source=$source, chunk_index=$idx, vector=$vec;",
            {"tb": "chunk", "id": f"{source}_{i}",
             "content": c, "source": source, "idx": i, "vec": vec},
        )
        records.append({"chunk_id": f"chunk:{source}_{i}", "length": len(c)})
    return json.dumps({"status": "ok", "chunks_created": len(records), "details": records})


@tool
def extract_legal_entities(chunks_json: str) -> str:
    """Run legal NER over every stored chunk and return merged entities JSON."""
    chunks_info = json.loads(chunks_json)
    all_entities: list[dict] = []
    seen: set[tuple] = set()
    for info in chunks_info.get("details", chunks_info):
        cid = info["chunk_id"] if isinstance(info, dict) else info
        short = cid.split(":")[-1] if ":" in cid else cid
        rows = conn.query(
            "SELECT content FROM type::thing($tb, $id);",
            {"tb": "chunk", "id": short},
        )
        if not rows or not rows[0]:
            continue
        text = rows[0][0]["content"] if isinstance(rows[0], list) else rows[0].get("content", "")
        prompt = LEGAL_NER_PROMPT.format(
            entity_types=", ".join(VALID_ENTITY_TYPES), text=text
        )
        resp = llm.invoke(prompt)
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
    return json.dumps(all_entities)


@tool
def form_legal_triplets(entities_json: str, chunks_json: str) -> str:
    """Extract relation triplets between entities for every chunk."""
    entities = json.loads(entities_json)
    chunks_info = json.loads(chunks_json)
    all_triplets: list[dict] = []
    for info in chunks_info.get("details", chunks_info):
        cid = info["chunk_id"] if isinstance(info, dict) else info
        short = cid.split(":")[-1] if ":" in cid else cid
        rows = conn.query(
            "SELECT content FROM type::thing($tb, $id);",
            {"tb": "chunk", "id": short},
        )
        if not rows or not rows[0]:
            continue
        text = rows[0][0]["content"] if isinstance(rows[0], list) else rows[0].get("content", "")
        prompt = LEGAL_TRIPLET_PROMPT.format(
            entities=json.dumps(entities[:60]),
            predicates=", ".join(VALID_PREDICATES),
            chunk_id=cid, text=text,
        )
        resp = llm.invoke(prompt)
        raw = resp.content if hasattr(resp, "content") else str(resp)
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            continue
        try:
            trips = json.loads(match.group())
        except json.JSONDecodeError:
            continue
        for t in trips:
            if t.get("predicate") in VALID_PREDICATES:
                all_triplets.append(t)
    return json.dumps(all_triplets)


@tool
def load_entities_and_triplets(entities_json: str, triplets_json: str) -> str:
    """Upsert legal entities (with vectors) and create relation edges."""
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
        for k in props:
            fields += f", {k}=$prop_{k}"
        params: dict = {"tb": table, "id": rid, "label": label, "vec": vec}
        for k, v in props.items():
            params[f"prop_{k}"] = v
        conn.query(f"CREATE type::thing($tb, $id) SET {fields};", params)
        ent_count += 1

    edge_count = 0
    for t in triplets:
        pred = t.get("predicate")
        if pred not in VALID_PREDICATES:
            continue
        subj_name = t.get("subject", "")
        obj_name = t.get("object", "")
        subj_type = next((e["entity_type"] for e in entities if e["name"] == subj_name), None)
        obj_type = next((e["entity_type"] for e in entities if e["name"] == obj_name), None)
        if not subj_type or not obj_type:
            continue
        subj_table = ENTITY_TYPE_TO_TABLE.get(subj_type)
        obj_table = ENTITY_TYPE_TO_TABLE.get(obj_type)
        if not subj_table or not obj_table:
            continue
        subj_id = f"{subj_table}:{_make_id(subj_name)}"
        obj_id = f"{obj_table}:{_make_id(obj_name)}"
        sc = t.get("source_chunk", "")
        conn.query(
            f"RELATE type::thing($from) ->{pred}-> type::thing($to) SET source_chunk=$sc;",
            {"from": subj_id, "to": obj_id, "sc": sc},
        )
        edge_count += 1

    return json.dumps({"entities_loaded": ent_count, "edges_created": edge_count})


@tool
def search_graph(query: str, top_k: int = 5) -> str:
    """Semantic vector search across all vectorised KG tables + one-hop graph walk."""
    qvec = embeddings.embed_query(query)
    results: list[dict] = []
    for tbl in VECTORIZED_TABLES:
        rows = conn.query(
            f"SELECT *, vector::similarity::cosine(vector, $qvec) AS score "
            f"FROM {tbl} WHERE vector != NONE "
            f"ORDER BY score DESC LIMIT $k;",
            {"qvec": qvec, "k": top_k},
        )
        flat = rows[0] if rows and isinstance(rows[0], list) else rows
        for r in (flat or []):
            results.append({
                "table": tbl,
                "id": str(r.get("id", "")),
                "label": r.get("label", r.get("content", "")),
                "score": r.get("score", 0),
            })
    results.sort(key=lambda x: x["score"], reverse=True)
    top = results[:top_k]
    enriched = []
    for r in top:
        rid = r["id"]
        out_rows = conn.query(f"SELECT ->? AS rels FROM {rid};")
        in_rows = conn.query(f"SELECT <-? AS rels FROM {rid};")
        r["outgoing"] = out_rows[0] if out_rows else []
        r["incoming"] = in_rows[0] if in_rows else []
        enriched.append(r)
    return json.dumps(enriched, default=str)


@tool
def mutate_graph(surql: str) -> str:
    """Execute a SurrealQL mutation (CREATE / RELATE / UPDATE / DELETE)."""
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
    tables = (
        list(set(ENTITY_TYPE_TO_TABLE.values()))
        + ["chunk", "document", "paragraph"]
        + VALID_PREDICATES
    )
    counts: dict[str, int] = {}
    for t in set(tables):
        try:
            r = conn.query(f"SELECT count() FROM {t} GROUP ALL;")
            flat = r[0] if r and isinstance(r[0], list) else r
            counts[t] = flat[0].get("count", 0) if flat else 0
        except Exception:
            counts[t] = 0
    return json.dumps(counts)


ALL_TOOLS = [
    ingest_text,
    extract_legal_entities,
    form_legal_triplets,
    load_entities_and_triplets,
    search_graph,
    mutate_graph,
    get_graph_summary,
]

QUERY_TOOLS = [search_graph, mutate_graph, get_graph_summary]

# ── LangGraph agent ───────────────────────────────────────────────────────────


class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    chunks_json: str
    entities_json: str
    triplets_json: str
    phase: str


def _latest_human_message(state: AgentState):
    return next(
        (m for m in reversed(state.get("messages", [])) if isinstance(m, HumanMessage)),
        None,
    )


def entry_node(state: AgentState) -> dict:
    return {}


def route_from_entry(state: AgentState) -> str:
    msg = _latest_human_message(state)
    if msg and "Analyze this legal document:" in msg.content:
        return "ingest"
    return "query"


def ingest_node(state: AgentState) -> dict:
    msg = _latest_human_message(state)
    if msg is None:
        return {"messages": [AIMessage(content="No input text found.")], "phase": "ingest"}
    content = msg.content
    marker = "Analyze this legal document:"
    text = content.split(marker, 1)[1].strip() if marker in content else content.strip()
    chunks_json = ingest_text.invoke({"text": text, "source": "legal_doc"})
    info = json.loads(chunks_json)
    return {
        "chunks_json": chunks_json,
        "messages": [AIMessage(content=f"Stored {info.get('chunks_created', 0)} chunks in SurrealDB.")],
        "phase": "ingest_completed",
    }


def extract_entities_node(state: AgentState) -> dict:
    chunks_json = state.get("chunks_json", "")
    if not chunks_json:
        return {"entities_json": "[]",
                "messages": [AIMessage(content="No chunks available.")],
                "phase": "extract_completed"}
    entities_json = extract_legal_entities.invoke({"chunks_json": chunks_json})
    count = len(json.loads(entities_json))
    return {
        "entities_json": entities_json,
        "messages": [AIMessage(content=f"Extracted {count} legal entities.")],
        "phase": "extract_completed",
    }


def form_triplets_node(state: AgentState) -> dict:
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


def load_node(state: AgentState) -> dict:
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


def query_node(state: AgentState) -> dict:
    bound = llm.bind_tools(QUERY_TOOLS)
    system = SystemMessage(
        content=(
            "You are a legal-document analysis assistant backed by a knowledge graph. "
            "RULES:\n"
            "1. For counting/overview questions, call get_graph_summary FIRST.\n"
            "2. For specific questions about parties, clauses, obligations, etc., use search_graph.\n"
            "3. If the user provides corrections or new facts, use mutate_graph.\n"
            "4. ALWAYS answer based on data returned by tools — do NOT guess.\n"
            "5. Cite the source chunk ID when possible for traceability."
        )
    )
    response = bound.invoke([system] + state.get("messages", []))
    return {"messages": [response], "phase": "query"}


query_tools_executor = ToolNode(QUERY_TOOLS)


def should_continue_query(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "query_tools"
    return END


def _build_graph() -> object:
    workflow = StateGraph(AgentState)
    workflow.add_node("entry", entry_node)
    workflow.add_node("ingest", ingest_node)
    workflow.add_node("extract_entities", extract_entities_node)
    workflow.add_node("form_triplets", form_triplets_node)
    workflow.add_node("load", load_node)
    workflow.add_node("query", query_node)
    workflow.add_node("query_tools", query_tools_executor)
    workflow.add_edge(START, "entry")
    workflow.add_conditional_edges("entry", route_from_entry,
                                   {"ingest": "ingest", "query": "query"})
    workflow.add_edge("ingest", "extract_entities")
    workflow.add_edge("extract_entities", "form_triplets")
    workflow.add_edge("form_triplets", "load")
    workflow.add_edge("load", END)
    workflow.add_conditional_edges("query", should_continue_query,
                                   {"query_tools": "query_tools", END: END})
    workflow.add_edge("query_tools", "query")
    return workflow.compile(checkpointer=MemorySaver())


# ── Public API ────────────────────────────────────────────────────────────────

def init_agent() -> None:
    """Connect to SurrealDB and Ollama, load embeddings, compile the graph."""
    global conn, embeddings, llm, graph, _initialised

    if _initialised:
        return

    _wait_for_surreal(SURREAL_URL)
    _check_ollama(OLLAMA_BASE_URL, OLLAMA_MODEL)

    conn = Surreal(SURREAL_URL)
    conn.signin({"username": SURREAL_USERNAME, "password": SURREAL_PASSWORD})
    conn.use(SURREAL_NAMESPACE, SURREAL_DATABASE)

    for stmt in SCHEMA_STATEMENTS:
        conn.query(stmt + ";")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)

    graph = _build_graph()
    _initialised = True


def is_ready() -> bool:
    return _initialised


# ── Groundedness computation ──────────────────────────────────────────────────

def _extract_graph_facts(tool_data) -> list[dict]:
    """Pull individual facts from search_graph output."""
    facts: list[dict] = []
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
                            facts.append({"type": "relationship",
                                          "text": f"{label} --{k}--> {v}",
                                          "table": table, "score": score})
    return facts


def compute_groundedness(answer: str, graph_facts: list[dict]) -> dict:
    """Score how well the answer is grounded in graph knowledge."""
    if not graph_facts or not answer.strip():
        return {
            "groundedness_score": 0.0,
            "token_coverage": 0.0,
            "avg_semantic_similarity": 0.0,
            "matched_facts": [],
            "answer_tokens_matched": [],
            "answer_tokens_unmatched": [],
            "total_graph_facts": len(graph_facts),
            "verdict": "No graph context available",
            "verdict_level": "none",
        }

    stop = {
        "the", "is", "was", "are", "in", "on", "at", "to", "of", "and", "or",
        "an", "it", "he", "she", "his", "her", "by", "for", "with", "that",
        "this", "from", "not", "but", "had", "has", "have", "been", "did",
        "does", "do", "a",
    }

    answer_lower = answer.lower()
    answer_tokens = set(re.findall(r"\b[a-z]{2,}\b", answer_lower)) - stop

    matched_facts: list[dict] = []
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
        fvec = np.array(embeddings.embed_query(mf["text"]))
        cos = float(
            np.dot(answer_vec, fvec)
            / (np.linalg.norm(answer_vec) * np.linalg.norm(fvec) + 1e-9)
        )
        mf["semantic_similarity"] = round(cos, 3)

    token_overlap = answer_tokens & graph_tokens
    token_coverage = len(token_overlap) / max(len(answer_tokens), 1)

    avg_semantic = 0.0
    if matched_facts:
        avg_semantic = sum(f.get("semantic_similarity", 0) for f in matched_facts) / len(matched_facts)

    score = round(0.6 * token_coverage + 0.4 * avg_semantic, 3)

    if score >= 0.7:
        verdict = "Well-grounded — answer closely follows graph knowledge"
        level = "high"
    elif score >= 0.4:
        verdict = "Partially grounded — some facts from graph, some from LLM"
        level = "medium"
    else:
        verdict = "Weakly grounded — answer mostly relies on LLM knowledge"
        level = "low"

    return {
        "groundedness_score": score,
        "token_coverage": round(token_coverage, 3),
        "avg_semantic_similarity": round(avg_semantic, 3),
        "matched_facts": sorted(matched_facts, key=lambda x: -x.get("semantic_similarity", 0)),
        "answer_tokens_matched": sorted(token_overlap),
        "answer_tokens_unmatched": sorted(answer_tokens - graph_tokens),
        "total_graph_facts": len(graph_facts),
        "verdict": verdict,
        "verdict_level": level,
    }


def ask_question(question: str, thread_config: dict) -> dict:
    """
    Stream the agent for a Q&A question and return:
      {answer, sources, groundedness, steps}
    """
    graph_facts: list[dict] = []
    final_answer = ""
    steps: list[dict] = []
    sources: list[dict] = []

    for event in graph.stream(
        {"messages": [HumanMessage(content=question)]},
        config=thread_config,
        stream_mode="values",
    ):
        last_msg = event["messages"][-1]
        role = getattr(last_msg, "type", "unknown")
        tool_calls = getattr(last_msg, "tool_calls", [])

        if tool_calls:
            for tc in tool_calls:
                steps.append({"step": "tool_call", "tool": tc["name"],
                               "args": tc.get("args", {})})

        elif role == "tool":
            tool_name = getattr(last_msg, "name", "unknown")
            raw = str(getattr(last_msg, "content", ""))
            try:
                parsed = json.loads(raw)
                if tool_name == "search_graph":
                    graph_facts = _extract_graph_facts(parsed)
                    # Build source references from top hits
                    for item in (parsed if isinstance(parsed, list) else []):
                        label = item.get("label", item.get("content", ""))
                        score = item.get("score", 0)
                        table = item.get("table", "")
                        if label and score > 0.1:
                            sources.append({
                                "label": label,
                                "table": table,
                                "score": round(float(score), 3),
                            })
                steps.append({"step": "tool_result", "tool": tool_name,
                               "facts_count": len(graph_facts)})
            except json.JSONDecodeError:
                steps.append({"step": "tool_result", "tool": tool_name, "raw": raw[:200]})

        elif role == "ai" and not tool_calls:
            content = str(getattr(last_msg, "content", "")).strip()
            if content:
                final_answer = content

    groundedness = compute_groundedness(final_answer, graph_facts)

    # De-duplicate and sort sources
    seen_labels: set[str] = set()
    unique_sources: list[dict] = []
    for s in sorted(sources, key=lambda x: -x["score"]):
        if s["label"] not in seen_labels:
            seen_labels.add(s["label"])
            unique_sources.append(s)

    return {
        "answer": final_answer,
        "sources": unique_sources,
        "groundedness": groundedness,
        "steps": steps,
    }


def run_ingest_pipeline(text: str, thread_config: dict) -> dict:
    """Ingest a document through the full pipeline and return a summary."""
    steps: list[str] = []
    for event in graph.stream(
        {"messages": [HumanMessage(content=f"Analyze this legal document:\n\n{text}")]},
        config=thread_config,
        stream_mode="values",
    ):
        phase = event.get("phase", "")
        last_msg = event["messages"][-1]
        content = str(getattr(last_msg, "content", "")).strip()
        if content:
            steps.append(content)

    entities_json = ""
    triplets_json = ""
    try:
        state = graph.get_state(thread_config)
        entities_json = state.values.get("entities_json", "[]") or "[]"
        triplets_json = state.values.get("triplets_json", "[]") or "[]"
    except Exception:
        pass

    entity_count = len(json.loads(entities_json))
    triplet_count = len(json.loads(triplets_json))

    return {
        "status": "ok",
        "steps": steps,
        "entities_extracted": entity_count,
        "triplets_formed": triplet_count,
    }
