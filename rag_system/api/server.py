import logging
import os
import re
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chunking.chunker import chunk_documents
from src.embeddings.embedder import get_embedding_model
from src.generation.generator import create_generator
from src.loaders.base import Document
from src.loaders.factory import get_loader
from src.memory.conversation import ConversationMemory
from src.retrieval.pipeline import RetrievalPipeline
from src.retrieval.vector_store import VectorStore
from src.utils.config import BASE_DIR, load_config
from src.utils.logging import LatencyTracker, setup_logging

logger = logging.getLogger(__name__)

config = load_config()
embedder = get_embedding_model(config["embedding"]["model"])
vs_en = VectorStore(collection_name="user_docs_en")
vs_multi = VectorStore(collection_name="user_docs_multi")
ret_en = RetrievalPipeline(vs_en, embedder, config)
ret_multi = RetrievalPipeline(vs_multi, embedder, config)
generator = create_generator(config)

sessions: dict[str, ConversationMemory] = {}
sessions_lock = threading.Lock()

DEFAULT_SESSION_ID = "__default__"
BENGALI_RANGE = re.compile(r"[\u0980-\u09FF]")
VISIBLE_DEFAULTS = {"fifa_rule_book.pdf", "history-kfc.txt", "stripe-history.txt", "world-history.pdf"}
TTL_HOURS = int(os.getenv("TTL_HOURS", "24"))

app = FastAPI(title="RAG Assistant API")


def has_bengali(text: str) -> bool:
    return bool(BENGALI_RANGE.search(text))


def ensure_utf8(text: str) -> str:
    if not text:
        return text
    try:
        fixed = text.encode("latin-1").decode("utf-8")
        if has_bengali(fixed) or fixed.isascii():
            return fixed
    except (UnicodeEncodeError, UnicodeDecodeError, LookupError):
        pass
    return text


def pick_store(language: str = "en"):
    if language == "bn":
        return embedder, vs_multi, ret_multi, "local"
    return embedder, vs_en, ret_en, "local"


def get_or_create_memory(session_id: str) -> ConversationMemory:
    with sessions_lock:
        if session_id not in sessions:
            sessions[session_id] = ConversationMemory(window_size=5)
        return sessions[session_id]


def get_all_raw_files() -> list[Path]:
    raw = BASE_DIR / "data" / "raw"
    if not raw.exists():
        return []
    return sorted(p for p in raw.iterdir() if p.is_file() and not p.name.startswith("."))


def parse_sources(answer_text: str) -> tuple[str, list[str]]:
    parts = answer_text.split("\n\nSources:\n")
    answer = parts[0]
    sources: list[str] = []
    if len(parts) > 1:
        for line in parts[1].strip().split("\n"):
            m = re.match(r"\d+\.\s*(.+)", line)
            if m:
                sources.append(m.group(1))
    return answer, sources


def ingest_text(text: str, title: str, session_id: str, language: str = "en") -> tuple[int, str]:
    emb, vs, _, model_type = pick_store(language)
    documents = [Document(text=text, metadata={"source_file": title})]
    chunk_size = config.get("chunking", {}).get("chunk_size", 500)
    overlap = config.get("chunking", {}).get("overlap", 100)
    chunks = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=overlap)
    now = datetime.now(timezone.utc).isoformat()
    for c in chunks:
        c.metadata["session_id"] = session_id
        c.metadata["source_file"] = title
        c.metadata["embedding_model"] = model_type
        c.metadata["created_at"] = now
    if not chunks:
        raise ValueError("No text could be extracted from this content.")

    texts = [c.text for c in chunks]
    embeddings = emb.embed_documents(texts)
    vs.add_documents(chunks, embeddings)
    return len(chunks), model_type


def ingest_file(file_path: str, session_id: str | None = None, source_name: str | None = None, language: str = "en") -> tuple[int, str]:
    extension = Path(file_path).suffix.lower()
    supported = {".txt", ".md", ".pdf", ".docx", ".csv", ".json", ".parquet"}
    if extension not in supported:
        raise ValueError(f"Unsupported file type: {extension}")

    loader = get_loader(file_path)
    documents = loader.load(file_path)
    emb, vs, _, model_type = pick_store(language)

    chunk_size = config.get("chunking", {}).get("chunk_size", 500)
    overlap = config.get("chunking", {}).get("overlap", 100)
    chunks = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=overlap)

    now = datetime.now(timezone.utc).isoformat()
    source = source_name or Path(file_path).name
    for c in chunks:
        if session_id:
            c.metadata["session_id"] = session_id
        c.metadata["source_file"] = source
        c.metadata["embedding_model"] = model_type
        c.metadata["created_at"] = now

    if not chunks:
        raise ValueError("No text could be extracted from this file. It may be empty or contain only non-extractable content (e.g., scanned images without OCR).")

    texts = [c.text for c in chunks]
    embeddings = emb.embed_documents(texts)
    vs.add_documents(chunks, embeddings)
    return len(chunks), model_type


def ensure_defaults_ingested():
    ingested_files: set[str] = set()
    for vs in (vs_en, vs_multi):
        existing = vs.collection.get(
            where={"session_id": {"$eq": DEFAULT_SESSION_ID}},
            include=["metadatas"],
        )
        ingested_files.update(m.get("source_file") for m in existing["metadatas"])
    for raw_path in get_all_raw_files():
        fname = raw_path.name
        if fname in ingested_files:
            continue
        try:
            logger.info(f"Ingesting default file: {fname}")
            loader = get_loader(str(raw_path))
            documents = loader.load(str(raw_path))
            full_text = "\n".join(d.text for d in documents)
            if not full_text.strip():
                logger.warning(f"Skipping {fname}: no extractable text")
                continue
            lang = "bn" if has_bengali(full_text) else "en"
            _, vs, _, _ = pick_store(lang)
            chunk_size = config.get("chunking", {}).get("chunk_size", 500)
            overlap = config.get("chunking", {}).get("overlap", 100)
            chunks = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=overlap)
            if not chunks:
                logger.warning(f"Skipping {fname}: no chunks produced")
                continue
            for c in chunks:
                c.metadata["session_id"] = DEFAULT_SESSION_ID
                c.metadata["source_file"] = fname
            texts = [c.text for c in chunks]
            embeddings = embedder.embed_documents(texts)
            vs.add_documents(chunks, embeddings)
        except Exception as e:
            logger.error(f"Failed to ingest {fname}: {e}")


def cleanup_old_chunks(vs_en_: VectorStore, vs_multi_: VectorStore, interval_seconds: int = 1800):
    while True:
        try:
            cutoff = datetime.now(timezone.utc).timestamp() - TTL_HOURS * 3600
            for vs in (vs_en_, vs_multi_):
                results = vs.collection.get(
                    where={"session_id": {"$ne": "__default__"}},
                    include=["metadatas"],
                )
                to_delete: list[str] = []
                for i, meta in enumerate(results["metadatas"]):
                    created = meta.get("created_at", "")
                    if created:
                        try:
                            created_ts = datetime.fromisoformat(created).timestamp()
                            if created_ts < cutoff:
                                to_delete.append(results["ids"][i])
                        except ValueError:
                            to_delete.append(results["ids"][i])
                if to_delete:
                    vs.collection.delete(ids=to_delete)
                    logger.info(f"TTL cleanup: deleted {len(to_delete)} chunks from {vs.collection.name}")
        except Exception as e:
            logger.error(f"TTL cleanup error: {e}")
        time.sleep(interval_seconds)


@app.on_event("startup")
def startup():
    ensure_defaults_ingested()
    ret_en._ensure_bm25_index()
    ret_multi._ensure_bm25_index()
    logger.info("BM25 indexes built: en=%d multi=%d", ret_en._bm25_count, ret_multi._bm25_count)
    t = threading.Thread(target=cleanup_old_chunks, args=(vs_en, vs_multi), daemon=True)
    t.start()
    logger.info("API server started")


@app.get("/api/status")
def get_status():
    raw_dir = BASE_DIR / "data" / "raw"
    raw_files = sorted(
        f for f in raw_dir.iterdir() if f.is_file() and not f.name.startswith(".")
    ) if raw_dir.exists() else []

    from src.ingestion.registry import get_registry
    reg = get_registry()

    total_chunks = vs_en.count() + vs_multi.count()

    return {
        "raw_files": [{"name": f.name, "size_bytes": f.stat().st_size} for f in raw_files],
        "ingested_files_count": len(reg),
        "total_chunks": total_chunks,
    }


@app.get("/api/files/default")
def get_default_files():
    seen: dict[str, dict] = {}
    for vs in (vs_en, vs_multi):
        results = vs.collection.get(
            where={"session_id": {"$eq": DEFAULT_SESSION_ID}},
            include=["metadatas"],
        )
        for meta in results["metadatas"]:
            fname = meta.get("source_file", "unknown")
            if fname not in VISIBLE_DEFAULTS:
                continue
            if fname not in seen:
                seen[fname] = {"name": fname, "chunks": 0}
            seen[fname]["chunks"] += 1
    return {"files": list(seen.values())}


@app.get("/api/files/user")
def get_user_files(session_id: str):
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    seen: dict[str, dict] = {}
    for vs in (vs_en, vs_multi):
        results = vs.collection.get(
            where={"session_id": {"$eq": session_id}},
            include=["metadatas"],
        )
        for meta in results["metadatas"]:
            fname = meta.get("source_file", "unknown")
            if fname not in seen:
                seen[fname] = {
                    "id": fname,
                    "name": fname,
                    "chunks": 0,
                    "model": meta.get("embedding_model", "?"),
                }
            seen[fname]["chunks"] += 1
    return {"files": list(seen.values())}


@app.delete("/api/files/user")
def delete_user_file(session_id: str, filename: str):
    if not session_id or not filename:
        raise HTTPException(status_code=400, detail="session_id and filename are required")
    total = 0
    for vs in (vs_en, vs_multi):
        total += vs.delete_by_metadata({
            "$and": [
                {"session_id": {"$eq": session_id}},
                {"source_file": {"$eq": filename}},
            ]
        })
    return {"deleted_chunks": total}


@app.post("/api/chat")
def chat(message: str = Form(...), session_id: str = Form(...)):
    if not message or not session_id:
        raise HTTPException(status_code=400, detail="message and session_id are required")
    message = ensure_utf8(message)

    memory = get_or_create_memory(session_id)
    tracker = LatencyTracker(logger)

    where = {
        "$or": [
            {"session_id": {"$eq": DEFAULT_SESSION_ID}},
            {"session_id": {"$eq": session_id}},
        ]
    }

    try:
        chunks_en = ret_en.retrieve(message, where=where, tracker=tracker)
        chunks_multi = ret_multi.retrieve(message, where=where, tracker=tracker)

        seen_ids = set()
        combined = []
        for c in chunks_en + chunks_multi:
            cid = c.get("id", "")
            if cid not in seen_ids:
                seen_ids.add(cid)
                combined.append(c)
        combined.sort(key=lambda x: x.get("score", 0), reverse=True)
        top_k = config.get("retrieval", {}).get("top_k", 5)
        chunks = combined[:top_k]

        if not chunks:
            answer = "I could not find this information in the knowledge base."
            sources: list[str] = []
        else:
            raw_answer = generator.generate(message, chunks, tracker)
            answer, sources = parse_sources(raw_answer)

        memory.add_exchange(message, answer)
        latency_summary = tracker.summary()

        return {
            "answer": answer,
            "sources": sources,
            "chunks_retrieved": len(chunks),
            "latency_summary": latency_summary,
        }
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query")
def query_(message: str = Form(...), session_id: str = Form(...)):
    if not message:
        raise HTTPException(status_code=400, detail="message is required")
    message = ensure_utf8(message)

    tracker = LatencyTracker(logger)

    where = None
    if session_id:
        where = {
            "$or": [
                {"session_id": {"$eq": DEFAULT_SESSION_ID}},
                {"session_id": {"$eq": session_id}},
            ]
        }

    try:
        chunks_en = ret_en.retrieve(message, where=where, tracker=tracker)
        chunks_multi = ret_multi.retrieve(message, where=where, tracker=tracker)
        seen_ids = set()
        combined = []
        for c in chunks_en + chunks_multi:
            cid = c.get("id", "")
            if cid not in seen_ids:
                seen_ids.add(cid)
                combined.append(c)
        combined.sort(key=lambda x: x.get("score", 0), reverse=True)
        top_k = config.get("retrieval", {}).get("top_k", 5)
        chunks = combined[:top_k]

        if not chunks:
            return {
                "answer": "I could not find this information in the knowledge base.",
                "sources": [],
                "chunks_retrieved": 0,
                "latency_summary": tracker.summary(),
            }
        raw_answer = generator.generate(message, chunks, tracker)
        answer, sources = parse_sources(raw_answer)
        return {
            "answer": answer,
            "sources": sources,
            "chunks_retrieved": len(chunks),
            "latency_summary": tracker.summary(),
        }
    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ingest/file")
async def ingest_file_api(file: UploadFile = File(...), session_id: str = Form(...), language: str = Form("en")):
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    suffix = Path(file.filename).suffix.lower()
    supported = {".txt", ".md", ".pdf", ".docx", ".csv", ".json", ".parquet"}
    if suffix not in supported:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        count, model_type = ingest_file(tmp_path, session_id, source_name=file.filename, language=language)
        return {"chunks_added": count, "model_type": model_type, "filename": file.filename}
    except Exception as e:
        logger.error(f"Ingest file error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/api/ingest/text")
def ingest_text_api(
    content: str = Form(...),
    title: str = Form(...),
    session_id: str = Form(...),
    language: str = Form("en"),
):
    if not content.strip():
        raise HTTPException(status_code=400, detail="Content cannot be empty")
    if not title.strip():
        raise HTTPException(status_code=400, detail="Title cannot be empty")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    try:
        count, model_type = ingest_text(content, title, session_id, language)
        return {"chunks_added": count, "model_type": model_type, "title": title}
    except Exception as e:
        logger.error(f"Ingest text error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/chat/history")
def get_chat_history(session_id: str):
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    memory = get_or_create_memory(session_id)
    return {"history": memory.get_history()}


@app.post("/api/chat/clear")
def clear_chat_history(session_id: str = Form(...)):
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    memory = get_or_create_memory(session_id)
    memory.clear()
    return {"cleared": True}


STATIC_DIR = Path(__file__).parent.parent / "static"
if STATIC_DIR.exists():
    from fastapi.responses import FileResponse

    @app.get("/static/{path:path}")
    async def serve_static(path: str):
        file_path = STATIC_DIR / path
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(STATIC_DIR / "index.html"))

    @app.get("/")
    async def serve_index():
        return FileResponse(str(STATIC_DIR / "index.html"))

    @app.exception_handler(404)
    async def not_found_handler(request, exc):
        path = request.url.path
        if path.startswith("/api/"):
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        return FileResponse(str(STATIC_DIR / "index.html"))
