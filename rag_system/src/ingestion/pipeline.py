import logging
import time
from pathlib import Path
from typing import Any

from src.chunking.chunker import chunk_documents
from src.embeddings.embedder import EmbeddingModel
from src.ingestion.registry import is_ingested, mark_ingested
from src.loaders.factory import get_loader
from src.loaders.base import Document
from src.retrieval.vector_store import VectorStore
from src.utils.config import BASE_DIR
from src.utils.logging import LatencyTracker


logger = logging.getLogger(__name__)


def run_ingestion(
    embedder: EmbeddingModel,
    vector_store: VectorStore,
    config: dict[str, Any],
    tracker: LatencyTracker | None = None,
) -> int:
    raw_dir = BASE_DIR / "data" / "raw"
    if not raw_dir.exists():
        logger.warning(f"Data directory not found: {raw_dir}")
        return 0

    file_paths = sorted(
        p for p in raw_dir.iterdir()
        if p.is_file() and not p.name.startswith(".")
    )
    logger.info(f"Found {len(file_paths)} files in data/raw")

    total_chunks = 0

    for file_path in file_paths:
        try:
            if is_ingested(str(file_path)):
                logger.info(f"Skipping (already ingested): {file_path.name}")
                print(f"  [SKIP] {file_path.name}  (already ingested)")
                continue

            print(f"  [INGEST] {file_path.name} ...", end=" ", flush=True)

            t0 = time.time()
            loader = get_loader(str(file_path))
            documents: list[Document] = loader.load(str(file_path))
            t1 = time.time()
            if tracker:
                tracker.record(f"load_{file_path.name}", t1 - t0)
            chunk_size = config.get("chunking", {}).get("chunk_size", 500)
            overlap = config.get("chunking", {}).get("overlap", 100)
            chunks = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=overlap)
            t2 = time.time()
            if tracker:
                tracker.record(f"chunk_{file_path.name}", t2 - t1)

            texts = [c.text for c in chunks]
            embeddings = embedder.embed_documents(texts)
            t3 = time.time()
            if tracker:
                tracker.record(f"embed_{file_path.name}", t3 - t2)

            vector_store.add_documents(chunks, embeddings)
            mark_ingested(str(file_path))
            total_chunks += len(chunks)

            print(f"{len(chunks)} chunks")

        except Exception as e:
            print(f"FAILED ({e})")
            logger.error(f"Failed to process {file_path.name}: {e}", exc_info=True)
            continue

    logger.info(f"Ingestion complete. Total chunks added: {total_chunks}")
    return total_chunks
