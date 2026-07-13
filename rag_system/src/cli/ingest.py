import logging
import time

from pathlib import Path

from src.embeddings.embedder import get_embedding_model
from src.ingestion.pipeline import run_ingestion
from src.retrieval.vector_store import VectorStore
from src.utils.config import BASE_DIR, load_config
from src.utils.logging import LatencyTracker, setup_logging


logger = logging.getLogger(__name__)


def main():
    setup_logging()
    config = load_config()

    embedder = get_embedding_model(config.get("embedding", {}).get("model"))
    vector_store = VectorStore()

    raw_dir = BASE_DIR / "data" / "raw"

    if not raw_dir.exists():
        print("data/raw directory not found.")
        return

    all_files = sorted(f for f in raw_dir.iterdir() if f.is_file() and not f.name.startswith("."))
    if not all_files:
        print("No files found in data/raw/.")
        return

    print(f"Found {len(all_files)} file(s) in data/raw/:")
    for f in all_files:
        print(f"  - {f.name}")
    print()

    tracker = LatencyTracker(logger)
    t_start = time.time()

    total = run_ingestion(embedder, vector_store, config, tracker)

    t_end = time.time()

    print()
    print(f"Done. Added {total} new chunk(s) in {t_end - t_start:.1f}s.")
    print()


if __name__ == "__main__":
    main()
