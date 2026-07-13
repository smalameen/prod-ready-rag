import logging

from src.embeddings.embedder import get_embedding_model
from src.retrieval.vector_store import VectorStore
from src.utils.config import BASE_DIR, load_config
from src.utils.logging import setup_logging


logger = logging.getLogger(__name__)


def main():
    setup_logging()
    config = load_config()

    raw_dir = BASE_DIR / "data" / "raw"

    print("=== Knowledge Base Status ===\n")

    raw_files = sorted(
        f for f in raw_dir.iterdir() if f.is_file() and not f.name.startswith(".")
    ) if raw_dir.exists() else []
    print(f"Files in data/raw/: {len(raw_files)}")
    for f in raw_files:
        size = f.stat().st_size
        print(f"  {f.name:40s} {size:>8,} bytes")
    print()

    vector_store = VectorStore()
    total = vector_store.count()

    from src.ingestion.registry import get_registry
    reg = get_registry()
    print(f"Ingested files: {len(reg)}")
    for name, info in sorted(reg.items()):
        print(f"  {name:40s}  ingested at: {info['ingested_at'][:19]}")
    print()

    print(f"Total chunks in vector database: {total}")
    print()


if __name__ == "__main__":
    main()
