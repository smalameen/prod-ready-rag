import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from src.utils.config import BASE_DIR


logger = logging.getLogger(__name__)

REGISTRY_PATH = BASE_DIR / "data" / "processed" / "registry.json"


def _load_registry() -> dict[str, Any]:
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, "r") as f:
            return json.load(f)
    return {}


def _save_registry(registry: dict[str, Any]):
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


def compute_hash(file_path: str) -> str:
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha.update(block)
    return sha.hexdigest()


def is_ingested(file_path: str) -> bool:
    registry = _load_registry()
    path = Path(file_path)
    entry = registry.get(path.name)
    if entry is None:
        return False
    current_hash = compute_hash(file_path)
    return entry.get("sha256") == current_hash


def mark_ingested(file_path: str):
    registry = _load_registry()
    path = Path(file_path)
    registry[path.name] = {
        "filename": path.name,
        "sha256": compute_hash(file_path),
        "ingested_at": __import__("datetime").datetime.now().isoformat(),
    }
    _save_registry(registry)
    logger.info(f"Marked as ingested: {path.name}")


def get_registry() -> dict[str, Any]:
    return _load_registry()
