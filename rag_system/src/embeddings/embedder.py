import logging
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.config import BASE_DIR


logger = logging.getLogger(__name__)

_EMBEDDER_CACHE: dict[str, "EmbeddingModel"] = {}


class EmbeddingModel:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        cache_folder = str(BASE_DIR / "embeddings" / "model_cache")
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(
            model_name,
            cache_folder=cache_folder,
        )
        self.dimension = self.model.get_embedding_dimension()
        logger.info(f"Embedding model loaded. Dimension: {self.dimension}")

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        return [e.tolist() for e in embeddings]


def get_embedding_model(model_name: str | None = None) -> EmbeddingModel:
    key = model_name or "BAAI/bge-small-en-v1.5"
    if key not in _EMBEDDER_CACHE:
        _EMBEDDER_CACHE[key] = EmbeddingModel(key)
    return _EMBEDDER_CACHE[key]
