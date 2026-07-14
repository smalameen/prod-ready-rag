import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from src.utils.config import BASE_DIR, get_openrouter_api_key


logger = logging.getLogger(__name__)

_EMBEDDER_CACHE: dict[str, Any] = {}


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
        logger.info(f"Local embedding model loaded. Dimension: {self.dimension}")

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        return [e.tolist() for e in embeddings]


class ApiEmbeddingModel:
    def __init__(self, model_name: str = "openai/text-embedding-3-small"):
        self.model_name = model_name
        self.dimension = 1536
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=get_openrouter_api_key(),
        )
        logger.info(f"API embedding model initialized: {model_name} (dim={self.dimension})")

    def embed_query(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            model=self.model_name,
            input=text,
        )
        return response.data[0].embedding

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        all_embeddings: list[list[float]] = []
        batch_size = 20
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch,
            )
            indexed = {e.index: e.embedding for e in response.data}
            all_embeddings.extend(indexed[j] for j in range(len(batch)))
            logger.info(f"Embedded batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}")
        return all_embeddings


def get_embedding_model(model_name: str | None = None) -> EmbeddingModel | ApiEmbeddingModel:
    key = model_name or "BAAI/bge-small-en-v1.5"
    if key not in _EMBEDDER_CACHE:
        if key.startswith("openai/"):
            _EMBEDDER_CACHE[key] = ApiEmbeddingModel(key)
        else:
            _EMBEDDER_CACHE[key] = EmbeddingModel(key)
    return _EMBEDDER_CACHE[key]
