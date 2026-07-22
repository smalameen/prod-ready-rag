import logging
import time
import os
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from openai import OpenAI

from src.utils.config import BASE_DIR, get_openrouter_api_key, load_config


logger = logging.getLogger(__name__)

_EMBEDDER_CACHE: dict[str, Any] = {}

DEFAULT_HF_MODEL = "BAAI/bge-small-en-v1.5"


class EmbeddingModel:
    def __init__(self, model_name: str = DEFAULT_HF_MODEL):
        from fastembed import TextEmbedding

        self.model_name = model_name
        logger.info(f"Loading embedding model: {model_name}")
        self.model = TextEmbedding(
            model_name=model_name,
            cache_dir=str(BASE_DIR / "embeddings" / "model_cache"),
            providers=["CPUExecutionProvider"],
        )
        self.dimension = 384
        logger.info(f"Local embedding model loaded. Dimension: {self.dimension}")

    def embed_query(self, text: str) -> list[float]:
        emb = next(self.model.embed(text))
        return emb.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [e.tolist() for e in self.model.embed(texts)]


class ApiEmbeddingModel:
    def __init__(self, model_name: str = "openai/text-embedding-3-small"):
        self.model_name = model_name
        self.dimension = 1536
        cfg = load_config()
        base_url = cfg.get("openrouter", {}).get("base_url", "https://openrouter.ai/api/v1")
        self.client = OpenAI(
            base_url=base_url,
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


class HuggingFaceEmbeddingModel:
    def __init__(self, model_name: str = DEFAULT_HF_MODEL):
        self.model_name = model_name
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        token = os.getenv("HF_TOKEN")
        if not token:
            raise ValueError("HF_TOKEN environment variable is required for HuggingFace embeddings")
        self.headers = {"Authorization": f"Bearer {token}"}
        self.dimension = 384
        logger.info(f"HuggingFace embedding model initialized: {model_name} (dim={self.dimension})")

    def embed_query(self, text: str) -> list[float]:
        result = self._call_api(text)
        return result if isinstance(result[0], (int, float)) else result[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        all_embeddings: list[list[float]] = []
        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            results = self._call_api(batch if len(batch) > 1 else batch[0])
            if results and isinstance(results[0], (int, float)):
                all_embeddings.append(results)
            else:
                all_embeddings.extend(results)
            logger.info(f"Embedded batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}")
            if i + batch_size < len(texts):
                time.sleep(0.5)
        return all_embeddings

    def _call_api(self, inputs: str | list[str]) -> list:
        for attempt in range(3):
            try:
                resp = httpx.post(
                    self.api_url,
                    headers=self.headers,
                    json={"inputs": inputs, "options": {"wait_for_model": True}},
                    timeout=120,
                )
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 503 and attempt < 2:
                    logger.info("Model loading on HF, retrying in 5s...")
                    time.sleep(5)
                    continue
                raise
        return []


def get_embedding_model(model_name: str | None = None) -> EmbeddingModel | ApiEmbeddingModel | HuggingFaceEmbeddingModel:
    cfg = load_config()
    provider = cfg.get("embedding", {}).get("provider", "local") or os.getenv("EMBEDDING_PROVIDER")
    key = model_name or cfg.get("embedding", {}).get("model", DEFAULT_HF_MODEL)

    if key not in _EMBEDDER_CACHE:
        if provider == "hf":
            _EMBEDDER_CACHE[key] = HuggingFaceEmbeddingModel(key)
        elif provider == "api":
            _EMBEDDER_CACHE[key] = ApiEmbeddingModel(key)
        else:
            _EMBEDDER_CACHE[key] = EmbeddingModel(key)
    return _EMBEDDER_CACHE[key]
