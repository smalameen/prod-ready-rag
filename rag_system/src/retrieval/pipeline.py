import logging
import time
from typing import Any

from src.embeddings.embedder import EmbeddingModel
from src.retrieval.reranker import Reranker
from src.retrieval.vector_store import VectorStore
from src.utils.logging import LatencyTracker


logger = logging.getLogger(__name__)


class RetrievalPipeline:
    def __init__(
        self,
        vector_store: VectorStore,
        embedder: EmbeddingModel,
        config: dict[str, Any],
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = config.get("retrieval", {}).get("top_k", 5)
        self.similarity_threshold = config.get("retrieval", {}).get("similarity_threshold", 0.75)
        self.reranker = Reranker()

    def retrieve(
        self,
        query: str,
        metadata_filter: dict[str, Any] | None = None,
        tracker: LatencyTracker | None = None,
    ) -> list[dict[str, Any]]:
        t0 = time.time()
        query_embedding = self.embedder.embed_query(query)
        t1 = time.time()
        if tracker:
            tracker.record("embed_query", t1 - t0)

        candidates = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=20,
            similarity_threshold=self.similarity_threshold,
            metadata_filter=metadata_filter,
        )
        t2 = time.time()
        if tracker:
            tracker.record("vector_search", t2 - t1)

        results = self.reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=self.top_k,
        )
        t3 = time.time()
        if tracker:
            tracker.record("rerank", t3 - t2)

        assert results is not None
        return results
