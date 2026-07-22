import logging
import re
import time
from typing import Any

from rank_bm25 import BM25Okapi

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
        self.provider = config.get("vectordb", {}).get("provider", "chromadb")

        self.bm25: BM25Okapi | None = None
        self.bm25_doc_ids: list[str] = []
        self.bm25_doc_texts: list[str] = []
        self.bm25_metadatas: list[dict] = []
        self._bm25_count = -1
        self._rrf_k = 60

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[\w\u0980-\u09FF]+", text.lower())

    def _ensure_bm25_index(self):
        if self.provider == "supabase":
            self.bm25 = None
            self._bm25_count = -1
            return
        current_count = self.vector_store.count()
        if self.bm25 is not None and current_count == self._bm25_count:
            return
        logger.info(f"Building BM25 index (%d docs)...", current_count)
        results = self.vector_store.collection.get(include=["documents", "metadatas"])
        ids = results.get("ids") or []
        texts = results.get("documents") or []
        metadatas = results.get("metadatas") or []
        tokenized = [self._tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(tokenized)
        self.bm25_doc_ids = list(ids)
        self.bm25_doc_texts = list(texts)
        self.bm25_metadatas = list(metadatas)
        self._bm25_count = current_count

    def _bm25_search(
        self,
        query: str,
        top_k: int = 20,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        self._ensure_bm25_index()
        if not self.bm25_doc_ids:
            return []

        valid_ids: set[str] | None = None
        if where is not None:
            filtered = self.vector_store.collection.get(where=where, include=["metadatas"])
            valid_ids = set(filtered.get("ids") or [])

        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        scored = [(i, scores[i]) for i in range(len(scores))]
        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scored:
            if len(results) >= top_k:
                break
            doc_id = self.bm25_doc_ids[idx]
            if valid_ids is not None and doc_id not in valid_ids:
                continue
            if score > 0:
                results.append({
                    "id": doc_id,
                    "text": self.bm25_doc_texts[idx],
                    "score": float(score),
                    "metadata": self.bm25_metadatas[idx] if idx < len(self.bm25_metadatas) else {},
                })
        return results

    def _rrf_fuse(
        self,
        semantic_results: list[dict[str, Any]],
        bm25_results: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        scores: dict[str, float] = {}
        id_to_item: dict[str, dict[str, Any]] = {}

        for i, r in enumerate(semantic_results):
            cid = r["id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (self._rrf_k + i + 1)
            if cid not in id_to_item:
                id_to_item[cid] = dict(r)

        for i, r in enumerate(bm25_results):
            cid = r["id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (self._rrf_k + i + 1)
            if cid not in id_to_item:
                id_to_item[cid] = dict(r)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        fused = []
        for cid, score in ranked[:top_k]:
            item = id_to_item[cid]
            item["hybrid_score"] = score
            fused.append(item)
        return fused

    def retrieve(
        self,
        query: str,
        metadata_filter: dict[str, Any] | None = None,
        where: dict[str, Any] | None = None,
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
            where=where,
        )
        t2 = time.time()
        if tracker:
            tracker.record("vector_search", t2 - t1)

        if self.provider != "supabase":
            bm25_candidates = self._bm25_search(query, top_k=20, where=where)
            t_bm25 = time.time()
            if tracker:
                tracker.record("bm25_search", t_bm25 - t2)

            fused = self._rrf_fuse(candidates, bm25_candidates, top_k=self.top_k * 2)
            t_fuse = time.time()
            if tracker:
                tracker.record("rrf_fuse", t_fuse - t_bm25)
        else:
            fused = candidates
            t_fuse = time.time()

        results = self.reranker.rerank(
            query=query,
            candidates=fused,
            top_k=self.top_k,
        )
        t3 = time.time()
        if tracker:
            tracker.record("rerank", t3 - t_fuse)

        assert results is not None
        return results
