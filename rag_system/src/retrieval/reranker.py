import logging
from typing import Any

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name
        self.model = None
        self._load_attempted = False

    def _load_model(self):
        if self._load_attempted:
            return
        self._load_attempted = True
        if not self.model_name:
            return
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading reranker model: {self.model_name}")
            self.model = CrossEncoder(self.model_name)
        except Exception as e:
            logger.warning(f"Failed to load reranker model: {e}")

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []

        self._load_model()

        if self.model is not None:
            try:
                pairs = [(query, c["text"]) for c in candidates]
                scores = self.model.predict(pairs)
                scored = list(zip(candidates, scores))
                scored.sort(key=lambda x: x[1], reverse=True)
                results = []
                for candidate, score in scored[:top_k]:
                    candidate["rerank_score"] = float(score)
                    results.append(candidate)
                return results
            except Exception as e:
                logger.warning(f"Reranking failed: {e}")

        candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
        return candidates[:top_k]
