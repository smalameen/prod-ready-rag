from src.retrieval.reranker import Reranker


class TestReranker:
    def test_rerank_empty(self):
        reranker = Reranker()
        results = reranker.rerank("test", [], top_k=5)
        assert results == []

    def test_rerank_preserves_fields(self):
        reranker = Reranker()
        candidates = [
            {"id": "1", "text": "First", "score": 0.9},
            {"id": "2", "text": "Second", "score": 0.8},
        ]
        results = reranker.rerank("test", candidates, top_k=5)
        assert len(results) == 2
        for r in results:
            assert "rerank_score" in r
