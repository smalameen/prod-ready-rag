class TestEmbeddingModel:
    def test_embed_query_returns_vector(self):
        from src.embeddings.embedder import get_embedding_model
        model = get_embedding_model()
        vec = model.embed_query("Hello world")
        assert isinstance(vec, list)
        assert len(vec) > 0
        assert all(isinstance(v, float) for v in vec)

    def test_embed_documents_returns_vectors(self):
        from src.embeddings.embedder import get_embedding_model
        model = get_embedding_model()
        vecs = model.embed_documents(["Hello", "World"])
        assert len(vecs) == 2
        assert len(vecs[0]) > 0
