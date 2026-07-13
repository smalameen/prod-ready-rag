from src.chunking.chunker import chunk_documents
from src.loaders.base import Document


class TestChunker:
    def test_chunk_small_document(self):
        doc = Document(text="Hello world.", metadata={"source_file": "test.txt"})
        chunks = chunk_documents([doc], chunk_size=500, chunk_overlap=50)
        assert len(chunks) == 1
        assert chunks[0].text == "Hello world."

    def test_chunk_large_document(self):
        text = " ".join(["word"] * 1000)
        doc = Document(text=text, metadata={"source_file": "test.txt"})
        chunks = chunk_documents([doc], chunk_size=100, chunk_overlap=20)
        assert len(chunks) > 1
        for c in chunks:
            assert "chunk_id" in c.metadata
            assert "chunk_index" in c.metadata

    def test_chunk_id_uniqueness(self):
        docs = [
            Document(text="First document.", metadata={"source_file": "a.txt"}),
            Document(text="Second document.", metadata={"source_file": "b.txt"}),
        ]
        chunks = chunk_documents(docs)
        ids = [c.metadata["chunk_id"] for c in chunks]
        assert len(set(ids)) == len(ids)
