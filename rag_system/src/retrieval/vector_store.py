import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from src.loaders.base import Document
from src.utils.config import DATA_DIR


logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, collection_name: str = "rag_documents"):
        self.persist_dir = str(DATA_DIR / "vectordb" / "chromadb")
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Vector store initialized at {self.persist_dir}")

    def add_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
    ) -> list[str]:
        ids: list[str] = []
        metadatas: list[dict[str, Any]] = []
        texts: list[str] = []

        for doc in documents:
            chunk_id = doc.metadata.get("chunk_id", "")
            ids.append(chunk_id)
            texts.append(doc.text)
            meta = {k: str(v) if not isinstance(v, (str, int, float, bool)) else v
                    for k, v in doc.metadata.items()}
            metadatas.append(meta)

        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids,
        )
        logger.info(f"Added {len(ids)} documents to vector store")
        return ids

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        similarity_threshold: float = 0.75,
        metadata_filter: dict[str, Any] | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if where is None and metadata_filter:
            where = {}
            for k, v in metadata_filter.items():
                where[k] = {"$eq": str(v) if not isinstance(v, (str, int, float, bool)) else v}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        hits: list[dict[str, Any]] = []
        if results["ids"] and results["ids"][0]:
            for idx, doc_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][idx]
                score = 1.0 - distance
                if score >= similarity_threshold:
                    hits.append({
                        "id": doc_id,
                        "text": results["documents"][0][idx],
                        "metadata": results["metadatas"][0][idx],
                        "score": score,
                    })

        return hits

    def delete_documents(self, ids: list[str]):
        if ids:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents from vector store")

    def delete_by_metadata(self, where: dict[str, Any]) -> int:
        results = self.collection.get(where=where)
        ids = results.get("ids", [])
        if ids:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents matching {where}")
        return len(ids)

    def count(self) -> int:
        return self.collection.count()
