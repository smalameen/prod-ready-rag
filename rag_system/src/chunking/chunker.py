import uuid
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.loaders.base import Document


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunked_docs: list[Document] = []
    for doc in documents:
        texts = text_splitter.split_text(doc.text)
        for i, chunk_text in enumerate(texts):
            chunk_meta: dict[str, Any] = {
                **doc.metadata,
                "chunk_id": str(uuid.uuid4()),
                "chunk_index": i,
            }
            chunked_docs.append(Document(text=chunk_text, metadata=chunk_meta))

    return chunked_docs
