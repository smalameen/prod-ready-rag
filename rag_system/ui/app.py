import logging
import os
import sys
import tempfile
import uuid
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chunking.chunker import chunk_documents
from src.embeddings.embedder import get_embedding_model
from src.generation.generator import AnswerGenerator
from src.loaders.factory import get_loader
from src.retrieval.pipeline import RetrievalPipeline
from src.retrieval.vector_store import VectorStore
from src.utils.config import BASE_DIR, load_config
from src.utils.logging import LatencyTracker

logger = logging.getLogger(__name__)

DEFAULT_SESSION_ID = "__default__"

DEFAULT_FILES = [
    "history-bd.txt",
    "stripe-history.txt",
    "sample_company_policy.md",
]

config = load_config()
embedder = get_embedding_model(config.get("embedding", {}).get("model"))
vector_store = VectorStore()
retriever = RetrievalPipeline(vector_store, embedder, config)
generator = AnswerGenerator(config)


def ingest_file(file_path: str, session_id: str | None = None) -> int:
    extension = Path(file_path).suffix.lower()
    supported = {".txt", ".md", ".pdf", ".docx", ".csv", ".json", ".parquet"}
    if extension not in supported:
        raise ValueError(f"Unsupported file type: {extension}")

    loader = get_loader(file_path)
    documents = loader.load(file_path)

    chunk_size = config.get("chunking", {}).get("chunk_size", 500)
    overlap = config.get("chunking", {}).get("overlap", 100)
    chunks = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=overlap)

    for c in chunks:
        if session_id:
            c.metadata["session_id"] = session_id
        source = Path(file_path).name
        c.metadata["source_file"] = source

    texts = [c.text for c in chunks]
    embeddings = embedder.embed_documents(texts)
    vector_store.add_documents(chunks, embeddings)
    return len(chunks)


def ensure_defaults_ingested():
    existing_defaults = vector_store.collection.get(
        where={"session_id": {"$eq": DEFAULT_SESSION_ID}},
        include=["metadatas"],
    )
    ingested_files = set(
        m.get("source_file") for m in existing_defaults["metadatas"]
    )
    for fname in DEFAULT_FILES:
        if fname in ingested_files:
            continue
        raw_path = BASE_DIR / "data" / "raw" / fname
        if not raw_path.exists():
            st.warning(f"Default file not found: {fname}")
            continue
        count = ingest_file(str(raw_path), DEFAULT_SESSION_ID)
        st.info(f"Ingested default: {fname} ({count} chunks)")


def get_session_files(session_id: str) -> list[dict]:
    results = vector_store.collection.get(
        where={"session_id": {"$eq": session_id}},
        include=["metadatas"],
    )
    seen: dict[str, dict] = {}
    for i, doc_id in enumerate(results["ids"]):
        meta = results["metadatas"][i]
        fname = meta.get("source_file", "unknown")
        if fname not in seen:
            seen[fname] = {"id": fname, "name": fname, "chunks": 0}
        seen[fname]["chunks"] += 1
    return list(seen.values())


def get_default_files_info() -> list[dict]:
    results = vector_store.collection.get(
        where={"session_id": {"$eq": DEFAULT_SESSION_ID}},
        include=["metadatas"],
    )
    seen: dict[str, dict] = {}
    for i, doc_id in enumerate(results["ids"]):
        meta = results["metadatas"][i]
        fname = meta.get("source_file", "unknown")
        if fname not in seen:
            seen[fname] = {"name": fname, "chunks": 0}
        seen[fname]["chunks"] += 1
    return list(seen.values())


def delete_session_file(session_id: str, filename: str):
    vector_store.delete_by_metadata({
        "$and": [
            {"session_id": {"$eq": session_id}},
            {"source_file": {"$eq": filename}},
        ]
    })


st.set_page_config(
    page_title="RAG Assistant",
    page_icon="",
    layout="wide",
)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "defaults_checked" not in st.session_state:
    st.session_state.defaults_checked = False

ensure_defaults_ingested()

st.title("RAG Assistant")
st.caption("Upload documents and ask questions based on your data.")

with st.sidebar:
    st.header("Default Knowledgebases")
    for df in get_default_files_info():
        st.markdown(f"- **{df['name']}** ({df['chunks']} chunks)")

    st.divider()

    st.header("Your Uploads")
    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["txt", "md", "pdf", "docx", "csv", "json", "parquet"],
    )

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        try:
            count = ingest_file(tmp_path, st.session_state.session_id)
            st.success(f"Ingested '{uploaded_file.name}' ({count} chunks)")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to ingest: {e}")
        finally:
            os.unlink(tmp_path)

    user_files = get_session_files(st.session_state.session_id)
    if user_files:
        for uf in user_files:
            col1, col2 = st.columns([3, 1])
            col1.markdown(f"**{uf['name']}** ({uf['chunks']} ch)")
            if col2.button("", key=f"del_{uf['name']}"):
                delete_session_file(st.session_state.session_id, uf["name"])
                st.rerun()
    else:
        st.caption("No files uploaded yet.")

    st.divider()

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question based on your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            tracker = LatencyTracker(logger)
            sid = st.session_state.session_id
            where = {
                "$or": [
                    {"session_id": {"$eq": DEFAULT_SESSION_ID}},
                    {"session_id": {"$eq": sid}},
                ]
            }
            try:
                chunks = retriever.retrieve(prompt, where=where, tracker=tracker)
                if not chunks:
                    answer = "I could not find this information in the knowledge base."
                else:
                    answer = generator.generate(prompt, chunks, tracker)
                st.markdown(answer)
                st.caption(f"Retrieved {len(chunks)} chunks | {tracker.summary()}")
            except Exception as e:
                st.error(f"Error: {e}")
                answer = f"An error occurred: {e}"

    st.session_state.messages.append({"role": "assistant", "content": answer})
