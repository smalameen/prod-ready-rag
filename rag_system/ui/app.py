import logging
import os
import re
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

logging.getLogger("streamlit.watcher").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner").setLevel(logging.ERROR)

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chunking.chunker import chunk_documents
from src.embeddings.embedder import get_embedding_model
from src.generation.generator import AnswerGenerator
from src.loaders.base import Document
from src.loaders.factory import get_loader
from src.retrieval.pipeline import RetrievalPipeline
from src.retrieval.vector_store import VectorStore
from src.utils.config import BASE_DIR, load_config
from src.utils.logging import LatencyTracker

logger = logging.getLogger(__name__)

DEFAULT_SESSION_ID = "__default__"
BENGALI_RANGE = re.compile(r"[\u0980-\u09FF]")
VISIBLE_DEFAULTS = {"fifa_rule_book.pdf", "history-kfc.txt", "stripe-history.txt"}
TTL_HOURS = int(os.getenv("TTL_HOURS", "24"))


def has_bengali(text: str) -> bool:
    return bool(BENGALI_RANGE.search(text))


@st.cache_resource
def get_services():
    cfg = load_config()
    local_emb = get_embedding_model("BAAI/bge-small-en-v1.5")
    api_emb = get_embedding_model("openai/text-embedding-3-small")
    vs_en = VectorStore(collection_name="user_docs_en")
    vs_multi = VectorStore(collection_name="user_docs_multi")
    ret_en = RetrievalPipeline(vs_en, local_emb, cfg)
    ret_multi = RetrievalPipeline(vs_multi, api_emb, cfg)
    gen = AnswerGenerator(cfg)
    return local_emb, api_emb, vs_en, vs_multi, ret_en, ret_multi, gen, cfg


(
    local_embedder,
    api_embedder,
    vs_en,
    vs_multi,
    ret_en,
    ret_multi,
    generator,
    config,
) = get_services()


@st.cache_resource
def start_cleanup_thread(_vs_en, _vs_multi):
    t = threading.Thread(target=cleanup_old_chunks, args=(_vs_en, _vs_multi), daemon=True)
    t.start()
    return True


def cleanup_old_chunks(vs_en, vs_multi, interval_seconds: int = 1800):
    while True:
        try:
            cutoff = datetime.now(timezone.utc).timestamp() - TTL_HOURS * 3600
            for vs in (vs_en, vs_multi):
                results = vs.collection.get(
                    where={"session_id": {"$ne": "__default__"}},
                    include=["metadatas"],
                )
                to_delete: list[str] = []
                for i, meta in enumerate(results["metadatas"]):
                    created = meta.get("created_at", "")
                    if created:
                        try:
                            created_ts = datetime.fromisoformat(created).timestamp()
                            if created_ts < cutoff:
                                to_delete.append(results["ids"][i])
                        except ValueError:
                            to_delete.append(results["ids"][i])
                if to_delete:
                    vs.collection.delete(ids=to_delete)
                    logger.info(f"TTL cleanup: deleted {len(to_delete)} chunks from {vs.collection.name}")
        except Exception as e:
            logger.error(f"TTL cleanup error: {e}")
        time.sleep(interval_seconds)


start_cleanup_thread(vs_en, vs_multi)


def pick_embedder(text: str):
    if has_bengali(text):
        return api_embedder, vs_multi, ret_multi, "multilingual"
    return local_embedder, vs_en, ret_en, "english"


def ingest_text(text: str, title: str, session_id: str) -> int:
    emb, vs, _, model_type = pick_embedder(text)
    documents = [Document(text=text, metadata={"source_file": title})]
    chunk_size = config.get("chunking", {}).get("chunk_size", 500)
    overlap = config.get("chunking", {}).get("overlap", 100)
    chunks = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=overlap)
    now = datetime.now(timezone.utc).isoformat()
    for c in chunks:
        c.metadata["session_id"] = session_id
        c.metadata["source_file"] = title
        c.metadata["embedding_model"] = model_type
        c.metadata["created_at"] = now
    texts = [c.text for c in chunks]
    embeddings = emb.embed_documents(texts)
    vs.add_documents(chunks, embeddings)
    return len(chunks)


def ingest_file(file_path: str, session_id: str | None = None) -> int:
    extension = Path(file_path).suffix.lower()
    supported = {".txt", ".md", ".pdf", ".docx", ".csv", ".json", ".parquet"}
    if extension not in supported:
        raise ValueError(f"Unsupported file type: {extension}")

    loader = get_loader(file_path)
    documents = loader.load(file_path)
    full_text = "\n".join(d.text for d in documents)
    emb, vs, _, model_type = pick_embedder(full_text)

    chunk_size = config.get("chunking", {}).get("chunk_size", 500)
    overlap = config.get("chunking", {}).get("overlap", 100)
    chunks = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=overlap)

    now = datetime.now(timezone.utc).isoformat()
    for c in chunks:
        if session_id:
            c.metadata["session_id"] = session_id
        source = Path(file_path).name
        c.metadata["source_file"] = source
        c.metadata["embedding_model"] = model_type
        c.metadata["created_at"] = now

    texts = [c.text for c in chunks]
    embeddings = emb.embed_documents(texts)
    vs.add_documents(chunks, embeddings)
    return len(chunks)


def get_all_raw_files() -> list[Path]:
    raw = BASE_DIR / "data" / "raw"
    if not raw.exists():
        return []
    return sorted(p for p in raw.iterdir() if p.is_file() and not p.name.startswith("."))


def ensure_defaults_ingested():
    if st.session_state.defaults_checked:
        return
    ingested_files: set[str] = set()
    for vs in (vs_en, vs_multi):
        existing = vs.collection.get(
            where={"session_id": {"$eq": DEFAULT_SESSION_ID}},
            include=["metadatas"],
        )
        ingested_files.update(m.get("source_file") for m in existing["metadatas"])
    for raw_path in get_all_raw_files():
        fname = raw_path.name
        if fname in ingested_files:
            continue
        with st.spinner(f"Ingesting: {fname}..."):
            loader = get_loader(str(raw_path))
            documents = loader.load(str(raw_path))
            full_text = "\n".join(d.text for d in documents)
            is_bn = has_bengali(full_text)
            emb = api_embedder if is_bn else local_embedder
            vs = vs_multi if is_bn else vs_en
            chunk_size = config.get("chunking", {}).get("chunk_size", 500)
            overlap = config.get("chunking", {}).get("overlap", 100)
            chunks = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=overlap)
            for c in chunks:
                c.metadata["session_id"] = DEFAULT_SESSION_ID
                c.metadata["source_file"] = fname
            texts = [c.text for c in chunks]
            embeddings = emb.embed_documents(texts)
            vs.add_documents(chunks, embeddings)
    st.session_state.defaults_checked = True


def get_user_files(session_id: str) -> list[dict]:
    seen: dict[str, dict] = {}
    for vs in (vs_en, vs_multi):
        results = vs.collection.get(
            where={"session_id": {"$eq": session_id}},
            include=["metadatas"],
        )
        for i, meta in enumerate(results["metadatas"]):
            fname = meta.get("source_file", "unknown")
            if fname not in seen:
                seen[fname] = {"id": fname, "name": fname, "chunks": 0, "model": meta.get("embedding_model", "?")}
            seen[fname]["chunks"] += 1
    return list(seen.values())


def get_default_files_info() -> list[dict]:
    seen: dict[str, dict] = {}
    for vs in (vs_en, vs_multi):
        results = vs.collection.get(
            where={"session_id": {"$eq": DEFAULT_SESSION_ID}},
            include=["metadatas"],
        )
        for i, meta in enumerate(results["metadatas"]):
            fname = meta.get("source_file", "unknown")
            if fname not in VISIBLE_DEFAULTS:
                continue
            if fname not in seen:
                seen[fname] = {"name": fname, "chunks": 0}
            seen[fname]["chunks"] += 1
    return list(seen.values())


def delete_session_file(session_id: str, filename: str):
    for vs in (vs_en, vs_multi):
        vs.delete_by_metadata({
            "$and": [
                {"session_id": {"$eq": session_id}},
                {"source_file": {"$eq": filename}},
            ]
        })


st.set_page_config(
    page_title="RAG Assistant",
    page_icon="\U0001f50d",
    layout="wide",
)

st.markdown("""
<style>
.stApp { background: #ffffff !important; font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Noto Sans,Helvetica,Arial,sans-serif; }
.stApp h1, .stApp h2, .stApp h3, .stMarkdown, .stApp p, .stApp li, .stApp span, .st-cb { color: #0d1117 !important; }
.stApp header, .stApp .st-bb, .stApp [data-testid="stSidebar"] { background: #f6f8fa !important; border-color: #d0d7de !important; }
.stApp [data-testid="stSidebar"] { border-right: 1px solid #d0d7de !important; }
.stApp [data-testid="stFileUploader"] { background: #ffffff !important; }
.stApp [data-testid="stFileUploader"] > div, .stApp [data-testid="stFileUploadDropzone"] { background: #ffffff !important; border: 1px solid #d0d7de !important; border-radius: 12px !important; padding: 1rem !important; }
.stApp [data-testid="stFileUploadDropzone"] * { background: transparent !important; color: #0d1117 !important; }
.stApp [data-testid="stFileUploadDropzone"] button { background: #ff6b35 !important; color: #fff !important; border: none !important; border-radius: 6px !important; }
.stApp [data-testid="stFileUploader"] [data-testid="stWidgetLabel"] { color: #0d1117 !important; }
.stApp div[data-testid="stChatMessage"] { background: #ffffff !important; border: 1px solid #d0d7de !important; border-radius: 12px !important; padding: 1rem !important; margin: 0.5rem 0 !important; }
.stApp [data-testid="stChatInput"] { background: transparent !important; }
.stApp [data-testid="stChatInput"] > div, .stApp [data-testid="stChatInput"] > div > div { background: #ffffff !important; border: 1px solid #d0d7de !important; border-radius: 12px !important; }
.stApp [data-testid="stChatInput"] textarea, .stApp [data-testid="stChatInput"] input { background: transparent !important; color: #0d1117 !important; caret-color: #0d1117 !important; }
.stApp .stTextInput input, .stApp .stTextArea textarea { background: #ffffff !important; color: #0d1117 !important; border: 1px solid #d0d7de !important; border-radius: 8px !important; }
.stApp .stButton > button { background: #ff6b35 !important; color: #fff !important; border: none !important; border-radius: 8px !important; font-weight: 600 !important; }
.stApp .stButton > button:hover { background: #e55a2b !important; }
.stApp .stAlert { border-radius: 8px !important; }
.stApp [data-testid="stPopoverButton"] > div:first-child { border: 2px dashed #d0d7de !important; border-radius: 12px !important; padding: 0.75rem !important; background: #f6f8fa !important; color: #0d1117 !important; font-weight: 600 !important; text-align: center !important; transition: all 0.2s !important; }
.stApp [data-testid="stPopoverButton"] > div:first-child:hover { border-color: #ff6b35 !important; background: #fff4f0 !important; }
a { color: #ff6b35 !important; }
.stApp .st-emotion-cache-1ln1c1s, .stApp .st-emotion-cache-10oheav { color: #656d76 !important; }
.stApp .st-b7 { border-color: #d0d7de !important; }
.stApp header button[kind="headerNoPadding"], .stApp header button[title*="Menu"], .stApp header [data-testid="stToolbar"] { display: none !important; }
.stApp header, .stApp header div[data-testid="stDecoration"] { display: none !important; }
.stApp section.main > div:first-child { padding-top: 0 !important; margin-top: 0 !important; }
.stApp .block-container { padding-top: 0 !important; margin-top: 0 !important; }
</style>
""", unsafe_allow_html=True)

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

if st.session_state.dark_mode:
    st.markdown("""
<style>
.stApp { background: #0d1117 !important; }
.stApp h1, .stApp h2, .stApp h3, .stMarkdown, .stApp p, .stApp li, .stApp span, .st-cb { color: #e6edf3 !important; }
.stApp header, .stApp .st-bb, .stApp [data-testid="stSidebar"] { background: #161b22 !important; border-color: #30363d !important; }
.stApp [data-testid="stSidebar"] { border-right: 1px solid #30363d !important; }
.stApp [data-testid="stFileUploader"] { background: #0d1117 !important; }
.stApp [data-testid="stFileUploader"] > div, .stApp [data-testid="stFileUploadDropzone"] { background: #21262d !important; border-color: #30363d !important; }
.stApp [data-testid="stFileUploadDropzone"] * { color: #e6edf3 !important; }
.stApp [data-testid="stFileUploadDropzone"] button { background: #ff6b35 !important; }
.stApp [data-testid="stFileUploader"] [data-testid="stWidgetLabel"] { color: #e6edf3 !important; }
.stApp div[data-testid="stChatMessage"] { background: #21262d !important; border-color: #30363d !important; }
.stApp [data-testid="stChatInput"] > div, .stApp [data-testid="stChatInput"] > div > div { background: #21262d !important; border-color: #30363d !important; }
.stApp [data-testid="stChatInput"] textarea, .stApp [data-testid="stChatInput"] input { color: #e6edf3 !important; caret-color: #e6edf3 !important; }
.stApp .stTextInput input, .stApp .stTextArea textarea { background: #21262d !important; color: #e6edf3 !important; border-color: #30363d !important; }
.stApp .st-emotion-cache-1ln1c1s, .stApp .st-emotion-cache-10oheav { color: #8b949e !important; }
.stApp .st-b7 { border-color: #30363d !important; }
.stApp [data-testid="stPopoverButton"] > div:first-child { border-color: #30363d !important; background: #161b22 !important; color: #e6edf3 !important; }
.stApp [data-testid="stPopoverButton"] > div:first-child:hover { border-color: #ff6b35 !important; background: #1c2333 !important; }
</style>
""", unsafe_allow_html=True)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "defaults_checked" not in st.session_state:
    st.session_state.defaults_checked = False
if "processed_uploads" not in st.session_state:
    st.session_state.processed_uploads = set()

ensure_defaults_ingested()

st.title("RAG Assistant")
st.caption("Upload documents and ask questions based on your data.")

with st.sidebar:
    st.header("Built-in Knowledgebases")
    for df in get_default_files_info():
        st.markdown(f"- **{df['name']}** ({df['chunks']} chunks)")
    st.caption("Ask about anything \u2014 all files are searchable.")

    st.divider()

    st.header("Your Uploads")
    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["txt", "md", "pdf", "docx", "csv", "json", "parquet"],
        key="file_uploader",
    )

    if uploaded_file and uploaded_file.name not in st.session_state.processed_uploads:
        st.session_state.processed_uploads.add(uploaded_file.name)
        preview = uploaded_file.getvalue().decode("utf-8", errors="replace")[:2000]
        lang = "multilingual" if has_bengali(preview) else "english"
        st.info(f"Detected: **{lang}** model")
        with st.spinner(f"Ingesting '{uploaded_file.name}'..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            try:
                count = ingest_file(tmp_path, st.session_state.session_id)
                st.success(f"Ingested '{uploaded_file.name}' ({count} chunks) using {lang} model")
            except Exception as e:
                st.error(f"Failed to ingest: {e}")
            finally:
                os.unlink(tmp_path)

    with st.popover("\u2795 Add text content", use_container_width=True):
        text_title = st.text_input("Title", key="text_title", placeholder="My Notes")
        text_content = st.text_area("Paste your content here", key="text_content", height=150)
        if st.button("Add Text", key="add_text_btn"):
            if not text_content.strip():
                st.warning("Please enter some text.")
            elif not text_title.strip():
                st.warning("Please enter a title.")
            elif text_title in st.session_state.processed_uploads:
                st.warning(f"'{text_title}' already exists.")
            else:
                lang = "multilingual" if has_bengali(text_content) else "english"
                st.session_state.processed_uploads.add(text_title)
                with st.spinner(f"Ingesting '{text_title}'..."):
                    try:
                        count = ingest_text(text_content, text_title, st.session_state.session_id)
                        st.success(f"Ingested '{text_title}' ({count} chunks) using {lang} model")
                    except Exception as e:
                        st.error(f"Failed to ingest: {e}")

    user_files = get_user_files(st.session_state.session_id)
    if user_files:
        for uf in user_files:
            col1, col2 = st.columns([3, 1])
            label = f"**{uf['name']}** ({uf['chunks']} ch, {uf['model']})"
            col1.markdown(label)
            if col2.button("\U0001f5d1", key=f"del_{uf['name']}"):
                delete_session_file(st.session_state.session_id, uf["name"])
                st.session_state.processed_uploads.discard(uf["name"])
                st.rerun()
    else:
        st.caption("No files uploaded yet.")

    st.divider()

    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    with col_b:
        dark = st.toggle("\U0001f319", st.session_state.dark_mode, key="theme_toggle")
        if dark != st.session_state.dark_mode:
            st.session_state.dark_mode = dark
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
            _, _, ret, _ = pick_embedder(prompt)
            sid = st.session_state.session_id
            where = {
                "$or": [
                    {"session_id": {"$eq": DEFAULT_SESSION_ID}},
                    {"session_id": {"$eq": sid}},
                ]
            }
            try:
                chunks = ret.retrieve(prompt, where=where, tracker=tracker)
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
