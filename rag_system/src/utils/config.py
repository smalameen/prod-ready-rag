import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR)))
CONFIG_DIR = BASE_DIR / "config"
CONFIG_PATH = CONFIG_DIR / "config.yaml"
ENV_PATH = BASE_DIR / ".env"


def load_config() -> dict[str, Any]:
    load_dotenv(ENV_PATH)

    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    provider = os.getenv("LLM_PROVIDER") or config.get("llm_provider", "openrouter")
    config["llm_provider"] = provider

    openrouter = config.get("openrouter", {})
    openrouter.setdefault("model", os.getenv("OPENROUTER_MODEL", "openai/gpt-5-mini"))
    openrouter.setdefault("base_url", os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))
    openrouter.setdefault("temperature", 0.2)
    openrouter.setdefault("max_tokens", 2000)

    ollama = config.get("ollama", {})
    ollama.setdefault("model", os.getenv("OLLAMA_MODEL", "llama3.2"))
    ollama.setdefault("base_url", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama.setdefault("temperature", 0.2)
    ollama.setdefault("max_tokens", 1500)

    embedding = config.get("embedding", {})
    embedding["model"] = os.getenv("EMBEDDING_MODEL") or embedding.get("model", "BAAI/bge-small-en-v1.5")

    retrieval = config.get("retrieval", {})
    retrieval.setdefault("top_k", int(os.getenv("TOP_K", "5")))
    retrieval.setdefault(
        "similarity_threshold", float(os.getenv("SIMILARITY_THRESHOLD", "0.70"))
    )

    chunking = config.get("chunking", {})
    chunking.setdefault("chunk_size", int(os.getenv("CHUNK_SIZE", "500")))
    chunking.setdefault("overlap", int(os.getenv("CHUNK_OVERLAP", "100")))

    vectordb = config.get("vectordb", {})
    vectordb.setdefault("provider", os.getenv("VECTOR_DB", "chromadb"))

    config["openrouter"] = openrouter
    config["ollama"] = ollama
    config["embedding"] = embedding
    config["retrieval"] = retrieval
    config["chunking"] = chunking
    config["vectordb"] = vectordb

    return config


def get_openrouter_api_key() -> str:
    load_dotenv(ENV_PATH)
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise ValueError(
            "OPENROUTER_API_KEY not set. Add it to the .env file."
        )
    return key
