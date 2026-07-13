# Universal Local RAG System

A terminal-based Retrieval Augmented Generation (RAG) system that ingests documents, generates local embeddings, and answers questions using OpenRouter LLMs.

## Features

- **Multi-format ingestion**: txt, md, pdf, docx, csv, json, parquet
- **Local embeddings**: BAAI/bge-small-en-v1.5 via sentence-transformers (no API cost)
- **Persistent vector store**: ChromaDB with metadata filtering
- **Reranking**: BAAI/bge-reranker-base for improved retrieval quality
- **OpenRouter LLM**: Switch models without code changes
- **Incremental ingestion**: SHA256-based dedup, skip unchanged files
- **Conversation memory**: Rolling window of 5 exchanges
- **Source attribution**: Every answer cites its sources
- **Docker support**: Containerized deployment

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

Copy `.env` and set your OpenRouter API key:

```bash
# Already configured - verify OPENROUTER_API_KEY in .env
```

Supported models (via OpenRouter):
- `openai/gpt-5`, `openai/gpt-5-mini`
- `anthropic/claude-sonnet-4`
- `google/gemini-2.5-pro`
- `meta-llama/llama-4`
- `qwen/qwen3`, `deepseek/deepseek-chat`
- `mistralai/mistral-large`

### 3. Ingest documents

Place your documents in `data/raw/`, then:

```bash
python ingest.py
```

### 4. Query

Single question:
```bash
python query.py
```

Interactive chat:
```bash
python main.py
```

## Usage

### Ingestion

```bash
python ingest.py
```

Scans `data/raw/`, processes new/modified files, skips duplicates.

### Query Mode

```bash
python query.py
```

Enter a question and get an answer with source references.

### Chat Mode

```bash
python main.py
```

Interactive session with conversation memory. Type `exit` to quit.

## Configuration

Edit `config/config.yaml` or `.env`:

| Setting | Default | Description |
|---------|---------|-------------|
| OPENROUTER_MODEL | openai/gpt-5-mini | LLM model |
| EMBEDDING_MODEL | BAAI/bge-small-en-v1.5 | Local embedding model |
| CHUNK_SIZE | 500 | Character chunk size |
| CHUNK_OVERLAP | 100 | Chunk overlap |
| TOP_K | 5 | Retrieved chunks |
| SIMILARITY_THRESHOLD | 0.75 | Minimum similarity score |

## Project Structure

```
rag_system/
├── data/raw/         # Input documents
├── config/           # YAML configuration
├── src/              # Source code
│   ├── loaders/      # Document loaders
│   ├── chunking/     # Text chunking
│   ├── embeddings/   # Local embedding model
│   ├── retrieval/    # Vector store + reranker
│   ├── generation/   # OpenRouter LLM wrapper
│   ├── ingestion/    # Ingestion pipeline
│   ├── memory/       # Conversation memory
│   └── utils/        # Logging, config
├── tests/            # Unit tests
├── vectordb/         # Persistent ChromaDB
├── ingest.py         # Ingestion entry point
├── query.py          # Query entry point
├── main.py           # Chat entry point
└── Dockerfile        # Container support
```

## Docker

```bash
make docker-build
make docker-ingest
make docker-run
```

## Development

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

## Architecture

```
Documents → Loaders → Chunking → Embeddings → Vector DB → Retriever → OpenRouter → Answer
```
