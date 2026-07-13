# Terminal Based RAG System PRD and Implementation Guide

## Project Name

Universal Local RAG System

## Objective

Build a Retrieval Augmented Generation (RAG) system capable of ingesting text from multiple sources and formats, converting them into embeddings, storing them in a vector database, and retrieving relevant information during question answering.

The first version does not require a frontend, API server, or web interface. The entire system should run locally from the terminal.

The system must support incremental document ingestion, metadata filtering, source tracking, conversational querying, and model switching without code changes.

All LLM requests and API based model interactions must use OpenRouter as the single gateway provider.

---

# Architecture Requirements

The system should follow the architecture below:

```text
Documents
    ↓
Document Loaders
    ↓
Chunking Pipeline
    ↓
Embedding Model
    ↓
Vector Database
    ↓
Retriever
    ↓
OpenRouter LLM
    ↓
Answer Generation
```

---

# Tech Stack

## Programming Language

Python 3.12+

## Framework

Preferred:

* LangChain

Alternative:

* LlamaIndex

The implementation should remain modular enough to switch frameworks later.

---

## Embedding Model

Default:

```text
BAAI/bge-small-en-v1.5
```

Alternative:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Requirements:

* Local embedding generation
* No API dependency
* Fast execution
* Low memory usage

Embeddings should always run locally to reduce cost and latency.

---

## Vector Database

Preferred:

```text
ChromaDB
```

Alternative:

```text
FAISS
```

Requirements:

* Persistent local storage
* Metadata filtering
* Similarity search
* Incremental updates

---

## LLM Provider

All LLM calls must go through OpenRouter.

Supported models should include:

```text
openai/gpt-5
openai/gpt-5-mini
anthropic/claude-sonnet-4
google/gemini-2.5-pro
meta-llama/llama-4
qwen/qwen3
deepseek/deepseek-chat
mistralai/mistral-large
```

The model should be selectable through configuration without requiring code changes.

No direct integration should exist for:

* OpenAI API
* Anthropic API
* Gemini API
* Groq API
* Together API

The application should communicate only with OpenRouter.

---

# Environment Variables

The project must use a `.env` file.

Example:

```env
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_MODEL=openai/gpt-5-mini

EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

VECTOR_DB=chromadb

CHUNK_SIZE=500
CHUNK_OVERLAP=100

TOP_K=5

SIMILARITY_THRESHOLD=0.75
```

---

# Project Structure

```text
rag_system/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── archive/
│
├── embeddings/
│
├── vectordb/
│
├── cache/
│
├── logs/
│
├── config/
│   └── config.yaml
│
├── src/
│   ├── loaders/
│   ├── chunking/
│   ├── embeddings/
│   ├── retrieval/
│   ├── generation/
│   ├── ingestion/
│   ├── memory/
│   ├── cli/
│   └── utils/
│
├── tests/
│
├── ingest.py
├── query.py
├── main.py
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

# Supported Input Sources

The ingestion pipeline must support:

## Documents

* txt
* md
* pdf
* docx

## Structured Data

* csv
* json
* parquet

## Knowledge Sources

* exported notes
* internal documents
* meeting notes
* technical documentation
* reports
* research papers

Future support:

* html
* xml
* websites
* youtube transcripts
* notion exports
* slack exports
* database connectors

---

# Data Ingestion Pipeline

## Step 1

Scan the `data/raw` folder.

---

## Step 2

Detect file type automatically.

---

## Step 3

Extract text using the corresponding parser.

Example parsers:

```text
pdf      → pypdf
csv      → pandas
docx     → python-docx
json     → recursive extraction
markdown → markdown parser
```

---

## Step 4

Generate metadata.

Required metadata:

```json
{
  "source_file": "",
  "source_type": "",
  "document_id": "",
  "chunk_id": "",
  "created_at": "",
  "title": "",
  "tags": []
}
```

---

## Step 5

Chunk documents.

Default configuration:

```text
chunk_size = 500
chunk_overlap = 100
```

Supported chunking strategies:

* Recursive chunking
* Semantic chunking
* Header aware chunking
* Parent child chunking

---

## Step 6

Generate embeddings locally.

Each chunk becomes:

```json
{
  "text": "",
  "embedding": [],
  "metadata": {}
}
```

---

## Step 7

Store embeddings inside ChromaDB.

The vector database must persist between sessions.

---

# Incremental Ingestion

The system must prevent duplicate ingestion.

Maintain an ingestion registry:

```json
{
  "filename": "",
  "sha256": "",
  "ingested_at": ""
}
```

Workflow:

* Calculate SHA256 hash.
* Compare with registry.
* Skip duplicates.
* Reprocess modified files only.

---

# Retrieval Pipeline

## Query Flow

```text
User Question
    ↓
Generate Query Embedding
    ↓
Retrieve Top K Chunks
    ↓
Optional Reranking
    ↓
Build Context Window
    ↓
Send Context to OpenRouter
    ↓
Generate Final Answer
```

---

## Retrieval Configuration

```text
top_k = 5
similarity_threshold = 0.75
```

---

## Reranking

Preferred model:

```text
BAAI/bge-reranker-base
```

Pipeline:

```text
Retrieve 20 Chunks
    ↓
Rerank Results
    ↓
Keep Best 5
    ↓
Generate Answer
```

---

# Prompt Template

```text
You are an assistant answering questions strictly using the provided context.

If the answer cannot be found in the context, reply:

"I could not find this information in the knowledge base."

Context:
{context}

Question:
{question}

Answer:
```

---

# Terminal Commands

## Ingestion

```bash
python ingest.py
```

Expected output:

```text
Loading files...
Extracting content...
Chunking documents...
Generating embeddings...
Saving vectors...
Completed successfully.
```

---

## Query Mode

```bash
python query.py
```

Example:

```text
Question:
What are the refund policies?

Retrieved Sources:
refund_policy.pdf
employee_handbook.md

Answer:
...
```

---

## Interactive Chat Mode

```bash
python main.py
```

Example:

```text
RAG Assistant Started

You:
What is the company leave policy?

Assistant:
...

You:
exit
```

---

# Conversation Memory

Maintain a rolling conversation history.

Configuration:

```text
conversation_window = 5
```

Future upgrades:

* persistent memory
* user profiles
* session management

---

# Source Attribution

Every answer must contain source references.

Example:

```text
Sources:

1. employee_handbook.pdf
2. refund_policy.md
3. onboarding_document.docx
```

---

# Configuration File

Example:

```yaml
openrouter:
  model: openai/gpt-5-mini
  temperature: 0.2
  max_tokens: 2000

embedding:
  model: BAAI/bge-small-en-v1.5

retrieval:
  top_k: 5
  similarity_threshold: 0.75

chunking:
  chunk_size: 500
  overlap: 100

vectordb:
  provider: chromadb
```

---

# Logging

Track:

* ingestion time
* embedding latency
* retrieval latency
* LLM latency
* token usage
* API cost
* failures

Store all logs inside:

```text
logs/
```

---

# Evaluation Metrics

Track the following metrics:

* Retrieval Precision
* Retrieval Recall
* Context Relevance
* Hallucination Rate
* Average Latency
* Cost Per Query

---

# Future Roadmap

Phase 2:

* FastAPI service layer
* REST API support

Phase 3:

* Authentication
* Multi user support

Phase 4:

* Hybrid Search
* BM25 + Vector Search

Phase 5:

* Agentic RAG
* Tool Calling
* Web Search Fallback

Phase 6:

* Multi Modal RAG
* Image Understanding
* Audio Understanding
* Video Understanding

---

# Deliverables Expected From OpenCode Agent

The generated repository should include:

* Complete project structure
* Dependency installation scripts
* OpenRouter integration
* Environment variable management
* ChromaDB integration
* Embedding generation pipeline
* Retrieval pipeline
* Terminal chat interface
* Logging system
* Unit tests
* Docker support
* Example dataset
* Makefile commands
* Comprehensive README

The final system should run with:

```bash
pip install -r requirements.txt

python ingest.py

python main.py
```
