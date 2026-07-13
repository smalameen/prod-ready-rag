# RAG Engineering — From Basics to Production

_A guide for beginners. If you understand this, you can build any RAG system._

---

## 1. What is RAG?

RAG stands for **Retrieval-Augmented Generation**.

Think of it like giving a student (the AI) a textbook before an exam. Without the textbook, the student guesses from memory. With the textbook, the student reads the relevant pages and gives a correct answer.

```
Without RAG:  "What is the refund policy?" → AI guesses (might be wrong)
With RAG:     "What is the refund policy?" → AI searches the knowledge base → reads the policy → answers correctly
```

**Three steps of RAG:**
1. **Ingest** — read documents, cut into pieces (chunks), store as numbers (embeddings)
2. **Retrieve** — when a question comes, find the most relevant pieces
3. **Generate** — send those pieces + question to an AI, get an answer

---

## 2. Core Concepts Used in This App

### 2.1 Chunking

**What:** Cutting a document into small pieces.

**Why:** AI models have a limit on how much text they can read at once (context window). You can't send a 100-page PDF, so you send only the relevant 2-3 chunks.

**In this app:** `CHUNK_SIZE=500`, `CHUNK_OVERLAP=100`

```
Document (1000 words)
  ├── Chunk 1 (words 1-100)
  ├── Chunk 2 (words 80-180)  ← overlap of 20 words
  └── Chunk 3 (words 160-260)
```

**Why overlap?** If a sentence is cut mid-way, the next chunk still has the context. Without overlap, you might lose meaning at boundaries.

**Strategies (from simple to advanced):**
| Strategy | How it works | When to use |
|----------|-------------|-------------|
| **Fixed size** | Cut every N characters | Simple text |
| **Recursive** | Cut at paragraphs first, then sentences, then words | General purpose — this app uses it |
| **Semantic** | Cut where meaning changes | Complex documents |
| **Header-aware** | Respect markdown/HTML headings | Structured docs |
| **Parent-child** | Store big chunks for context, small chunks for search | Production systems |

---

### 2.2 Embeddings

**What:** Converting text into a list of numbers (a vector).

**Why:** Computers can't compare "meaning" of words directly. But they can compare numbers. A vector is like a GPS coordinate for meaning.

```
"dog"    → [0.2, 0.8, -0.3, 0.1, ...]  (384 numbers)
"puppy"  → [0.21, 0.79, -0.28, 0.11, ...]  (very similar)
"table"  → [-0.5, 0.1, 0.7, -0.2, ...]  (very different)
```

**In this app:** Uses `BAAI/bge-small-en-v1.5` — a small, fast model that produces 384-dimensional vectors. Runs locally, no API needed.

**Key terms:**
- **Embedding dimension** — how many numbers in the vector (higher = more precision but slower). 384 is good for small apps, 1536 for production.
- **Normalization** — making all vectors the same length so only direction matters (cosine similarity).
- **Dense vs Sparse embeddings:**
  - Dense (this app): captures meaning, like synonyms
  - Sparse (BM25): captures exact word matches, like keywords

---

### 2.3 Vector Database (Vector Store)

**What:** A database specialized for storing and searching vectors.

**Why:** Normal databases search by exact match (`WHERE name = "John"`). Vector databases search by similarity (`WHERE embedding ≈ query_embedding`).

**In this app:** ChromaDB (persistent, local file-based).

**How search works:**
```
Query: "What is the leave policy?"
  → Convert to vector: [0.3, 0.7, ...]
  → Compare with ALL stored vectors
  → Find the 5 most similar
  → Return those chunks
```

**Similarity metrics:**
| Metric | What it measures | Range |
|--------|-----------------|-------|
| **Cosine similarity** | Angle between vectors (direction) | -1 to 1 (this app uses it) |
| **Euclidean distance** | Straight-line distance | 0 to ∞ |
| **Dot product** | Direction × magnitude | -∞ to ∞ |

**This app:** Uses cosine distance internally, converts to score: `score = 1 - distance`.

---

### 2.4 SIMILARITY_THRESHOLD

**What:** A cutoff score. If no chunk scores above this, return nothing.

**Why:** Prevents the AI from making up answers using irrelevant context. Better to say "I don't know" than give a wrong answer.

**Analogy:** In a exam, if none of the textbook pages look relevant to the question, don't try to answer. The threshold is the minimum confidence level.

**In this app:** `0.70` (meaning 70% similarity).

**Trade-off:**
- **High threshold (0.85)** — fewer wrong answers, but might miss relevant ones
- **Low threshold (0.50)** — finds more results, but might include garbage

**How to tune it:**
1. Run many test queries
2. Check the scores of results
3. Set threshold just below the lowest "good" score
4. Set threshold just above the highest "bad" score

---

### 2.5 TOP_K

**What:** How many chunks to retrieve.

**Why:** Sending too many chunks wastes AI tokens (money). Sending too few might miss the answer.

**In this app:** `5` (retrieve 20, rerank to keep 5).

**Analogy:** You search Google. You look at the top 5 results, not all 10 million.

---

### 2.6 Reranking

**What:** A second, more accurate search on the initial results.

**Why:** Embeddings are fast but approximate. Reranking is slower but more accurate. Use both: fast to get candidates, accurate to pick the best.

**Flow:**
```
Retrieve 20 chunks (fast, ~10ms)
  → Rerank (slow, ~500ms)
  → Keep top 5 (best quality)
```

**In this app:** Optional `BAAI/bge-reranker-base` model. Falls back to score-based sorting if model isn't available.

**Embedding vs Reranker:**
| Step | Model | Speed | Accuracy | Purpose |
|------|-------|-------|----------|---------|
| Search | bge-small | Fast (10ms) | Good | Find candidates |
| Rerank | bge-reranker | Slow (500ms) | Excellent | Pick best |

---

### 2.7 Retrieval Pipeline

**What:** The complete chain: question → embedding → search → rerank → context.

```
Question: "What is the refund policy?"
    ↓
Embed query → [0.3, 0.7, ...]
    ↓
Vector search → 20 chunks with scores
    ↓
Rerank → 5 best chunks
    ↓
Build context window
    ↓
Send to LLM + question
    ↓
Generate answer
```

---

### 2.8 LLM (Large Language Model)

**What:** The AI that reads context and answers questions.

**In this app:** Uses OpenRouter as a gateway to many models:
- `openai/gpt-5-mini` (default)
- `anthropic/claude-sonnet-4`
- `google/gemini-2.5-pro`

**Without OpenRouter:** You'd need separate API keys for OpenAI, Anthropic, Google, etc. OpenRouter gives one API key for all models.

**Key parameters:**
| Parameter | What it does | This app |
|-----------|-------------|----------|
| **temperature** | Creativity (0 = factual, 1 = creative) | 0.2 |
| **max_tokens** | Max length of answer | 2000 |
| **model** | Which AI to use | gpt-5-mini |

---

### 2.9 Prompt Template

**What:** The instructions given to the AI before every question.

**Why:** Without instructions, the AI might answer from its own knowledge (hallucination) instead of the provided context.

**This app's prompt:**
```
You are an assistant answering questions strictly using the provided context.

If the answer cannot be found in the context, reply:
"I could not find this information in the knowledge base."

Context:
{context}

Question:
{question}

Answer:
```

**Why this matters:** The prompt is the most important part. A bad prompt = bad answers even with perfect retrieval.

---

### 2.10 Context Window

**What:** The combined text from retrieved chunks that gets sent to the AI.

**Why:** AI models can only read a limited amount of text at once. This is measured in **tokens** (roughly ¾ of a word).

| Model | Context window |
|-------|---------------|
| GPT-3.5 | 4,096 tokens (~3,000 words) |
| GPT-4 | 8,192 tokens (~6,000 words) |
| GPT-5-mini | 128,000 tokens (~96,000 words) |
| Claude-4 | 200,000 tokens |

**Challenge:** If you retrieve too many chunks, you fill the context window and the AI can't see the question. You need to fit: context + question + instructions ≤ model limit.

---

### 2.11 Incremental Ingestion

**What:** Only processing new or changed files, skipping unchanged ones.

**Why:** Re-processing 10,000 documents every time you add one file is wasteful.

**How it works:**
```
File added → Compute SHA256 hash (a fingerprint)
  → Compare with registry (JSON file)
  → If hash matches → skip (already ingested)
  → If hash differs → re-process
  → If not in registry → process as new
```

**In this app:** Registry stored at `data/processed/registry.json`.

---

### 2.12 Source Attribution

**What:** Every answer includes which documents were used.

**Why:** Trust. If you can't verify where the answer came from, you can't trust it.

**Example output:**
```
Answer: Employees get 20 days of annual leave.
Sources:
1. employee_handbook.txt
2. company_policy.pdf
```

---

### 2.13 Conversation Memory

**What:** Remembering previous questions and answers.

**Why:** A conversation isn't one question — it's a back-and-forth. Without memory, each question is isolated.

**In this app:** Rolling window of last 5 exchanges. Old messages get discarded to save tokens.

```
You: What is the leave policy?
AI: 20 days per year.

You: Can I carry unused days?     ← Without memory, AI doesn't know "it" = leave
AI: Yes, up to 5 days.            ← With memory, AI knows we're talking about leave
```

---

### 2.14 Metadata Filtering

**What:** Searching only within specific documents or categories.

**Why:** If you ask "What did Bob say in the June meeting?", you want to search only `meeting_notes.json`, not all documents.

**In this app:** Basic metadata stored with each chunk: `source_file`, `source_type`, `document_id`, etc.

**Example usage (future):**
```python
results = vector_store.search(
    query="What was the budget?",
    filter={"source_file": "meeting_notes.json"}
)
```

---

### 2.15 Latency & Token Tracking

**What:** Measuring how long each step takes and how many tokens the AI uses.

**Why:** If the system is slow or expensive, you need to know which part to optimize.

**This app logs:**
```
Latency [embed_query]: 0.0234s
Latency [vector_search]: 0.0156s
Latency [llm_generation]: 3.2456s
LLM Call | Model=gpt-5-mini | Prompt=523 | Completion=187 | Cost=$0.000106
```

**Where time goes in RAG:**
| Step | Typical time | Optimization |
|------|-------------|-------------|
| Embedding query | 10-50ms | Use a smaller model |
| Vector search | 5-50ms | Use ANN (approximate) instead of brute force |
| Reranking | 200-1000ms | Skip if not needed, use a smaller model |
| LLM generation | 1-10s | Use a faster/cheaper model |
| Total | 1.5-12s | — |

---

### 2.16 Cost Estimation

**What:** Tracking how much each query costs.

**Why:** AI models charge per token. If you have 1,000 users asking 10 questions/day, costs add up.

**This app's formula:**
```python
cost = (prompt_tokens × 1.0 + completion_tokens × 2.0) / 1_000_000 × 0.15
```

**Real costs (approximate):**
| Model | Cost per 1M input tokens | Cost per 1M output tokens |
|-------|-------------------------|--------------------------|
| GPT-5-mini | $0.15 | $0.60 |
| GPT-5 | $1.00 | $2.00 |
| Claude-4 | $3.00 | $15.00 |

**At 100 queries/day:** GPT-5-mini ≈ $0.01/day. Claude-4 ≈ $0.50/day.

---

## 3. Concepts to Learn for Production RAG

### 3.1 Hybrid Search

**Problem:** Pure vector search is bad at exact keyword matching. If a document says "The code is ABC-123-XYZ" and you search for "ABC-123-XYZ", vector search might not find it because it's just a code, not a concept.

**Solution:** Combine vector search (meaning) with BM25 (keyword).

```
Score = 0.5 × vector_similarity + 0.5 × bm25_score
```

**BM25** is an algorithm that scores documents based on exact word matches, like Google's original algorithm.

### 3.2 Evaluation (RAGAS)

**Problem:** How do you know if your RAG system is good?

**Solution:** Use a framework like RAGAS that measures:
- **Faithfulness** — does the answer match the context? (not hallucinating)
- **Answer relevancy** — does the answer actually answer the question?
- **Context precision** — are the retrieved chunks all relevant?
- **Context recall** — are all needed chunks retrieved?

### 3.3 Advanced Retrieval Techniques

| Technique | What it does | Why |
|-----------|-------------|-----|
| **Multi-query** | Generate 5 versions of the question, search all | Handles phrasing differences |
| **HyDE** | Generate a hypothetical answer, use IT for search | Better matching for complex questions |
| **Parent retriever** | Retrieve small chunks, return their parents | More context for the AI |
| **Self-query** | AI generates metadata filters from question | Automatic filtering |
| **Query transformation** | Rewrite the question for better search | Handles vague or complex questions |

### 3.4 Chunking Optimization

**Rule of thumb:** There's no single best chunk size. It depends on your data.

| Content type | Chunk size | Strategy |
|-------------|-----------|----------|
| Tweets/emails | 100-200 | Fixed |
| News articles | 500-1000 | Recursive |
| Academic papers | 1000-2000 | Semantic |
| Code | Function-level | Language-specific |
| Books | 2000-5000 | Header-aware |

### 3.5 Embedding Model Selection

| Model | Dimensions | Size | Speed | Quality |
|-------|-----------|------|-------|---------|
| `bge-small-en-v1.5` | 384 | 33MB | Fast | Good |
| `all-MiniLM-L6-v2` | 384 | 80MB | Fast | Good |
| `bge-base-en-v1.5` | 768 | 130MB | Medium | Better |
| `bge-large-en-v1.5` | 1024 | 330MB | Slow | Best |
| `text-embedding-3-small` | 1536 | API | Fast | Excellent |

### 3.6 Vector Database Comparison

| Database | Type | Free tier | Best for |
|----------|------|-----------|----------|
| **ChromaDB** | Embedded | ✅ Free | Prototyping, local apps |
| **FAISS** | Library | ✅ Free | High performance, research |
| **Pinecone** | Cloud | ❌ Paid | Production, managed |
| **Weaviate** | Cloud/Self-host | ✅ Free tier | Production, Kubernetes |
| **Qdrant** | Cloud/Self-host | ✅ Free tier | Production, Rust-based |
| **Milvus** | Cloud/Self-host | ✅ Free tier | Billion-scale |

### 3.7 Streaming

**What:** Showing the AI's answer word-by-word as it's generated, instead of waiting for the complete answer.

**Why:** Users perceive a 100ms response as instant, but a 5-second wait as slow. Streaming makes it feel fast even if it's not.

### 3.8 Guardrails

**What:** Checking the AI's output before showing it to the user.

**Why:** AI can still produce harmful, incorrect, or off-topic content even with good context.

**Examples:**
- Block profanity
- Ensure the answer cites sources
- Reject questions outside the knowledge domain
- Check for personally identifiable information (PII)

### 3.9 Caching

**What:** Store answers to common questions so you don't call the AI every time.

**Analogy:** If 100 people ask "What is the refund policy?" today, don't call the AI 100 times. Answer from cache 99 times.

**Two levels:**
1. **Exact match** — same question word-for-word
2. **Semantic cache** — similar question, same answer

### 3.10 Monitoring & Observability

**Production must-haves:**
- Dashboard showing queries, latency, cost, errors
- Logging every query + answer (for debugging)
- Alerting when error rate spikes or latency increases
- A/B testing different retrieval strategies

### 3.11 User Feedback

**Closed-loop improvement:**
```
User asks question → System answers → User rates (👍/👎)
    → Collect data → Find patterns → Improve retrieval/chunking
```

Without feedback, you don't know if your system is getting better or worse.

---

## 4. Architecture Evolution

### Stage 1: What this app is (Terminal RAG)
```
Text files → Local embeddings → ChromaDB → OpenRouter → Terminal
```

### Stage 2: API + UI (Next step)
```
UI (Streamlit/FastAPI) → API → Same pipeline → Database
```

### Stage 3: Production RAG
```
Multiple document sources
  → Async ingestion queue
  → Hybrid search (vector + BM25)
  → Reranking
  → Query transformation
  → Guardrails
  → Streaming
  → Cache
  → Monitoring
  → User feedback loop
```

---

## 5. Glossary

| Term | Simple definition |
|------|------------------|
| **Token** | ~¾ of a word. AI measures everything in tokens |
| **Embedding** | Converting text to numbers (like GPS coordinates for meaning) |
| **Vector** | A list of numbers representing meaning |
| **Cosine similarity** | Measuring how similar two vectors are by their angle |
| **Chunk** | A piece of a document (usually 200-1000 words) |
| **Context window** | How much text an AI can read at once |
| **Hallucination** | AI making up false information |
| **Latency** | Time taken for a response |
| **Throughput** | How many queries per second the system can handle |
| **ANN** | Approximate Nearest Neighbor — fast but approximate vector search |
| **KNN** | K-Nearest Neighbors — exact but slow vector search |
| **BM25** | Keyword-based search algorithm (like old Google) |
| **Hybrid search** | Vector search + keyword search combined |
| **Reranking** | A second, more accurate search on top results |
| **RAGAS** | Framework for measuring RAG quality |
| **Fine-tuning** | Training an AI model on your specific data |
| **Agentic RAG** | AI that can call tools, search multiple times, reason step-by-step |
| **Multi-modal RAG** | RAG with images, audio, video in addition to text |

---

## 6. Where to Go Next

**Learning path:**
1. ✅ Understand this app's code (terminal RAG)
2. ➡️ Add a Streamlit UI (web interface)
3. ➡️ Implement hybrid search (vector + BM25)
4. ➡️ Set up RAGAS evaluation
5. ➡️ Add query transformation (multi-query, HyDE)
6. ➡️ Move to a production vector DB (Qdrant/Weaviate)
7. ➡️ Add monitoring + caching
8. ➡️ Scale to millions of documents

---

_If you understand everything in this document, you know more about RAG than most developers. The rest is just practice with larger datasets._
