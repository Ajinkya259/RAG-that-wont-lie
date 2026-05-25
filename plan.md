# RAG Pipeline — Detailed Build Plan

**Dataset:** English Wikipedia XML dump (`enwiki-latest-pages-articles-multistream.xml.bz2`, ~25GB)  
**Goal:** 10M+ document RAG system with near-zero hallucination  
**Stack:** Python, FAISS, BM25s, sentence-transformers, LangChain (optional)

---

## Phase 1 — Ingestion & Parsing

### Step 1.1 — Parse Wikipedia XML
- Use `mwxml` or `mwparserfromhell` to stream the bz2 XML without loading it all into memory
- Extract per article:
  - `title` (string)
  - `raw_wikitext` (string)
  - `last_revised` (ISO timestamp)
  - `article_id` (int)
  - `url` → construct as `https://en.wikipedia.org/wiki/{title}`
- Skip redirect articles (`#REDIRECT` in text)
- Skip disambiguation pages (title contains "(disambiguation)")
- Skip stub articles (< 200 characters after cleaning)
- Output: JSONL file — one article per line

### Step 1.2 — Clean Wikitext
- Use `mwparserfromhell` to strip:
  - Templates (`{{ }}`)
  - Infoboxes
  - Tables (`{| ... |}`)
  - File/Image links
  - External links (keep anchor text)
  - HTML tags
  - Citation markers (`[1]`, `[2]`)
- Keep: plain text, section headings
- Output: clean `text` field per article

### Step 1.3 — Deduplication
- Hash each article's cleaned text using SHA-256
- Drop duplicates by hash
- Expected output: ~6.5–6.7M unique articles

**Deliverable:** `data/articles.jsonl`  
**Schema:** `{ "id": int, "title": str, "text": str, "url": str, "timestamp": str }`

---

## Phase 2 — Chunking

### Step 2.1 — Sliding Window Chunking
- Tokenize using `tiktoken` (cl100k_base tokenizer)
- Chunk size: **512 tokens**
- Overlap: **64 tokens** (to preserve context across chunk boundaries)
- Minimum chunk size: 100 tokens (drop smaller chunks)
- Each chunk inherits metadata from parent article

### Step 2.2 — Metadata Attachment
Each chunk gets:
- `chunk_id` — unique ID: `{article_id}_{chunk_index}`
- `article_title` — for citation
- `article_url` — for citation
- `timestamp` — article last revised date (for freshness scoring)
- `chunk_index` — position within article
- `total_chunks` — total chunks in article
- `text` — chunk text

**Deliverable:** `data/chunks.jsonl`  
**Estimated volume:** ~30–35M chunks from 6.5M articles  
**Schema:** `{ "chunk_id": str, "article_title": str, "article_url": str, "timestamp": str, "chunk_index": int, "total_chunks": int, "text": str }`

---

## Phase 3 — Indexing

### Step 3.1 — Embedding
- Model: `BAAI/bge-base-en-v1.5` (768-dim, strong retrieval performance, open source)
- Batch size: 256 (tune based on available RAM/GPU)
- Run on CPU (slow but works) or GPU if available
- Save embeddings as numpy `.npy` file or directly into FAISS index
- Estimated time: ~24–48 hrs on CPU for 30M chunks — consider sampling 10M chunks first

### Step 3.2 — FAISS Vector Index
- Index type: `IndexIVFFlat` (ANN — Approximate Nearest Neighbor)
  - `nlist` = 4096 (number of Voronoi cells)
  - Train on 1M random sample first, then add all vectors
- Save index to disk: `indexes/faiss.index`
- Also save chunk_id → index mapping: `indexes/id_map.json`

### Step 3.3 — BM25 Index
- Library: `bm25s` (fast, no Java dependency unlike Elasticsearch)
- Build from `chunks.jsonl` text field
- Save index to disk: `indexes/bm25.index`

**Deliverables:**
- `indexes/faiss.index`
- `indexes/bm25.index`
- `indexes/id_map.json`
- `indexes/chunk_metadata.db` (SQLite for fast chunk_id → metadata lookup)

---

## Phase 4 — Retrieval

### Step 4.1 — Hybrid Retrieval
- For a given query:
  1. Run BM25 → top-100 results (with scores)
  2. Run FAISS ANN → top-100 results (with cosine similarity scores)
  3. Merge using **Reciprocal Rank Fusion (RRF)**:
     - `RRF_score(d) = Σ 1 / (k + rank(d))` where k=60
  4. Return top-20 merged candidates

### Step 4.2 — Re-ranking
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (fast cross-encoder)
- Input: (query, chunk_text) pairs for top-20 candidates
- Output: re-ranked top-5 chunks with relevance scores
- This is the "deep scoring" step — more expensive but more accurate

### Step 4.3 — Confidence Scoring
Each retrieved chunk gets a composite confidence score:

```
confidence = w1 * relevance_score        # from re-ranker (0–1)
           + w2 * freshness_score        # based on timestamp recency (0–1)
           + w3 * retrieval_consistency  # did BM25 and vector agree? (0–1)
```

- `w1=0.6, w2=0.2, w3=0.2` (tune based on eval)
- Freshness: articles revised in last 2 years score 1.0, older decay linearly
- Retrieval consistency: chunk appears in both BM25 and FAISS results → score 1.0, only one → 0.5

**Threshold:** confidence < 0.4 → trigger hallucination fallback

---

## Phase 5 — Generation

### Step 5.1 — Prompt Construction
Build a strict constrained prompt:

```
You are a factual assistant. Answer ONLY using the provided context.
If the answer is not in the context, respond with: "Insufficient evidence."

Context:
[chunk_1_text] (Source: {title}, {timestamp})
[chunk_2_text] (Source: {title}, {timestamp})
...

Question: {query}
Answer:
```

### Step 5.2 — LLM Generation
- Model: Claude claude-sonnet-4-6 via Anthropic API (or local `llama3` via Ollama for offline)
- Temperature: 0.0 (deterministic, reduces hallucination)
- Max tokens: 512
- System prompt enforces no outside knowledge

### Step 5.3 — Hallucination Fallback
- If **average confidence of top chunks < 0.4** → skip LLM, return:
  ```
  { "answer": "Insufficient evidence in the knowledge base.", "sources": [] }
  ```
- If LLM generates an answer → attach citations:
  ```
  { "answer": "...", "sources": [{ "title": "...", "url": "...", "timestamp": "..." }] }
  ```

---

## Phase 6 — Evaluation

### Step 6.1 — Benchmark Datasets
- **Natural Questions (NQ)** — 3,610 test questions, Wikipedia-grounded
- **TriviaQA** — 11,313 test questions
- Metric: **Exact Match (EM)** and **F1 score**
- Hallucination rate: % of answers containing content not in retrieved chunks

### Step 6.2 — Adversarial Testing
- Inject questions with no answer in corpus → expect "Insufficient evidence"
- Inject ambiguous questions → test confidence scoring behaviour
- Inject questions about very recent events (post-dump) → test fallback

### Step 6.3 — Continuous Eval Script
- `eval/run_eval.py` — runs benchmark, logs per-query results
- Output: `eval/results.jsonl` with query, retrieved chunks, answer, EM, F1, confidence

---

## Phase 7 — Observability

### Step 7.1 — Trace Logging
Every query logs:
```json
{
  "query_id": "uuid",
  "query": "...",
  "bm25_top10": ["chunk_id", ...],
  "faiss_top10": ["chunk_id", ...],
  "reranked_top5": ["chunk_id", ...],
  "confidence_scores": [0.82, 0.74, ...],
  "retrieval_path": "bm25+faiss→rrf→rerank",
  "token_attribution": { "chunk_id": "sentence span" },
  "final_answer": "...",
  "hallucination_fallback_triggered": false,
  "latency_ms": 340
}
```
- Store in SQLite: `logs/traces.db`

### Step 7.2 — Caching
- Cache query embeddings (exact match): `cache/query_embeddings.db`
- Cache top-k retrieval results for repeated queries: `cache/retrieval_cache.db`
- TTL: 24 hours

---

## Directory Structure

```
RAG/
├── data/
│   ├── articles.jsonl          # parsed articles
│   └── chunks.jsonl            # chunked with metadata
├── indexes/
│   ├── faiss.index
│   ├── bm25.index
│   ├── id_map.json
│   └── chunk_metadata.db
├── eval/
│   ├── run_eval.py
│   └── results.jsonl
├── logs/
│   └── traces.db
├── cache/
│   ├── query_embeddings.db
│   └── retrieval_cache.db
├── pipeline/
│   ├── parse.py                # Phase 1
│   ├── chunk.py                # Phase 2
│   ├── index.py                # Phase 3
│   ├── retrieve.py             # Phase 4
│   ├── generate.py             # Phase 5
│   └── evaluate.py             # Phase 6
├── enwiki-latest-pages-articles-multistream.xml.bz2
├── reel_DYpSunLBQLE_transcript.txt
└── plan.md
```

---

## Build Order

| Phase | Script | Input | Output | Est. Time |
|---|---|---|---|---|
| 1 | `parse.py` | XML bz2 | `articles.jsonl` | 2–3 hrs |
| 2 | `chunk.py` | `articles.jsonl` | `chunks.jsonl` | 1–2 hrs |
| 3a | `index.py --bm25` | `chunks.jsonl` | `bm25.index` | 1 hr |
| 3b | `index.py --faiss` | `chunks.jsonl` | `faiss.index` | 24–48 hrs CPU / 4–6 hrs GPU |
| 4–5 | `retrieve.py` + `generate.py` | query | answer + sources | real-time |
| 6 | `evaluate.py` | NQ / TriviaQA | `results.jsonl` | 2–4 hrs |

---

## Key Libraries

```
mwxml                  # Wikipedia XML streaming parser
mwparserfromhell       # wikitext cleaner
tiktoken               # tokenizer for chunking
sentence-transformers  # embedding model
faiss-cpu              # vector index
bm25s                  # BM25 keyword index
anthropic              # LLM generation
datasets               # load NQ / TriviaQA for eval
sqlitedict             # metadata + cache storage
```
