# RAG That Won't Lie

A retrieval-augmented generation (RAG) pipeline over the full English Wikipedia,
designed for **grounded, citation-backed answers and near-zero hallucination**. It
runs entirely on local hardware: no hosted vector database, no external LLM API.

The defining property is restraint. When the retrieved evidence does not support an
answer, the system returns *"insufficient evidence"* rather than generating a
plausible-sounding guess.

```
Q: Who developed the theory of relativity?
A: Albert Einstein developed the theory of relativity. [1]
   [1] Theory of relativity — en.wikipedia.org

Q: What did Albert Einstein eat for breakfast?
A: I don't have enough reliable information to answer that.
```

Einstein is well represented in the corpus, so retrieval succeeds — but no passage
states what he ate, so the system declines to answer.

---

## Architecture

A ten-stage pipeline. Stages 1–4 build the searchable corpus offline; stages 5–10
serve and govern queries online.

```
Wikipedia dump (25GB)
   │
   ▼
[1] Ingest          parse + clean wikitext (wikiextractor)
[2] Chunk           sentence-aware split, 384 tokens, 50 overlap   → 13.98M chunks
[3] Embed           bge-small-en-v1.5 → 384-dim vectors (float16)
[4] Index           FAISS (semantic) + BM25 (lexical) + SQLite (metadata)
   │
   ▼  per query
[5] Retrieve        hybrid search → Reciprocal Rank Fusion → cross-encoder rerank
[6] Score           confidence from rerank relevance + source agreement
[7] Gate            below threshold → refuse; otherwise → generate
[8] Generate        local Qwen2.5, context-only prompt, inline citations
   │
   ├─ Evals (9)        curated suite, refusal-rate as the anti-hallucination metric
   ├─ Caching (10a)    normalized-hash query cache (SQLite)
   └─ Observability    per-query JSONL trace of the full retrieval path
```

Each stage maps to one script in [`pipeline/`](pipeline/) and one document in
[`docs/`](docs/).

---

## Anti-hallucination design

Two independent guards:

1. **Confidence gate (pre-generation).** After reranking, if the top candidate
   scores below a threshold, the query is refused before the LLM is ever called.
   Catches off-topic and adversarial inputs.
2. **Constrained generation (in-prompt).** The model is instructed to answer only
   from the supplied passages and to return "insufficient evidence" otherwise.
   Catches on-topic questions the corpus cannot actually answer.

Every answer carries citations back to the source passages.

---

## Results (measured)

Continuous evaluation (`pipeline/evaluate.py`) over a curated set in three buckets —
answerable from corpus, on-topic but unanswerable, and adversarial:

```
answerable:  answer rate     100.0%
answerable:  citation rate   100.0%
refusal rate (unanswerable + adversarial)   100.0%
overall behaved as expected 100.0%
```

| Metric | Value |
|---|---|
| Source corpus | English Wikipedia, ~25GB compressed |
| Usable chunks | 13.98M |
| Chunks embedded (this build) | 750,000 |
| Embedding dimension | 384 (float16) |
| Vector index | FAISS `IndexFlatIP` |
| Cache hit latency | ~1 ms (vs ~10 s to generate) |
| Query latency | ~0.4–0.8 s retrieval, ~10–15 s generation (CPU) |

The current build embeds the first 750K chunks. The pipeline scales to the full
14M without changes; the remainder is compute time.

---

## Setup

```bash
pip install -r requirements.txt
```

Download the Wikipedia dump (~25GB):

```bash
curl -C - -L -o enwiki-latest-pages-articles-multistream.xml.bz2 \
  https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream.xml.bz2
```

## Usage

```bash
# 1. Parse → clean text
wikiextractor --json --processes 10 -o wiki_extracted/ \
  enwiki-latest-pages-articles-multistream.xml.bz2

# 2. Chunk
python3 pipeline/chunk.py

# 3. Embed
python3 pipeline/embed.py

# 4. Build indexes (FAISS + BM25 + SQLite)
python3 pipeline/index.py --all

# 5. Query
LLM_DEVICE=cpu RETRIEVER_DEVICE=cpu \
  python3 pipeline/generate.py "Who developed the theory of relativity?"

# Continuous evaluation
python3 pipeline/evaluate.py

# Inspect a query's full retrieval trace
python3 pipeline/observability.py --tail 5
```

The dump, extracted text, chunks, embeddings, and indexes are gitignored and
regenerated locally.

---

## Implementation notes

- **Hybrid retrieval** combines BM25 (exact terms) and dense vectors (semantics)
  via Reciprocal Rank Fusion, then reranks the top candidates with a cross-encoder.
- **Local inference.** Embeddings use `bge-small-en-v1.5`; generation uses Qwen2.5.
  Generation runs on CPU in float32 for stability (`float16` is unsupported on CPU).
- **Threading.** `faiss` and `torch` each bundle an OpenMP runtime; loaded together
  they can collide on a thread barrier and crash. The scripts set
  `OMP_NUM_THREADS=1` and `KMP_DUPLICATE_LIB_OK=TRUE` to avoid this.

---

## Project structure

```
RAG-that-wont-lie/
├── README.md
├── requirements.txt
├── pipeline/
│   ├── parse.py            # ingest: stream + clean the XML dump
│   ├── chunk.py            # sentence-aware chunking
│   ├── embed.py            # embeddings (bge-small-en-v1.5)
│   ├── embed_test.py       # single-batch smoke test
│   ├── index.py            # FAISS + BM25 + SQLite
│   ├── retrieve.py         # hybrid retrieval, RRF, rerank, confidence
│   ├── generate.py         # confidence gate, constrained generation, citations
│   ├── cache.py            # query cache
│   ├── observability.py    # per-query trace logging + viewer
│   ├── evaluate.py         # continuous evals
│   └── test_e2e.py         # end-to-end check
└── docs/                   # per-stage documentation (1–6)
```

---

## Stack

Python · wikiextractor · `bge-small-en-v1.5` · FAISS · bm25s ·
`ms-marco-MiniLM-L-6-v2` (reranker) · Qwen2.5 · SQLite. All open source, all local.
