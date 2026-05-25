# Step 5 — Retrieval Plan (Detailed)

## TL;DR (Plain English First)

We have three indexes built. Step 5 is where they finally get *used*. Given a
question, we:

1. Turn the question into a fingerprint (same as Step 3)
2. Search the meaning-index (FAISS) → top 100 chunks
3. Search the keyword-index (BM25) → top 100 chunks
4. Merge the two lists smartly (RRF) — chunks both agree on rank higher
5. Re-judge the top merged chunks with a smarter (slower) model — "reranking"
6. Score our confidence in each result
7. Return the best chunks with their text, title, and URL

**Step 5 = take a question, find the most relevant chunks.**

No LLM yet — this step just *finds* the right chunks. Writing the answer is Step 6.

---

## What We Have to Work With

| Asset | What it gives us |
|---|---|
| `indexes/faiss.index` | meaning search (returns row numbers) |
| `indexes/faiss_ids.json` | row number → chunk_id |
| `indexes/bm25/` | keyword search (returns doc indices) |
| `indexes/bm25_ids.json` | doc index → chunk_id |
| `indexes/chunk_metadata.db` | chunk_id → text/title/url/timestamp |
| `BAAI/bge-small-en-v1.5` | the embedding model (for the query) |

All keyed on the same 750K chunk_ids.

---

## The Retrieval Pipeline (Step by Step)

### 5.1 — Embed the Query

```python
query = "Who developed the theory of relativity?"
prefixed = "Represent this sentence for searching relevant passages: " + query
q_vec = model.encode([prefixed], normalize_embeddings=True)  # shape (1, 384)
```

**Critical:** BGE models need the instruction prefix on the QUERY (but NOT on the
corpus chunks — which is why Step 3 had no prefix). Skipping this measurably hurts
retrieval quality.

### 5.2 — FAISS Search (Meaning)

```python
D, I = faiss_index.search(q_vec.astype('float32'), 100)
# I = row numbers, D = inner-product scores (= cosine sim, 0..1)
faiss_hits = [(faiss_ids[row], score) for row, score in zip(I[0], D[0])]
```
→ top-100 `(chunk_id, similarity)` pairs

### 5.3 — BM25 Search (Keywords)

```python
q_tokens = bm25s.tokenize([query])          # tokenize the raw query (no prefix)
results, scores = bm25_retriever.retrieve(q_tokens, k=100)
bm25_hits = [(bm25_ids[doc_idx], score) for doc_idx, score in zip(results[0], scores[0])]
```
→ top-100 `(chunk_id, bm25_score)` pairs

### 5.4 — Reciprocal Rank Fusion (Merge)

The two lists use different score scales (cosine 0–1 vs BM25 unbounded), so we
merge by **rank**, not raw score:

```python
def rrf(faiss_hits, bm25_hits, k=60):
    scores = {}
    for rank, (cid, _) in enumerate(faiss_hits, 1):
        scores[cid] = scores.get(cid, 0) + 1/(k + rank)
    for rank, (cid, _) in enumerate(bm25_hits, 1):
        scores[cid] = scores.get(cid, 0) + 1/(k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])
```

- `k=60` is the standard RRF constant (dampens top-rank dominance)
- A chunk appearing in BOTH lists gets two contributions → ranks higher
- We keep the **top 50** merged candidates for reranking
- We also track which retriever(s) found each chunk → used in confidence scoring

### 5.5 — Cross-Encoder Reranking

FAISS/BM25 are fast but approximate. A cross-encoder reads (query, chunk) together
and scores true relevance — much more accurate, but slow, so we only run it on the
top 50 candidates.

```python
from sentence_transformers import CrossEncoder
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='mps')
pairs = [(query, metadata[cid]['text']) for cid in top50_ids]
rel_scores = reranker.predict(pairs)   # higher = more relevant
```

Then sort by `rel_scores`, keep **top 5**.

**Model note:** Using `ms-marco-MiniLM-L-6-v2` (smaller/faster) rather than the
L-12 from the original plan — keeps reranking snappy on MPS for a demo, and L-6
is still strong. Reranking 50 pairs runs in well under a second.

### 5.6 — Confidence Scoring

For each of the final top-5 chunks:

```python
confidence = (
    0.6 * normalized_rerank_score   # cross-encoder score, sigmoid-normalized to 0..1
  + 0.2 * retrieval_consistency     # 1.0 if in BOTH faiss+bm25, else 0.5
  + 0.2 * freshness_score           # neutral 0.5 (our chunks have no timestamp)
)
```

- Rerank scores can be negative/large → squash with sigmoid to 0–1 first
- Freshness is neutral 0.5 for now (wikiextractor didn't give us timestamps)
- **Threshold: if the top chunk's confidence < 0.4 → flag as low-confidence**
  (Step 6 will use this to trigger the "insufficient evidence" fallback)

### 5.7 — Return Structure

```python
{
  "query": "...",
  "results": [
    {
      "chunk_id": "12_0003",
      "title": "Anarchism",
      "url": "https://en.wikipedia.org/wiki?curid=12",
      "text": "...",
      "confidence": 0.82,
      "rerank_score": 7.4,
      "found_by": ["faiss", "bm25"]
    },
    ... up to 5 ...
  ],
  "low_confidence": false,
  "latency_ms": 340
}
```

---

## Script: `pipeline/retrieve.py`

### Design: A Reusable `Retriever` Class
Load everything ONCE (indexes + models stay in memory), then answer many queries.
This matters — loading FAISS (1.1GB) + BM25 (656MB) + model takes ~10–20s, but
should only happen once, not per query.

```python
class Retriever:
    def __init__(self):
        # load faiss, faiss_ids, bm25, bm25_ids, sqlite, embed model, reranker
    def search(self, query, top_k=5) -> dict:
        # 5.1 → 5.7 pipeline above
```

### CLI for Testing
```bash
python3 pipeline/retrieve.py "Who developed the theory of relativity?"
```
Prints the top-5 chunks with titles, confidence, and a text snippet.

---

## Memory & Speed Budget

| Component | RAM | Per-query time |
|---|---|---|
| FAISS index | ~1.1GB | ~10ms |
| BM25 index | ~1GB | ~50ms |
| SQLite | minimal (disk) | ~5ms |
| Embed model | ~150MB | ~30ms (one short query) |
| Reranker | ~90MB | ~300ms (50 pairs) |
| **Total** | **~3.3GB** | **~400ms/query** |

One-time load: ~15–20s. Then every query is sub-second.

---

## Safety: No Step-3-Style Crashes

- Models load once and stay resident (no repeated heavy GPU loads)
- Each query is tiny (one short text to embed, 50 pairs to rerank)
- No long sustained GPU runs → no sleep/SIGKILL risk
- Reranker on MPS is fine; can fall back to CPU if MPS misbehaves on short loads

---

## Verification Plan

Test with queries that probe both search types:

| Query | Tests |
|---|---|
| "Who developed the theory of relativity?" | meaning + keyword (Einstein) |
| "photosynthesis process in plants" | meaning (finds biology chunks) |
| "What is the capital of France?" | exact-fact retrieval |
| "asdfghjkl qwerty nonsense" | low-confidence path (should flag) |

For each: confirm top results are topically correct, confidence is sensible, and
`found_by` shows hybrid behavior (some chunks found by both retrievers).

Also confirm: a known chunk's text returned by the reranker matches the SQLite
metadata (no ID misalignment between indexes).

---

## Dependencies (Already Installed)

```
faiss-cpu, bm25s, sentence-transformers, numpy
# sqlite3 built-in
# cross-encoder model downloads on first run (~80MB)
```

---

## What Step 5 Does NOT Do

- Does NOT write a natural-language answer (Step 6 — the LLM)
- Does NOT do the "insufficient evidence" fallback message (Step 6 — it just
  *flags* low confidence here)
- Does NOT log traces or cache (Step 7/8)
- Just: question in → ranked relevant chunks out.
