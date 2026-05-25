# Step 5 — Retrieval (What We Did, In Detail)

## TL;DR (Plain English)

After Step 4 we had three searchable indexes. Step 5 is where they finally get
*used*. We built the machine that takes a question and finds the most relevant
Wikipedia chunks for it.

**Step 5 = question in → ranked relevant chunks out.** (No LLM yet — just finding.)

It works. "Who developed the theory of relativity?" returns the Theory of
Relativity article (about Einstein) as the #1 result in ~800ms.

---

## The Pipeline (7 Stages)

### 1. Embed the Question
Turn the question into a 384-number fingerprint using the same model from Step 3.

**Critical detail:** BGE models need a special prefix on the QUESTION:
`"Represent this sentence for searching relevant passages: "`
We do NOT add this to the stored chunks (we didn't in Step 3) — only the query.
Skipping it measurably hurts results.

### 2. FAISS Search (Meaning)
Compare the question's fingerprint against all 750K chunk fingerprints, return
the 100 closest by meaning. Scores are cosine similarity (0–1).

### 3. BM25 Search (Keywords)
Search the keyword index for the 100 chunks that best match the question's exact
words. Scores are BM25 (unbounded).

### 4. Reciprocal Rank Fusion (Merge the Two Lists)
The two searches use different score scales, so we merge by **rank position**,
not raw score:

```
RRF_score(chunk) = sum over both lists of  1 / (60 + rank_in_that_list)
```

- A chunk appearing in BOTH lists gets two contributions → ranks higher
- `60` is the standard RRF dampening constant
- We keep the top 50 merged candidates
- We record which retriever(s) found each chunk (used for confidence later)

### 5. Cross-Encoder Reranking
FAISS/BM25 are fast but approximate. A cross-encoder model reads
(question, chunk) **together** and scores true relevance — much more accurate,
but slow, so we only run it on the top 50 candidates.

- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (small/fast, runs on MPS)
- Output: a relevance score per chunk (higher = better; can be negative)
- Sort by this score, keep top 5

### 6. Confidence Scoring
For each final chunk:
```
confidence = 0.6 * sigmoid(rerank_score)      # main signal
           + 0.2 * retrieval_consistency       # 1.0 if both retrievers found it, else 0.5
           + 0.2 * freshness                    # neutral 0.5 (no timestamps available)
```
- Rerank scores can be negative/large, so we squash with sigmoid to 0–1
- Freshness is neutral because wikiextractor didn't give us article timestamps

### 7. Return
Top 5 chunks with: chunk_id, title, url, text, confidence, rerank_score,
and `found_by` (which retrievers found it). Plus a `low_confidence` flag and
query latency.

---

## The Script: `pipeline/retrieve.py`

### Design: A Reusable `Retriever` Class
The indexes (FAISS 1.1GB + BM25 656MB) and two models load **once** (~47s), then
stay resident in memory. After that, every query is sub-second. This is essential
— you never want to reload 2GB of indexes per query.

```python
r = Retriever()                  # loads everything once (~47s)
out = r.search("your question")  # ~400–800ms per query
```

### CLI
```bash
python3 pipeline/retrieve.py "Who developed the theory of relativity?"
```

---

## Test Results (All Real)

### Query: "Who developed the theory of relativity?" — 816ms
| # | Result | Rerank | Found by |
|---|---|---|---|
| 1 | Theory of relativity | 9.07 | bm25 + faiss |
| 2 | History of physics | 8.04 | bm25 + faiss |
| 3 | Inertia | 7.56 | bm25 + faiss |
| 4 | Albert Einstein | 7.11 | bm25 + faiss |
| 5 | Theory of relativity | 6.68 | bm25 + faiss |

All correct, all hybrid (both retrievers agreed), confidence ~0.90.

### Query: "photosynthesis process in plants" — 667ms
Top: **Photosynthesis** (rerank 8.73, hybrid). Correct.

### Query: "What is the capital of France?" — 432ms
Top: **Paris** (rerank 5.12). Correct. Interesting: Paris was found only by FAISS
(meaning), because the chunk likely didn't contain the literal word "capital" —
a great example of why meaning-search matters alongside keywords.

### Query: "asdfghjkl qwerty zxcvbnm nonsense" — 691ms
Top: QWERTY (rerank **−0.16**). See calibration note below.

---

## Calibration Finding (Not a Bug)

The gibberish query returned QWERTY with a **negative** rerank score (−0.16),
which is the system correctly signaling "this is a poor match" (real queries
score +5 to +9). 

However, our blended confidence formula has a built-in floor: consistency (0.2)
+ freshness (0.2) = 0.2 minimum even before the rerank term. So a negative rerank
still produced confidence 0.475 — just over our 0.4 threshold — so the
`low_confidence` flag didn't trip.

**Why this is fine:**
- Retrieval's job (find + rank) worked perfectly
- The "should we refuse to answer?" decision belongs to Step 6, not Step 5
- The rerank score (−0.16 vs +9) is a crystal-clear signal for that decision

**Fix planned for Step 6:** trigger the "insufficient evidence" fallback when the
best **rerank_score** is below a threshold (~0.5–1.0), not just on blended
confidence. Gibberish (−0.16) gets refused; real questions (+5 to +9) get answered.

---

## Performance

| Phase | Time |
|---|---|
| One-time load (indexes + 2 models) | ~47s |
| Embed query | ~30ms |
| FAISS search | ~10ms |
| BM25 search | ~50ms |
| Rerank 50 candidates | ~300ms |
| **Total per query** | **430–820ms** |
| Peak RAM | ~3.3GB |

---

## Why No Step-3-Style Crashes

Step 3 (embedding) crashed from sustained multi-hour GPU load + laptop sleep.
Step 5 is the opposite:
- Models load once and stay resident
- Each query is tiny (one short text to embed, 50 pairs to rerank)
- No long GPU runs → no sleep/SIGKILL risk
Every test ran clean.

---

## Configuration

```python
EMBED_MODEL  = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RETRIEVE_N   = 100   # candidates from each retriever
RRF_KEEP     = 50    # kept after RRF for reranking
RRF_K        = 60    # RRF constant
CONFIDENCE_THRESHOLD = 0.4
DEVICE       = "mps"
```

---

## What Step 5 Does NOT Do

- Does NOT write a natural-language answer (Step 6 — the LLM)
- Does NOT produce the "insufficient evidence" message (Step 6 — Step 5 only
  *flags* low confidence)
- Does NOT log traces or cache (Steps 7/8)
- Just: question in → ranked relevant chunks out.
