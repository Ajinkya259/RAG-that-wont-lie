# Step 4 — Indexing (What We Did, In Detail)

## TL;DR (Plain English)

After Step 3 we had 750K number-fingerprints in files. But files aren't
searchable — if someone asked a question, we'd have to compare it against all
750K one by one. Slow and clumsy.

Step 4 organized those 750K chunks into **three fast-lookup structures**:

1. **FAISS** — a vector index for *meaning* search (uses the fingerprints)
2. **BM25** — a keyword index for *exact word* search
3. **SQLite DB** — a lookup table to fetch a chunk's full text/title/URL instantly

**Step 4 = build the searchable indexes so lookups are instant.**

It took **1.2 minutes** total. (Compare to Step 3's multi-hour GPU pain — this
step is pure CPU/disk and finished in one go, no crashes.)

---

## Why Three Indexes Instead of One?

Real RAG systems use **hybrid retrieval** — combining two kinds of search because
each catches what the other misses:

| Search type | Good at | Misses |
|---|---|---|
| **Meaning (FAISS)** | Paraphrases — "how cells divide" finds "mitosis" | Exact rare terms, codes, names |
| **Keyword (BM25)** | Exact words — "Python 3.12", "GDP", proper nouns | Synonyms, rephrasing |

Then the **SQLite DB** is just the "phone book" — once a search returns a chunk_id,
we look up its full content there for building the answer and citations.

---

## Part A — FAISS Vector Index (Meaning Search)

### The Big Decision: We Changed the Index Type

The original plan called for `IndexIVFPQ` — a *compressed* index designed for
**30 million** vectors, where you must shrink data to fit in RAM. But we only
embedded **750K** vectors. At that scale, compression is pointless and only
hurts accuracy.

So we switched to **`IndexFlatIP`**:

| | IndexIVFPQ (old plan) | IndexFlatIP (what we used) |
|---|---|---|
| Designed for | 10M–1B vectors | up to ~1–2M vectors |
| Accuracy | ~95% (lossy compression) | 100% (exact) |
| RAM for 750K | ~0.1GB | ~1.1GB |
| Search speed @ 750K | ~3ms | ~5–15ms |
| Setup | needs training step | none — just add vectors |

**"IP" = Inner Product.** Because our embeddings are L2-normalized (Step 3),
inner product equals cosine similarity. So finding the highest inner product =
finding the most semantically similar chunk.

### How We Built It
1. Loaded all 3 `.npy` files → concatenated into one (750000, 384) array
2. Converted float16 → float32 (FAISS requires float32)
3. Created `faiss.IndexFlatIP(384)` and added all 750K vectors (no training needed)
4. Saved `indexes/faiss.index` (1.1GB)
5. Saved `indexes/faiss_ids.json` — maps each FAISS row number → its chunk_id

**Why the ids file matters:** FAISS only knows vectors by row number (0, 1, 2...).
The ids file translates "row 47 is the closest match" → "that's chunk 12_0003".

---

## Part B — BM25 Keyword Index (Word Search)

### Library: bm25s
- Fast, pure-Python (NumPy under the hood)
- No Java or Elasticsearch server needed
- Handles 750K docs easily

### How We Built It
1. Read the first 750K chunks from `chunks.jsonl`
2. Extracted the `text` field from each
3. Tokenized — split into words, lowercased, stemmed (bm25s built-in)
4. Built the BM25 index (standard params: k1=1.5, b=0.75)
5. Saved `indexes/bm25/` (656MB) + `indexes/bm25_ids.json`

**What BM25 actually does:** It scores how well a document matches query keywords,
accounting for (a) how often the word appears and (b) how rare the word is across
all docs. Rare words count more — matching "photosynthesis" matters more than
matching "the".

---

## Part C — SQLite Metadata Store (The Phone Book)

### Why
When a search returns chunk_id `12_0003`, we need its full text, title, and URL
— for building the LLM prompt and citations. We can't load the 19GB chunks.jsonl
into RAM every query. SQLite gives instant O(1) lookup by chunk_id.

### Schema
```sql
CREATE TABLE chunks (
    chunk_id      TEXT PRIMARY KEY,
    article_title TEXT,
    article_url   TEXT,
    timestamp     TEXT,
    chunk_index   INTEGER,
    total_chunks  INTEGER,
    text          TEXT
);
```

### How We Built It
1. Read first 750K chunks from `chunks.jsonl`
2. Inserted in batches of 10K (fast bulk inserts)
3. Committed → `indexes/chunk_metadata.db` (1.3GB)

---

## The Critical Rule: All Three Must Cover the SAME Chunks

The vector index, keyword index, and metadata DB must all describe the **exact
same 750K chunks** — otherwise a search could return a chunk_id that doesn't exist
in another index.

Since embedding ran in file order, the embedded chunks are simply the first 750K
lines of `chunks.jsonl`. We used the `ids_*.json` files (from Step 3) as the
source of truth for which chunk_ids are included, and filtered all three builds
to that exact set.

---

## The Script

`pipeline/index.py` — run modes:
```bash
python3 pipeline/index.py --faiss      # vector index only
python3 pipeline/index.py --bm25       # keyword index only
python3 pipeline/index.py --metadata   # SQLite store only
python3 pipeline/index.py --all        # all three
```

We ran `--faiss` first (instant), then `--bm25 --metadata` together (1.2 min).

---

## Verification (All Passed)

| Check | Result |
|---|---|
| FAISS vector count | 750,000 ✓ |
| FAISS self-search (vector vs itself) | score 1.0001 (perfect match) ✓ |
| Metadata row count | 750,000 ✓ |
| chunk_id lookup `12_0000` | → "Anarchism" ✓ |
| BM25 documents indexed | 750,000 ✓ |
| **All three indexes cover identical chunk set** | **True ✓** |

---

## Final Output

```
indexes/
├── faiss.index          1.1GB   vector/meaning search
├── faiss_ids.json       10MB    FAISS row → chunk_id
├── bm25/                656MB   keyword search index
├── bm25_ids.json        10MB    BM25 doc → chunk_id
└── chunk_metadata.db    1.3GB   chunk_id → full text/title/url
```

Total: ~3.1GB, 750K chunks, all three indexes aligned.

---

## Why This Step Was Painless (Unlike Step 3)

Step 3 (embedding) crashed repeatedly due to sustained GPU load + laptop sleep.
Step 4 was completely different:
- **No GPU** — all CPU/disk
- **FAISS** — just loading vectors into RAM (seconds)
- **BM25** — CPU tokenization (~1 min)
- **SQLite** — batched disk inserts (seconds)

Everything finished in 1.2 minutes, one run, no issues.

---

## What Step 4 Did NOT Do

- Did NOT run a real user query yet (Step 5 — retrieval)
- Did NOT merge results or rerank (Step 5)
- Did NOT call an LLM (Step 6)
- Just organized the 750K chunks into 3 fast-lookup structures.
