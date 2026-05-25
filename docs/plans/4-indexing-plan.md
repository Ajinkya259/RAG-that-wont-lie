# Step 4 — Indexing Plan (Detailed)

## TL;DR (Plain English First)

Right now we have 750K number-fingerprints sitting in files. If someone asks a
question, we *could* compare it against all 750K one by one — but that's slow and
clumsy, and it only catches "meaning" matches, not exact keyword matches.

Step 4 builds two smart "search indexes" so lookups are instant:

1. **Vector index (FAISS)** — finds chunks with similar *meaning* (the fingerprints)
2. **Keyword index (BM25)** — finds chunks with matching *exact words*

Plus a small **metadata database** so once we find a chunk, we can instantly pull
its full text, title, and URL for citations.

**Step 4 = organize the 750K chunks into fast-searchable indexes.**

Real RAG systems use BOTH meaning-search and keyword-search together (called
"hybrid retrieval") because each catches what the other misses:
- Keyword search nails exact terms ("Python 3.12", names, codes)
- Meaning search nails paraphrases ("how cells split" → finds "mitosis")

---

## What We're Working With

| Thing | Value |
|---|---|
| Embedded chunks | 750,000 (first 750K lines of chunks.jsonl) |
| Vector files | `embeddings_00/01/02.npy` + `ids_00/01/02.json` |
| Vector dimensions | 384 (float16) |
| Source text | `data/chunks.jsonl` (first 750K lines) |

**Important:** The vector index, keyword index, and metadata DB must all cover
the *exact same* 750K chunks. Since embedding ran in file order, that's simply
the first 750,000 lines of `chunks.jsonl`. We'll use the `ids_*.json` files as
the source of truth for which chunk_ids are included.

---

## Part A — FAISS Vector Index

### Index Type Decision: IndexFlatIP (changed from the original IVFPQ plan)

The original plan called for `IndexIVFPQ` — but that was designed for **30M**
vectors where compression is essential. For our **750K** vectors:

| Option | RAM | Recall | Speed @ 750K | Verdict |
|---|---|---|---|---|
| **IndexFlatIP** (exact) | ~1.1GB | 100% (perfect) | ~5–15ms/query | ✅ **Use this** |
| IndexHNSWFlat | ~1.3GB | ~98% | ~2ms/query | overkill |
| IndexIVFPQ | ~0.1GB | ~95% | ~3ms/query | only needed at 10M+ |

**Why IndexFlatIP:**
- 750K × 384 × 4 bytes = ~1.1GB — fits easily in RAM
- Brute-force exact search on 750K is just milliseconds (FAISS uses SIMD)
- Perfect recall — no accuracy loss from compression
- Dead simple — no training step, no tuning

`IP` = Inner Product. Since our embeddings are L2-normalized, inner product =
cosine similarity. So nearest-neighbor by IP = most semantically similar.

### Build Steps
1. Load all 3 `.npy` files, concatenate → (750000, 384) array
2. Convert float16 → float32 (FAISS needs float32)
3. Create `faiss.IndexFlatIP(384)`
4. `index.add(vectors)` — no training needed for flat index
5. Save: `faiss.write_index(index, "indexes/faiss.index")`
6. Save the ordered list of chunk_ids → `indexes/faiss_ids.json`
   (maps FAISS row number → chunk_id)

### Output
- `indexes/faiss.index` (~1.1GB)
- `indexes/faiss_ids.json` (750K chunk_ids in row order)

---

## Part B — BM25 Keyword Index

### Library: bm25s
- Fast, pure-Python (NumPy), no Java/Elasticsearch needed
- Handles 750K docs easily in ~2–3GB RAM

### Build Steps
1. Read the first 750K chunks from `chunks.jsonl`
2. Extract `text` field from each
3. Tokenize (bm25s built-in tokenizer — lowercase, split, optional stopword removal)
4. Build BM25 index (default params: k1=1.5, b=0.75)
5. Save index + the chunk_id order to `indexes/bm25/`

### Output
- `indexes/bm25/` (bm25s saves a small directory of arrays)
- `indexes/bm25_ids.json` (chunk_ids in BM25 doc order)

---

## Part C — Metadata Store (SQLite)

### Why
When retrieval returns a chunk_id, we need to instantly look up its full text,
title, and URL — for building the LLM prompt and for citations. SQLite gives
O(1) lookup by chunk_id without loading the 19GB jsonl into RAM.

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
CREATE INDEX idx_chunk_id ON chunks(chunk_id);
```

### Build Steps
1. Read first 750K chunks from `chunks.jsonl`
2. Insert each into SQLite (batched inserts of 10K for speed)
3. Commit

### Output
- `indexes/chunk_metadata.db` (~1GB)

---

## Script: `pipeline/index.py`

### Usage
```bash
python3 pipeline/index.py --faiss     # build vector index
python3 pipeline/index.py --bm25      # build keyword index
python3 pipeline/index.py --metadata  # build SQLite store
python3 pipeline/index.py --all       # build all three
```

### Logic Flow
```
--faiss:
  load embeddings_*.npy → concat → float32
  IndexFlatIP(384).add(vectors)
  write faiss.index + faiss_ids.json

--bm25:
  read first 750K chunks from chunks.jsonl
  tokenize texts
  bm25s.BM25().index(tokens)
  save bm25/ + bm25_ids.json

--metadata:
  read first 750K chunks
  insert into SQLite (batched)
```

---

## Why This Step Is Safe (No Repeat of Step 3's Crashes)

Step 3 crashed because of sustained MPS GPU load + laptop sleep. Step 4 is
totally different:
- **No GPU** — all CPU/disk work
- **FAISS build** — just loading vectors into RAM, seconds to minutes
- **BM25 build** — CPU tokenization, ~5–10 min
- **SQLite** — disk I/O, ~5 min

Each part finishes in minutes, not hours. Even if interrupted, each part is
independent and quick to re-run.

---

## Time & Size Estimates

| Part | Time | Output Size |
|---|---|---|
| FAISS index | ~2–5 min | ~1.1GB |
| BM25 index | ~5–10 min | ~0.5GB |
| SQLite metadata | ~5 min | ~1GB |
| **Total** | **~15–20 min** | **~2.6GB** |

---

## Verification After Running

```python
# FAISS: load and test a search
import faiss, numpy as np, json
index = faiss.read_index("indexes/faiss.index")
print("FAISS vectors:", index.ntotal)   # expect 750000

# Search with the first embedding — should return itself as top hit
emb = np.load("data/embeddings/embeddings_00.npy")[0:1].astype('float32')
D, I = index.search(emb, 5)
print("Top hit distance:", D[0][0])      # expect ~1.0 (self-match)

# BM25: test a keyword query
import bm25s
retriever = bm25s.BM25.load("indexes/bm25")
# ...query "albert einstein physics"...

# SQLite: test a lookup
import sqlite3
con = sqlite3.connect("indexes/chunk_metadata.db")
row = con.execute("SELECT article_title FROM chunks LIMIT 1").fetchone()
print("Sample title:", row)
```

End-to-end sanity: pick a known chunk_id, confirm it exists in FAISS ids, BM25
ids, AND the metadata DB — all three must agree on the same 750K chunks.

---

## Dependencies Needed

```bash
pip3 install faiss-cpu bm25s
# sqlite3 is built into Python
```

---

## What Step 4 Does NOT Do

- Does NOT run any actual user query yet (that's Step 5 — retrieval)
- Does NOT do reranking or confidence scoring (Step 5)
- Does NOT call an LLM (Step 6)
- Just organizes the 750K chunks into 3 fast-lookup structures.
