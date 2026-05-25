# Step 3 — Embedding Plan (Detailed)

## TL;DR (Plain English First)

Step 2 gave us 14M text chunks. But text is just words — a computer can't
mathematically compare two sentences to figure out if they mean the same thing.

Step 3 converts every chunk into a list of 384 numbers (called a vector or
embedding). These numbers capture the *meaning* of the text. Two chunks that
talk about similar things will have similar numbers — so the computer can find
them by comparing numbers instead of words.

**Step 3 = turn every chunk from words into numbers that represent meaning.**

Example:
- "Einstein developed the theory of relativity" → [0.23, -0.11, 0.87, ...]
- "Einstein's special relativity changed physics" → [0.21, -0.09, 0.84, ...]  ← similar numbers
- "The cat sat on the mat" → [-0.45, 0.33, -0.12, ...]  ← very different numbers

---

## Real Numbers (From Actual Data)

| Metric | Value |
|---|---|
| Total chunks to embed | 13,980,714 |
| Avg tokens per chunk | 307 |
| Embedding model | BAAI/bge-small-en-v1.5 |
| Output dimensions | 384 numbers per chunk |
| Raw embedding size (float32) | 21.5 GB |
| Raw embedding size (float16) | **10.7 GB** ← we use this |
| Saved as | 28 files × 500K chunks each (~384 MB per file) |
| Est. time (8 cores, batch=512) | **~2–3 hours** |

---

## The Embedding Model: BAAI/bge-small-en-v1.5

**Why this model?**
- Small (33M parameters) — fast on CPU, fits easily in RAM
- 384-dim output — compact but powerful
- Strong on Wikipedia-style factual text (84%+ BEIR benchmark)
- Open source, free, no API needed

**What it does:**
Takes a sentence/paragraph → runs it through a neural network → spits out
384 numbers that encode the semantic meaning.

**BGE instruction prefix (important):**
- When embedding CORPUS chunks (what we do in Step 3): NO prefix needed
- When embedding QUERIES at search time (Step 5): prepend
  `"Represent this sentence for searching relevant passages: "` to the query

---

## Memory Strategy: Save as float16 in Batches

21.5GB (float32) is too big to hold in RAM all at once.
float16 halves it to 10.7GB — still big but manageable.

We save in 28 batches of 500K chunks each:
- `data/embeddings/embeddings_00.npy` (~384 MB)
- `data/embeddings/embeddings_01.npy` (~384 MB)
- ...
- `data/embeddings/embeddings_27.npy`

Each file also has a matching ID file:
- `data/embeddings/ids_00.json` — list of chunk_ids in same order as embeddings

This lets us load one batch at a time into FAISS in Step 4 without OOM.

---

## Parallelism Strategy

CPU embedding is slow single-threaded. We speed it up by:

1. **Large batch size (512)** — GPU-style batching even on CPU
   - Model processes 512 chunks at once vs 1 at a time
   - Uses vectorized matrix operations → ~10x faster than batch=1

2. **sentence-transformers `pool` mode** — uses multiple CPU threads internally

3. **No multiprocessing** (unlike Step 2) — the model can't safely be shared
   across processes due to PyTorch internals. Batch size does the heavy lifting.

---

## Checkpointing

13.98M chunks takes hours. If it crashes midway, we don't want to restart.

Strategy:
- Process in batches of 500K chunks
- After each batch: save embeddings to disk, save progress to `data/embed_checkpoint.json`
- On restart: load checkpoint, skip already-done batches

```json
// data/embed_checkpoint.json
{
  "batches_done": 14,
  "chunks_done": 7000000,
  "total_chunks": 13980714
}
```

---

## Script: `pipeline/embed.py`

### Inputs
- `data/chunks.jsonl` — 13.98M chunks

### Outputs
- `data/embeddings/embeddings_XX.npy` — float16 numpy arrays
- `data/embeddings/ids_XX.json` — chunk_ids matching each embedding
- `data/embed_checkpoint.json` — progress tracker

### Logic Flow

```
Load model: bge-small-en-v1.5
Load checkpoint (resume if exists)
Open chunks.jsonl

For each batch of 500K chunks:
  Skip if already done (checkpoint)
  Extract texts from batch
  Encode with model (batch_size=512)
  Convert to float16
  Save embeddings_XX.npy
  Save ids_XX.json
  Update checkpoint
  Print progress

Done → print total time + chunks embedded
```

---

## Exact Model Parameters

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-small-en-v1.5')

embeddings = model.encode(
    texts,
    batch_size=512,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,   # L2 normalize → cosine sim = dot product
)
```

`normalize_embeddings=True` is critical — it means at search time we can use
fast dot product instead of slow cosine distance calculation.

---

## Time & Size Estimates

| Batch | Chunks | Time (est.) | File Size |
|---|---|---|---|
| 0 | 0–500K | ~15 min | 384 MB |
| 1 | 500K–1M | ~15 min | 384 MB |
| ... | ... | ... | ... |
| 27 | 13.5M–14M | ~15 min | ~370 MB |
| **Total** | **13.98M** | **~2–3 hrs** | **~10.7 GB** |

---

## Verification After Running

```bash
# Check all 28 batch files exist
ls data/embeddings/ | wc -l
# Expect: 56 files (28 .npy + 28 .json)

# Check shape of first batch
python3 -c "
import numpy as np
e = np.load('data/embeddings/embeddings_00.npy')
print(f'Shape: {e.shape}')      # expect (500000, 384)
print(f'Dtype: {e.dtype}')      # expect float16
print(f'Sample: {e[0][:5]}')    # 5 numbers from first embedding
"

# Sanity: two similar chunks should have high dot product
python3 -c "
import numpy as np
e = np.load('data/embeddings/embeddings_00.npy')
# Dot product of a vector with itself = 1.0 (normalized)
print(f'Self similarity: {np.dot(e[0], e[0]):.4f}')  # should be ~1.0
# Two random vectors should have low similarity
print(f'Random pair sim: {np.dot(e[0], e[100]):.4f}') # should be low
"
```

---

## What Step 3 Does NOT Do

- Does NOT build the search index (Step 4)
- Does NOT do any retrieval or search (Step 5)
- Does NOT talk to any LLM (Step 6)
- Just converts text → numbers. Pure math.

---

## Dependencies Needed

```bash
pip3 install sentence-transformers numpy
```

First run will also download the model (~130MB) from HuggingFace automatically.
