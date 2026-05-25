# Step 3 — Embedding (What We Did + Issues Faced)

## TL;DR (Plain English)

Step 2 gave us 14M text chunks. A computer can't compare two sentences to see if
they mean the same thing — text is just letters to it. So Step 3 turns every chunk
into a list of 384 numbers (a "vector" or "embedding") that captures its meaning.
Two chunks about similar topics get similar numbers, so the computer can find
related text by comparing numbers instead of words.

**Step 3 = turn text chunks into numbers that represent meaning.**

We embedded **750,000 chunks** — more than enough for a working demo and testing.

---

## The Model We Used

`BAAI/bge-small-en-v1.5`
- Small (33M parameters), fast, open source, no API/internet needed
- Outputs 384 numbers per chunk
- Runs on the M4's GPU via Apple's MPS backend
- We normalized the vectors so similarity = simple dot product

---

## What We Produced

| Thing | Value |
|---|---|
| Chunks embedded | **750,000** |
| Output files | `embeddings_00/01/02.npy` (3 × 183MB) |
| Matching ID files | `ids_00/01/02.json` |
| Vector dimensions | 384 |
| Data type | float16 (half precision — saves disk) |
| Location | `data/embeddings/` |
| Device used | MPS (M4 GPU) |

Each `.npy` file holds 250,000 vectors. The matching `ids_XX.json` lists which
chunk each vector belongs to, in the same order.

Verified: all 3 batches load cleanly, shapes are (250000, 384), IDs match counts.

---

## Why We Stopped at 750K (Not the Full 14M or even 1M)

1. **750K is plenty for a demo** — a working RAG system feels real with 50K–100K
   chunks. 750K (≈150K Wikipedia articles) is generous.
2. **Embedding all 14M on a laptop would take days** and lag the machine the whole time.
3. **The last 250K of our 1M target kept crashing** (see issues below) and wasn't
   worth the fight — it adds no meaningful demo value.

So we locked Step 3 at 750K and moved on.

---

## Issues We Faced (The Real Story)

### Issue 1 — The machine hung at batch_size=512
**What happened:** First attempt used `batch_size=512`. The machine froze/lagged hard.

**Why:** On a CPU/GPU with no thread limits, PyTorch + OpenMP + MKL each spawned
their own threads (30+ total on 10 cores), all fighting for resources. A 512-chunk
forward pass was too heavy.

**Fix:**
- Set `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `TOKENIZERS_PARALLELISM` before importing torch
- Forced `device='mps'` (use M4 GPU, don't let it guess)
- Dropped `batch_size` to 64

### Issue 2 — Slow on CPU (74-hour ETA)
**What happened:** Early test showed ~52 chunks/sec → 74 hours for 14M chunks.

**Why:** Small batch on CPU is slow.

**Fix:** Switched to the M4 GPU (MPS). Also decided to embed a 1M subset, not all 14M.

### Issue 3 — Process kept dying mid-run (the big one)
**What happened:** The run completed 3 batches (750K) over ~5.5 hours, then died.
Every attempt to resume the final 250K died at the exact same point — right when
the first encode after resume started. No Python error, no traceback.

**Why (diagnosis):** A SIGKILL with no traceback = the OS killed the process
externally, not a code crash. RAM was only ~97MB so it wasn't memory exhaustion.
The most likely cause: **the laptop sleeping** (lid closing or system sleep).
`caffeinate -i` only blocks *idle* sleep — it does NOT prevent lid-close sleep.
Sustained MPS GPU work doesn't survive sleep.

**What we tried:**
- `caffeinate -i -s` to hold the system awake
- Smaller save batches (250K → 50K) so interruptions lose less
- Bumping batch size to 256 when machine was idle (also died)
- Capturing unbuffered output to catch any error (there was none → confirmed external kill)

**Resolution:** Rather than keep fighting the last 250K (which adds no demo value),
we accepted 750K as the final count for Step 3.

---

## Key Lessons

1. **Set thread limits before importing torch** — prevents thread-storm hangs.
2. **`caffeinate -i` is not enough** — it doesn't stop lid-close sleep. For long
   GPU jobs on a laptop, keep the lid open and plugged in, or use `caffeinate -dis`.
3. **Checkpoint frequently** — saving every 50K (vs 250K) meant we never lost much.
4. **Know when "enough" is enough** — 750K embeddings is a complete, demo-ready
   corpus. Chasing 100% wasn't worth the time.
5. **float16 halves storage** with negligible quality loss for retrieval.

---

## The Script

`pipeline/embed.py`
- Loads chunks from `data/chunks.jsonl`
- Embeds in batches with `bge-small-en-v1.5` on MPS
- Saves float16 `.npy` + matching `ids.json` per batch
- Checkpoints to `data/embed_checkpoint.json` (resume-safe)

Config used: `BATCH_SIZE=64`, `device='mps'`, thread limits set, normalized embeddings.

---

## What Step 3 Did NOT Do

- Did NOT build the search index (Step 4 — FAISS + BM25)
- Did NOT do any retrieval/search (Step 5)
- Did NOT touch an LLM (Step 6)
- Just converted 750K text chunks → 750K vectors of 384 numbers each.

---

## Current Data State

```
data/
├── chunks.jsonl            (13.98M chunks, 19GB)
└── embeddings/
    ├── embeddings_00.npy    (250K vectors, 183MB)
    ├── embeddings_01.npy    (250K vectors, 183MB)
    ├── embeddings_02.npy    (250K vectors, 183MB)
    ├── ids_00.json          (250K chunk_ids)
    ├── ids_01.json          (250K chunk_ids)
    └── ids_02.json          (250K chunk_ids)
```

**750,000 chunks embedded and ready for Step 4 (indexing).**
