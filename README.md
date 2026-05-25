# RAG that won't lie

A retrieval-augmented generation system built from scratch over the **full English
Wikipedia**, on a laptop, fully offline. The goal that started it: the "design a
RAG system for 10 million documents with near-zero hallucination" question that
keeps showing up online. So I gave it a shot and shipped the real thing.

No vector-DB SaaS. No OpenAI key. No framework magic. Just the actual pipeline,
end to end, with every step explained from first principles in [`docs/`](docs/).

> **Q:** Who developed the theory of relativity?
> **A:** Albert Einstein developed the theory of relativity. `[1]`
> **Sources:** [1] Theory of relativity — en.wikipedia.org
>
> **Q:** What did Albert Einstein eat for breakfast?
> **A:** I don't have enough reliable information to answer that.

The second answer is the whole point. Einstein is *in* the corpus, so retrieval
finds him — but the chunks don't mention breakfast, so the system **refuses
instead of making something up.** That is what "won't lie" means here.

---

## How it works

```
Wikipedia dump (25GB bz2)
      │
      ▼
[1] PARSE       wikiextractor → clean text per article
      │
      ▼
[2] CHUNK       sentence-aware split, 384 tokens, 50 overlap   → 13.98M chunks
      │
      ▼
[3] EMBED       bge-small-en-v1.5 → 384-dim vectors (float16)
      │
      ▼
[4] INDEX       FAISS (meaning) + BM25 (keywords) + SQLite (metadata)
      │
      ▼
[5] RETRIEVE    hybrid search → Reciprocal Rank Fusion → cross-encoder rerank
      │
      ▼
[6] GENERATE    local Qwen2.5, context-only prompt, cite sources, refuse if unsure
```

Every box is one script in [`pipeline/`](pipeline/) and one explainer in
[`docs/`](docs/). The explainers are written in plain English first, then the
technical detail — readable whether or not you've built RAG before.

---

## Why "won't lie" — the two-layer anti-hallucination design

1. **Confidence gate (before the LLM).** After reranking, if the best match
   scores below a floor, the system refuses immediately and never calls the LLM.
   Catches gibberish and off-topic questions.
2. **Constrained generation (inside the LLM).** The model is instructed to answer
   *only* from the retrieved passages and to say "Insufficient evidence" if the
   answer isn't there. Catches questions that are on-topic but unanswerable from
   the corpus (the Einstein-breakfast case).

Plus citations: every answer is traceable back to the Wikipedia articles it used.

---

## All ten layers

The blueprint this started from is a ten-layer RAG pipeline. All ten are implemented:

| # | Layer | Where |
|---|---|---|
| 01 | Ingest + normalize | `pipeline/parse.py`, `pipeline/chunk.py` |
| 02 | Hybrid retrieval (BM25 + embeddings) | `pipeline/embed.py`, `pipeline/index.py` |
| 03 | ANN + reranking (two-stage) | `pipeline/retrieve.py` |
| 04 | Source confidence scoring | `pipeline/retrieve.py` |
| 05 | Constrained generation | `pipeline/generate.py` |
| 06 | Citation-backed responses | `pipeline/generate.py` |
| 07 | Hallucination fallback | `pipeline/generate.py` |
| 08 | Continuous evals | `pipeline/evaluate.py` |
| 09 | Caching + memory | `pipeline/cache.py` |
| 10 | Observability | `pipeline/observability.py` |

**The three production layers, measured on a real run:**

- **Evals** (`python3 pipeline/evaluate.py`) — a curated set across three buckets
  (answerable / unanswerable-on-topic / adversarial):
  ```
  answerable: answer rate    100.0%   (6 q)
  answerable: citation rate  100.0%
  refusal rate (unans + adv) 100.0%   (6 q)  <- anti-hallucination
  overall behaved as expected 100.0%
  ```
- **Caching** (`pipeline/cache.py`) — a repeated question returns from SQLite in
  ~0.001s instead of ~10s of generation.
- **Observability** (`pipeline/observability.py --tail N`) — every query writes a
  trace: BM25 / FAISS / RRF order, reranked top-5 with scores, the gate verdict,
  and per-stage latency. Any answer can be explained after the fact.

---

## The numbers (real run, on an Apple M4)

| Stage | Result |
|---|---|
| Source | English Wikipedia dump, ~25GB compressed |
| Articles extracted | 18.8M (incl. redirects/stubs) |
| Usable chunks | **13.98M** |
| Chunks embedded (this build) | 750,000 |
| Embedding dim | 384 (float16) |
| Indexes | FAISS `IndexFlatIP` + BM25 + SQLite |
| Query latency | ~0.4–0.8s retrieval, ~10–15s generation (CPU) |

> This build embedded the first 750K chunks — plenty for a working, demo-able
> system. The pipeline scales to all 14M; it's just hours of compute. See
> [`docs/3-embedding.md`](docs/3-embedding.md) for why I stopped at 750K.

---

## What broke (the honest part)

Building this on one laptop surfaced real, specific problems — not textbook ones:

- **The first parser would've taken 23 days.** A hand-rolled `mwparserfromhell`
  loop was too slow for 22M pages. Switched to `wikiextractor` (multiprocess) →
  done in ~3 hours. ([`docs/1-parsing.md`](docs/1-parsing.md))
- **Embedding kept dying overnight.** The job got SIGKILLed every time the laptop
  slept. `caffeinate -i` wasn't enough (it doesn't stop lid-close sleep).
  ([`docs/3-embedding.md`](docs/3-embedding.md))
- **"Python quit unexpectedly" — repeatedly.** Generation segfaulted on every
  multi-query run. I blamed the GPU, then float16. Both wrong. The crash report
  named the real culprit: **`libomp.dylib`** — `faiss` and `torch` each ship their
  own OpenMP runtime and collide on a thread barrier. Fix: force single-threaded
  OpenMP + `KMP_DUPLICATE_LIB_OK=TRUE`. The lesson: read the crash report before
  guessing. ([`docs/6-generation.md`](docs/6-generation.md))

These are documented in full where they happened, because the debugging *is* the
build.

---

## Run it yourself

```bash
pip install -r requirements.txt

# 1. Get the Wikipedia dump (~25GB)
curl -C - -L -o enwiki-latest-pages-articles-multistream.xml.bz2 \
  https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream.xml.bz2

# 2. Parse → clean text
wikiextractor --json --processes 10 -o wiki_extracted/ \
  enwiki-latest-pages-articles-multistream.xml.bz2

# 3. Chunk
python3 pipeline/chunk.py

# 4. Embed (set how many in pipeline/embed.py; uses MPS/CPU)
python3 pipeline/embed.py

# 5. Build indexes
python3 pipeline/index.py --all

# 6. Ask a question
LLM_DEVICE=cpu RETRIEVER_DEVICE=cpu \
  python3 pipeline/generate.py "Who developed the theory of relativity?"
```

The 25GB dump, the chunks, the embeddings, and the indexes are all gitignored —
this repo is the *recipe*, not the groceries.

---

## Repo layout

```
RAG-that-wont-lie/
├── README.md
├── requirements.txt
├── plan.md                 — the architecture plan (and how it was revised)
├── pipeline/               — one script per stage / layer
│   ├── parse.py            — (legacy slow parser, kept for the story)
│   ├── chunk.py
│   ├── embed.py
│   ├── embed_test.py       — safe single-batch smoke test
│   ├── index.py            — FAISS + BM25 + SQLite
│   ├── retrieve.py         — hybrid retrieval, RRF, rerank, confidence
│   ├── generate.py         — constrained generation, gate, citations
│   ├── cache.py            — layer 09: query cache
│   ├── observability.py    — layer 10: trace logging + viewer
│   ├── evaluate.py         — layer 08: continuous evals
│   └── test_e2e.py
└── docs/
    ├── 1-parsing.md … 6-generation.md   — plain-English + technical, per stage
    ├── plans/                            — the planning docs per stage
    └── reel-transcript.txt               — the 10-step problem that started it
```

---

## Stack

Python · wikiextractor · `bge-small-en-v1.5` (embeddings) · FAISS · bm25s ·
`ms-marco-MiniLM-L-6-v2` (reranker) · Qwen2.5 (generation) · SQLite. All open
source, all local.

---

## A note on how this was built

This was vibe-coded — built conversationally with Claude, end to end, including
the debugging. The docs capture the real path: the wrong turns, the crashes, and
the fixes, not a cleaned-up after-the-fact version.
