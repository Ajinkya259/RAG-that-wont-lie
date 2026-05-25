# Step 2 — Chunking Plan (Detailed)

## TL;DR (Plain English First)

Step 1 gave us 20GB of clean articles. But each article is way too long to feed
into an AI search system. Imagine asking someone "find me relevant text about
gravity" and handing them a 10-page article — they'd have to read the whole
thing.

Instead we cut each article into small pieces (~384 tokens = ~300 words each).
Each piece overlaps slightly with the next one (50 tokens) so we don't lose
context at the edges. Every piece remembers where it came from (which article,
which URL, which timestamp) so we can cite it later.

**Step 2 = cut articles into small searchable pieces with labels on them.**

---

## What We're Working With (Real Numbers)

From sampling the actual extracted data:

| Metric | Value |
|---|---|
| Total extracted articles | 18,883,645 |
| Empty / useless pages | ~54% (redirects, stubs) |
| Usable articles (≥300 chars) | ~8.7M |
| Average article length | ~6,400 characters |
| Estimated chunks per article | ~4–5 |
| **Estimated total chunks** | **~35M chunks** |
| Input location | `wiki_extracted/` (201 folders, 20GB) |
| Output location | `data/chunks.jsonl` |

---

## What Each Chunk Will Look Like

Every chunk = one JSON line in `chunks.jsonl`:

```json
{
  "chunk_id": "12_0003",
  "article_id": "12",
  "article_title": "Anarchism",
  "article_url": "https://en.wikipedia.org/wiki?curid=12",
  "timestamp": "2024-01-15",
  "chunk_index": 3,
  "total_chunks": 7,
  "text": "Anarchists employ a range of approaches to social change, often
           categorised as revolutionary or evolutionary, though the two
           frequently overlap..."
}
```

The `chunk_id` = article_id + chunk index. This is the unique key for every
chunk in the whole system — used in retrieval, citations, and observability.

---

## Filtering Rules (What Gets Dropped)

Before chunking, filter out bad articles:

| Rule | Why |
|---|---|
| `text` is empty | Wikiextractor emits empty text for redirect pages |
| `text` length < 300 chars | Too short to be useful — stubs, list pages |
| title contains `"List of"` | List pages have no prose, just bullet points |
| title contains `"(disambiguation)"` | Just links to other pages, no content |
| title contains `":"` | Namespace pages (File:, Category:, etc.) leaked through |

Expected: drops ~54% → keeps ~8.7M usable articles.

---

## Chunking Strategy

### Why Not Just Split by Paragraph?
Wikipedia paragraphs vary wildly — some are 2 sentences, some are 20. A search
index works best when chunks are roughly the same size.

### Why 384 Tokens?
- Small enough to be precise in retrieval (you get exactly the relevant passage)
- Large enough to have real context (not just 2 sentences)
- Matches our embedding model's sweet spot (bge-small-en-v1.5 was trained on ~256–512 token passages)

### Why 50 Token Overlap?
When you cut an article into pieces, sometimes important context is split across
two chunks. The 50-token overlap means each chunk shares a bit with its neighbor,
so nothing important falls through the cracks.

### The Splitter: RecursiveCharacterTextSplitter
Splits text in this priority order:
1. Double newline `\n\n` (paragraph boundary) — split here first
2. Single newline `\n` (line boundary)
3. Period + space `. ` (sentence boundary)
4. Space ` ` (word boundary)
5. Character by character (last resort)

This means it always tries to cut at natural language boundaries first,
not in the middle of a sentence.

---

## Exact Parameters

```python
chunk_size    = 384    # tokens (using tiktoken cl100k_base tokenizer)
chunk_overlap = 50     # tokens shared between consecutive chunks
min_chunk_len = 100    # tokens — drop chunks smaller than this
```

---

## Processing Strategy

### Problem: 20GB input, 35M output chunks
Can't load everything into RAM. Must stream.

### Solution: Process file by file
- Walk through all 201 folders in `wiki_extracted/`
- For each `wiki_XX` file, read line by line
- Filter → chunk → write to `chunks.jsonl`
- Never load more than one file at a time into memory

### Parallelism
- Use Python `multiprocessing.Pool` with 8 workers
- Each worker handles one wiki file at a time
- Output written to `chunks.jsonl` with a lock to avoid corruption
- Progress tracked with tqdm

### Checkpointing
- Track which files are done in `data/chunk_checkpoint.json`
- If script crashes, resume from last completed file

---

## Script: `pipeline/chunk.py`

### Inputs
- `wiki_extracted/` — all wikiextractor output folders

### Outputs
- `data/chunks.jsonl` — all chunks, one per line (~35M lines)

### Logic Flow

```
For each folder in wiki_extracted/:
  For each wiki_XX file in folder:
    For each line (article) in file:
      1. Parse JSON
      2. Apply filters (empty, too short, bad title)
      3. Split text into chunks (RecursiveCharacterTextSplitter)
      4. For each chunk:
         - Build chunk_id = f"{article_id}_{chunk_index:04d}"
         - Attach metadata (title, url, timestamp, indices)
         - Write JSON line to chunks.jsonl
```

### Timestamp Handling
Wikiextractor doesn't include `timestamp` in the output — it only has
`id`, `revid`, `url`, `title`, `text`. We'll generate a placeholder
timestamp from the `revid` field, or leave it as `""` and fill it in
from the original XML dump later if needed. For now, `""` is fine —
the freshness scoring in Phase 5 will just treat unknown timestamps
as neutral (score 0.5).

---

## Time & Size Estimates

| Step | Estimate |
|---|---|
| Filtering 18.8M articles | ~20 min |
| Chunking 8.7M articles (8 cores) | ~40–60 min |
| Writing 35M lines to disk | ~30 min |
| **Total** | **~1.5–2 hours** |
| Output file size | ~30–35GB |

---

## Verification Checks (After Running)

```bash
# Count total chunks
wc -l data/chunks.jsonl
# Expect: ~30–35 million

# Spot check a random chunk
shuf -n 1 data/chunks.jsonl | python3 -m json.tool

# Check no empty texts snuck through
python3 -c "
import json
bad = 0
with open('data/chunks.jsonl') as f:
    for line in f:
        c = json.loads(line)
        if not c['text'].strip():
            bad += 1
print(f'Bad chunks: {bad}')
"
```

---

## What Step 2 Does NOT Do

- Does NOT create vectors/embeddings (Step 3)
- Does NOT build any index (Step 4)
- Does NOT do any AI/ML — purely text processing
- Does NOT need internet connection

---

## Dependencies Needed

```bash
pip3 install langchain-text-splitters tiktoken tqdm
```
