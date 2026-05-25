# Step 2 — What We Did (In Plain English)

## TL;DR (Simplest Version)

Step 1 gave us full Wikipedia articles. Each article is like a full book chapter —
way too long to search through efficiently.

Imagine you're looking for "what did Einstein say about gravity?" and someone hands
you a 10-page article about Einstein. You'd have to read the whole thing to find
the one paragraph that answers your question.

**Step 2 = cut every article into small bite-sized pieces.**

---

## The Cutting (Chunking)

Each piece is ~300 words (384 tokens). And each piece slightly overlaps with the
next one by ~50 words — so if the answer happens to be near a cut point, you
don't lose it.

Like cutting a long rope into sections, but letting each section share a few
inches with the next one.

Every piece also gets a label stitched onto it:
- Which article it came from
- The article's URL (for citations later)
- Its position in the article (chunk 3 of 7)
- A unique ID like `12_0003`

So when we find a relevant chunk later, we know exactly where it came from.

---

## What Got Filtered Out

Before cutting, we threw away bad pages:

| What | Why |
|---|---|
| Empty pages | These were redirect pages — "USA" → "United States", no real content |
| Pages < 300 chars | Stub articles — basically empty |
| "List of..." pages | Just bullet points, no real prose |
| Disambiguation pages | Just links to other pages, no content |
| Pages with ":" in title | Namespace pages that leaked through (Category:, File:, etc.) |

This dropped us from **18.8M raw pages → 13.98M usable chunks** from real articles.

---

## How the Cutting Works (The Splitter)

We used a smart splitter called `RecursiveCharacterTextSplitter`. It tries to cut
at natural language boundaries in this order:

1. Double newline `\n\n` — paragraph break (best place to cut)
2. Single newline `\n` — line break
3. Period + space `. ` — end of sentence
4. Space ` ` — between words
5. Character by character — last resort

So it never cuts in the middle of a sentence if it can avoid it.

---

## Exact Parameters Used

```
Chunk size    : 384 tokens (~300 words)
Overlap       : 50 tokens (~40 words shared between consecutive chunks)
Min chunk size: 100 tokens (drop anything smaller)
Min article   : 300 characters (filter out stubs)
Workers       : 8 CPU cores running in parallel
```

---

## What Each Chunk Looks Like

Every line in `chunks.jsonl` is one chunk:

```json
{
  "chunk_id": "12_0003",
  "article_id": "12",
  "article_title": "Anarchism",
  "article_url": "https://en.wikipedia.org/wiki?curid=12",
  "timestamp": "",
  "chunk_index": 3,
  "total_chunks": 7,
  "text": "Anarchists employ a range of approaches to social change..."
}
```

---

## The Script

`pipeline/chunk.py` — walks all 201 folders in `wiki_extracted/`, processes
each file using 8 parallel workers, writes one chunk per line to `chunks.jsonl`,
and checkpoints every 100 files so it can resume if it crashes.

---

## Stats

| Thing | Value |
|---|---|
| Input | `wiki_extracted/` — 201 folders, 20,095 files, 20GB |
| Raw pages | 18,883,645 |
| After filtering | ~8.7M usable articles |
| Total chunks written | **13,980,714** |
| Output file | `data/chunks.jsonl` |
| Output size | **19GB** |
| Time taken | ~15 minutes (8 cores) |
| Workers | 8 CPU cores |

---

## What Step 2 Did NOT Do

- Did NOT create vectors or embeddings (Step 3)
- Did NOT build any search index (Step 4)
- Did NOT do any AI/ML — purely text splitting
- Did NOT need internet connection

It purely took long articles and cut them into small searchable pieces with labels.
Nothing more, nothing less.
