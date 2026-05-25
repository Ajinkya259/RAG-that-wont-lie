# Step 1 — What We Did (In Plain English)

## TL;DR (Simplest Version)

Wikipedia is basically a giant book. But the file they give you isn't a normal
readable book — it's written in a weird internal code that looks like this:

```
{{Infobox person | name = Einstein}}
[[theoretical physicist]] who developed <ref>{{cite}}</ref>
```

Garbage, right? Nobody can use that.

**Step 1 = clean that garbage into normal readable text. That's literally it.**

We first wrote our own cleaner — worked fine but would've taken 23 days for the
full file. Too slow.

So we used a tool called `wikiextractor` — purpose-built for this exact job,
runs on all 10 CPU cores at once. Took 3 hours instead of 23 days.

End result: 20GB of clean JSON files in `wiki_extracted/`. Each line looks like:

```json
{"title": "Anarchism", "text": "Anarchism is a political philosophy...", "url": "..."}
```

That's it. Messy 25GB Wikipedia blob → clean readable text. Nothing fancy yet.

---

## The Big Picture

Wikipedia gives you a single giant file with every article ever written — but it's
in a format that computers use internally, not something you can just read and use.
Think of it like getting a book written in a secret code. Step 1 was about
**decoding that file and pulling out the clean, readable text.**

---

## The File We Started With

`enwiki-latest-pages-articles-multistream.xml.bz2` — 25GB compressed file.

What's inside:
- Every English Wikipedia article (~6.7M real articles)
- Plus millions of redirect pages ("Barack Obama" → "Obama")
- Plus category pages, template pages, talk pages, disambiguation pages
- All written in "wikitext" — a messy markup language that looks like this:

```
{{Infobox person
| name = Albert Einstein
| birth_date = {{birth date|1879|3|14}}
}}
Albert Einstein ({{IPA|...}}) was a [[theoretical physicist]] who...
<ref>{{cite book|title=...}}</ref>
```

Nobody wants that. We want clean text like:

```
Albert Einstein was a theoretical physicist who...
```

---

## Why We Couldn't Just Open It Normally

The file is 25GB compressed. When you open it, it expands to ~100GB of raw XML.
You can't load 100GB into RAM — your laptop would crash.

So we needed a tool that reads it **one article at a time**, processes it, and
moves on — never loading the whole thing into memory. Like reading a book one
page at a time instead of memorizing the whole thing first.

---

## What We First Tried (and Why It Failed)

We wrote our own Python script (`pipeline/parse.py`) using a library called
`mwparserfromhell`. It worked correctly — but it was **way too slow**.

After 31 minutes it had processed only ~20,000 pages. The dump has 22 million
pages. At that rate it would take **23 days** to finish.

Why so slow? `mwparserfromhell` does very careful, thorough cleanup of every
single template and markup tag. Great quality, terrible speed.

---

## What We Actually Used — Wikiextractor

We switched to a tool called `wikiextractor`. It's purpose-built for exactly
this job — extracting clean text from Wikipedia dumps.

Key advantages:
- Written specifically for Wikipedia — knows all the edge cases
- Runs on **multiple CPU cores at once** (we used all 10 cores)
- ~100x faster than our manual approach

---

## The Two Phases Wikiextractor Goes Through

### Phase A — Template Preprocessing (~1 hour)
Before it can clean articles, it needs to understand all the Wikipedia
"templates" — reusable chunks like `{{cite book}}`, `{{infobox person}}`, etc.

It does one full scan of the 25GB file just to collect all these template
definitions. Single-threaded, slow, but necessary. Like reading the glossary
before reading the book.

### Phase B — Actual Extraction (~2 hours)
Now it goes through the file again, this time on all 10 CPU cores in parallel.
For each article it:
1. Strips all the wikitext markup (templates, infoboxes, tables, links, refs)
2. Keeps only clean plain text
3. Saves it as JSON: `{"id": "...", "title": "...", "text": "...", "url": "..."}`

---

## The Command We Ran

```bash
wikiextractor --json --processes 10 -o wiki_extracted/ enwiki-latest-pages-articles-multistream.xml.bz2
```

Breaking it down:
- `--json` → save output as JSON (one article per line), not raw text
- `--processes 10` → use all 10 CPU cores in parallel
- `-o wiki_extracted/` → save output into this folder
- the last part → the input file (our 25GB Wikipedia dump)

---

## What Came Out

The output is in `wiki_extracted/` folder:
- 201 sub-folders (AA, AB, AC ... BT, BU ...)
- Each folder has multiple files: `wiki_00`, `wiki_01`, `wiki_02` etc.
- Each file contains many JSON articles, one per line
- **Total: 20GB, 18.8 million articles**

Each article looks like:
```json
{"id": "12", "revid": "...", "url": "https://en.wikipedia.org/wiki/Anarchism", "title": "Anarchism", "text": "Anarchism is a political philosophy and movement..."}
```

---

## Why 18.8M Articles and Not 6.7M?

Wikipedia has ~6.7M "real" articles but the dump also contains:
- Redirect pages (e.g. "USA" → "United States") — millions of these
- Disambiguation pages ("Mercury" could mean the planet, element, or god)
- Very short stub articles (just 2-3 sentences)
- List pages, index pages, etc.

Wikiextractor extracted all of them. In Step 2 (chunking), we'll filter down
to only the real, substantial articles.

---

## Stats

| Thing | Value |
|---|---|
| Input file | 25GB (compressed) / ~100GB uncompressed |
| Tool used | wikiextractor v3.0.6 |
| CPU cores used | 10 |
| Total time | ~3 hours |
| Articles extracted | 18,883,645 |
| Output size | 20GB |
| Output format | JSON, one article per line |
| Output location | RAG/wiki_extracted/ |

---

## What Step 1 Did NOT Do

- Did NOT chunk articles into smaller pieces (that's Step 2)
- Did NOT create embeddings/vectors (that's Step 3)
- Did NOT build any search index (that's Step 4)
- Did NOT filter out redirects and stubs (that's Step 2)

It purely converted the raw Wikipedia dump into clean, readable JSON text.
Nothing more, nothing less.
