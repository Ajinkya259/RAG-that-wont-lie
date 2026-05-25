# Step 6 — Generation Plan (Detailed)

## TL;DR (Plain English First)

Step 5 finds the most relevant chunks. But it hands you raw Wikipedia paragraphs,
not an answer. Step 6 takes those chunks, gives them to an LLM with strict
instructions — "answer ONLY from this text, cite your sources, and if the answer
isn't here, say so" — and produces a clean, cited answer.

**Step 6 = turn retrieved chunks into a trustworthy, cited answer (or honestly
refuse).**

This is the step that fights hallucination — the whole point from the original reel.

---

## The Core Idea: Grounded Generation

A normal LLM answers from its training memory — which can be wrong or made up
(hallucination). We don't want that. We want the LLM to answer **only** from the
chunks we retrieved, so every claim traces back to a real Wikipedia source.

Three guardrails enforce this:
1. **Strict prompt** — "use ONLY the context, no outside knowledge"
2. **Citations** — every answer names its source chunks
3. **Fallback** — if retrieval was weak, refuse to answer instead of guessing

---

## Decision Needed: Which LLM?

| Option | Pros | Cons |
|---|---|---|
| **A. Claude API** (`claude-sonnet-4-6`) | Best quality, fast, reliable | Needs `ANTHROPIC_API_KEY`, costs ~$ per query, sends data to API |
| **B. Local Qwen2.5-3B** (already cached) | Free, fully offline, private | Lower quality, slower on laptop, uses RAM/GPU |

The script will **support both** and auto-select: use Claude if `ANTHROPIC_API_KEY`
is set, otherwise fall back to local Qwen. You choose at runtime.

(My recommendation: Claude API for the demo if you have a key — cleaner output and
no extra load on the already-strained laptop. Qwen if you want it 100% offline/free.)

---

## The Generation Pipeline

### 6.1 — Get Retrieved Chunks
Reuse the `Retriever` from Step 5:
```python
retrieval = retriever.search(query, top_k=5)
```
Returns top-5 chunks with text, title, url, rerank_score, confidence.

### 6.2 — Hallucination Fallback Check (BEFORE calling the LLM)
Using the Step 5 calibration finding — key the decision off **rerank_score**, not
blended confidence:

```python
best_rerank = retrieval["results"][0]["rerank_score"] if results else -999
if best_rerank < RERANK_FLOOR:   # RERANK_FLOOR ≈ 0.5
    return {
        "answer": "I don't have enough reliable information to answer that.",
        "sources": [],
        "fallback": true
    }
```

This cleanly refuses gibberish (rerank −0.16) while answering real questions
(rerank +5 to +9). No LLM call wasted on bad retrievals.

### 6.3 — Build the Constrained Prompt

**System prompt:**
```
You are a factual assistant. Answer the question using ONLY the numbered context
passages provided. Follow these rules strictly:
- Use ONLY information in the context. Do not use any outside knowledge.
- If the context does not contain the answer, reply exactly: "Insufficient evidence."
- Cite the passages you used with their numbers, like [1] or [2].
- Be concise and factual.
```

**User message:**
```
Context:
[1] {chunk_1_text}
[2] {chunk_2_text}
[3] {chunk_3_text}
[4] {chunk_4_text}
[5] {chunk_5_text}

Question: {query}
```

### 6.4 — Call the LLM
```python
# Claude path
client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=512,
    temperature=0.0,          # deterministic, reduces hallucination
    system=SYSTEM_PROMPT,
    messages=[{"role": "user", "content": user_msg}],
)
```
- `temperature=0.0` → most deterministic, least creative/hallucinatory
- `max_tokens=512` → answers are short and factual

Local Qwen path: same prompt via the chat template, `do_sample=False`.

### 6.5 — Parse Citations & Attach Sources
- The LLM cites passages as `[1]`, `[2]`, etc.
- Map those numbers back to the actual chunks → real titles + URLs
- Only include sources the LLM actually cited

### 6.6 — Return Structure
```json
{
  "query": "Who developed the theory of relativity?",
  "answer": "The theory of relativity was developed by Albert Einstein, comprising special relativity (1905) and general relativity (1915). [1][4]",
  "sources": [
    {"n": 1, "title": "Theory of relativity", "url": "https://en.wikipedia.org/wiki?curid=30001"},
    {"n": 4, "title": "Albert Einstein", "url": "https://en.wikipedia.org/wiki?curid=736"}
  ],
  "fallback": false,
  "model": "claude-sonnet-4-6",
  "retrieval_latency_ms": 816,
  "generation_latency_ms": 1200
}
```

Note: the LLM may itself output "Insufficient evidence" even when retrieval passed
the rerank floor — if the chunks are topically near but don't actually answer the
question. That's the second layer of hallucination protection.

---

## Script: `pipeline/generate.py`

### Design
```python
class Generator:
    def __init__(self, retriever):
        self.retriever = retriever
        self.backend = "claude" if os.getenv("ANTHROPIC_API_KEY") else "qwen"
        # load the chosen LLM
    def answer(self, query) -> dict:
        # 6.1 retrieve → 6.2 fallback → 6.3 prompt → 6.4 LLM → 6.5 cite → 6.6 return
```

### CLI
```bash
python3 pipeline/generate.py "Who developed the theory of relativity?"
```
Prints the answer, then the cited sources, then timing.

---

## Two-Layer Hallucination Defense

| Layer | Catches | Mechanism |
|---|---|---|
| 1. Rerank floor (pre-LLM) | Gibberish, off-topic queries | `rerank_score < 0.5` → refuse, skip LLM |
| 2. Prompt instruction (in-LLM) | Topically-close-but-wrong chunks | LLM told to say "Insufficient evidence" |

This is exactly the "constrained generation + hallucination fallback" from the
original reel (steps 5 & 7).

---

## Test Plan

| Query | Expected |
|---|---|
| "Who developed the theory of relativity?" | Cited answer naming Einstein, sources [1]/[4] |
| "What is photosynthesis?" | Cited factual answer |
| "What is the capital of France?" | "Paris", with source |
| "asdfghjkl qwerty nonsense" | Fallback refusal (rerank floor) |
| "What did Einstein eat for breakfast?" | "Insufficient evidence" (in corpus but unanswerable) |

The last one is the key test — it's about a real topic (Einstein) so retrieval
succeeds, but the answer isn't in the chunks, so the LLM must refuse rather than
invent. That's the anti-hallucination proof.

---

## Config

```python
LLM_MODEL_CLAUDE = "claude-sonnet-4-6"
LLM_MODEL_LOCAL  = "Qwen/Qwen2.5-3B-Instruct"
MAX_TOKENS       = 512
TEMPERATURE      = 0.0
RERANK_FLOOR     = 0.5     # below this → refuse (fixes Step 5 calibration finding)
TOP_K_CONTEXT    = 5       # chunks fed to the LLM
```

---

## Dependencies

```bash
# Claude path:
pip3 install anthropic        # already installed (0.74.1)
export ANTHROPIC_API_KEY=...  # you provide

# Local path:
pip3 install transformers torch   # already installed
# Qwen2.5-3B-Instruct already cached locally
```

---

## What Step 6 Does NOT Do

- Does NOT do retrieval (reuses Step 5's `Retriever`)
- Does NOT log traces or cache results (Steps 7/8)
- Does NOT build a web UI (optional later)
- Just: question → retrieve → grounded, cited answer (or honest refusal).
