# Step 6 — Generation (What We Did + The Big Debugging Saga)

## TL;DR (Plain English)

Step 5 found the relevant chunks. Step 6 takes those chunks, hands them to a
local AI model (Qwen) with strict instructions — "answer ONLY from this text,
cite your sources, and if the answer isn't here, say so" — and produces a clean,
cited answer. Or it honestly refuses.

**Step 6 = turn retrieved chunks into a trustworthy, cited answer (or honest
refusal).**

This is the anti-hallucination finale from the original reel. And it works:

> **Q:** Who developed the theory of relativity?
> **A:** Albert Einstein developed the theory of relativity. [1]
> **Source:** [1] Theory of relativity

---

## The LLM: Local Qwen2.5 (Offline, Free, Private)

We chose a **local** model over the Claude API — fully offline, no API key, no
cost, data never leaves the machine.

- Final model: `Qwen/Qwen2.5-1.5B-Instruct` (float32, CPU)
- Was originally 3B, dropped to 1.5B for stability + lower RAM (see saga below)
- `temperature=0` (greedy decoding) → deterministic, least hallucination
- `max_new_tokens=512`

---

## Two-Layer Anti-Hallucination Defense

| Layer | When | Catches | How |
|---|---|---|---|
| 1. Rerank floor | BEFORE the LLM | gibberish, off-topic, weak retrievals | if top rerank_score < 0.5 → refuse, skip LLM entirely |
| 2. Prompt instruction | INSIDE the LLM | topically-close-but-wrong chunks | LLM told to reply "Insufficient evidence" |

This is exactly steps 5 & 7 from the reel: constrained generation + hallucination
fallback.

---

## The Pipeline

1. **Retrieve** top-5 chunks (reuses Step 5's `Retriever`)
2. **Fallback check** — if best rerank_score < 0.5, return refusal immediately
   (no LLM call wasted on bad retrievals)
3. **Build strict prompt** — system rules + numbered context + the question
4. **Generate** with Qwen (temp 0)
5. **Parse `[n]` citations** → map back to real chunk titles + URLs
6. **Return** answer + sources + fallback flag + timing

---

## Verified Test Results

| Query | Answer | Behavior |
|---|---|---|
| Who developed the theory of relativity? | "Albert Einstein developed the theory of relativity. [1]" | ✓ answered + cited |
| What is photosynthesis? | Full factual answer (light → glucose + oxygen) | ✓ answered, grounded |
| asdfghjkl qwerty nonsense | "I don't have enough reliable information to answer that." | ✓ refused (rerank floor) |
| What did Einstein eat for breakfast? | "I don't have enough reliable information to answer that." | ✓ **refused — did NOT hallucinate** |

The breakfast test is the crucial proof: Einstein IS in our corpus, so retrieval
finds Einstein chunks — but none mention breakfast. Instead of inventing an
answer (what a normal LLM would do), the system refused. That's the whole point.

---

## THE BIG DEBUGGING SAGA: "Python quit unexpectedly"

This step took many tries because of a nasty crash. Here's the full story, because
the wrong guesses are as instructive as the fix.

### The Symptom
Every multi-query run crashed with macOS popup **"Python quit unexpectedly"** and
exit code **139 (segmentation fault)**. Sometimes a single query survived, which
made it look random.

### Wrong Guess #1 — "It's MPS (the GPU)"
We blamed Apple's MPS GPU backend (it HAD caused the Step 3 embedding crashes).
Moved the small models to CPU, kept the LLM on MPS. **Still crashed.**

### Wrong Guess #2 — "It's float16 on CPU"
We figured CPUs can't do float16 math (which is true and can crash). Switched to
float32 + dropped to Qwen-1.5B. **Still crashed.**

### The Right Move — Read the Actual Crash Report
Instead of guessing, we parsed the macOS crash report
(`~/Library/Logs/DiagnosticReports/Python*.ips`). It named the EXACT faulting
library and function:

```
EXC_BAD_ACCESS / SIGSEGV in libomp.dylib
  __kmp_suspend_64 → __kmp_fork_barrier → __kmp_launch_worker
```

### The Real Root Cause
**OpenMP runtime conflict.** Both `faiss-cpu` and `torch` bundle their OWN copy
of the OpenMP threading library (`libomp.dylib`). When loaded in the same Python
process and both spawn parallel worker threads, the two runtimes collide on a
thread barrier → segfault.

This explains the "sometimes it works" randomness — it's a **thread race**, not a
deterministic bug. Single queries occasionally finished before the threads
collided; multiple operations reliably tripped it.

It was NEVER about MPS or float16. Those were red herrings (though float16-on-CPU
IS a real issue we'd have hit later anyway).

### The Fix
```python
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # tolerate duplicate OpenMP runtime
os.environ["OMP_NUM_THREADS"] = "1"           # force single-threaded OpenMP
os.environ["MKL_NUM_THREADS"] = "1"
faiss.omp_set_num_threads(1)                  # keep faiss single-threaded
```
Set in `retrieve.py`, `generate.py`, and `test_e2e.py` — before any heavy import.

Result: **exit code 0, no crash, no popup.** All four test queries ran clean.

### Lesson
When you get a native segfault, **read the crash report** before guessing. The
`.ips` file names the faulting dylib and function — it pointed straight at libomp
and saved us from more wrong guesses. faiss + torch in one process on macOS is a
known OpenMP-conflict trap.

---

## The Script: `pipeline/generate.py`

```python
class Generator:
    def __init__(self, retriever):   # loads Qwen once
    def answer(self, query) -> dict: # retrieve → fallback → prompt → LLM → cite
```

CLI:
```bash
LLM_DEVICE=cpu RETRIEVER_DEVICE=cpu python3 pipeline/generate.py "your question"
```

Env knobs:
- `LLM_DEVICE` (default `cpu`) — set `mps` to try GPU (unstable on this machine)
- `LLM_MODEL` (default Qwen2.5-1.5B-Instruct)
- `RETRIEVER_DEVICE` (default `cpu`)

---

## Citations: How We Made Them Robust

Qwen-1.5B often doesn't emit `[n]` citation markers even when its answer is fully
grounded — small models follow formatting instructions less reliably than larger
ones. We strengthened the prompt (explicit citation example), which helps but
isn't 100%.

The robust fix: a **two-tier citation system**:
1. **Explicit citations** — if the model emits `[1]`, `[2]`, we map those exact
   numbers to their chunk titles/URLs (precise, `cited: true`).
2. **Sources consulted** (fallback) — if the model gives a real answer but no
   markers, we attach the top chunks it was given as context (`cited: false`).
   This is honest: the answer is *constrained* to only those chunks, so they ARE
   the sources — we just can't pinpoint which specific one.

Result: **every answer is traceable to real Wikipedia sources**, regardless of
the small model's formatting compliance. For pinpoint per-sentence citations,
use a 3B+ model or the Claude API path.

---

## Config

```python
LLM_MODEL      = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 512
TEMPERATURE    = 0.0          # greedy / deterministic
RERANK_FLOOR   = 0.5          # below this → refuse before LLM
TOP_K_CONTEXT  = 5
DEVICE         = "cpu"        # float32; mps was unstable
```

---

## Performance

| Phase | Time |
|---|---|
| Load indexes + retriever (CPU) | ~10s |
| Load Qwen-1.5B (CPU, float32) | ~15s |
| Per answer (generation) | ~10–15s on CPU |
| Refusal (rerank floor, no LLM) | instant (0ms gen) |

---

## What Step 6 Does NOT Do

- Does NOT do retrieval (reuses Step 5)
- Does NOT log traces or cache (Steps 7/8, optional)
- Does NOT have a web UI (optional later)
- Just: question → retrieve → grounded cited answer, or honest refusal.

---

## STATUS: The RAG system is end-to-end functional.

Question in → hybrid retrieval → rerank → confidence gate → grounded generation
→ cited answer (or honest "insufficient evidence"). The core build (the reel's
10 steps) is working on 750K Wikipedia chunks, fully offline.
