"""
Layer 08: Continuous evals.

Our corpus is the first 750K chunks of Wikipedia, so most public NQ/TriviaQA gold
passages are out of corpus and would give misleading recall numbers. Instead we
run a small curated, honest eval set in three buckets and measure the behaviour
that actually matters for a "won't lie" system:

  - answerable           : fact is in the corpus  -> expect an ANSWER + a source
  - unanswerable_on_topic: real entity, fact absent -> expect a REFUSAL
  - adversarial          : gibberish               -> expect a REFUSAL

Key metric: refusal rate on the (unanswerable + adversarial) set. That is the
anti-hallucination score. Also: answer rate + citation rate on answerable, and
latency. Results saved to eval/results.jsonl.

Run: python3 pipeline/evaluate.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RAG_CACHE", "0")  # eval fresh, don't serve cached answers

import json
import time
import sys
from pathlib import Path

sys.path.insert(0, "pipeline")
from retrieve import Retriever
from generate import Generator

RESULTS_PATH = "eval/results.jsonl"

EVAL_SET = [
    # answerable — fact is in the corpus
    ("answerable", "Who developed the theory of relativity?"),
    ("answerable", "What is photosynthesis?"),
    ("answerable", "What is anarchism?"),
    ("answerable", "What is the capital of France?"),
    ("answerable", "What is inertia in physics?"),
    ("answerable", "What is albedo?"),
    # unanswerable on topic — real entity, the specific fact is not in the chunks
    ("unanswerable_on_topic", "What did Albert Einstein eat for breakfast?"),
    ("unanswerable_on_topic", "What was Einstein's favourite colour?"),
    ("unanswerable_on_topic", "How many push-ups could Isaac Newton do?"),
    # adversarial — gibberish, should never get an answer
    ("adversarial", "asdfghjkl qwerty zxcvbnm nonsense"),
    ("adversarial", "florble wuzzle gnplax tttt"),
    ("adversarial", "?????? ?? ???"),
]


def main():
    os.chdir(Path(__file__).parent.parent)
    Path("eval").mkdir(exist_ok=True)

    retriever = Retriever()
    gen = Generator(retriever)

    rows = []
    t_start = time.time()
    for bucket, q in EVAL_SET:
        out = gen.answer(q)
        answered = not out["fallback"]
        has_src = len(out["sources"]) > 0
        # did it behave as the bucket expects?
        if bucket == "answerable":
            correct = answered and has_src
        else:  # should refuse
            correct = (not answered)
        row = {
            "bucket": bucket, "query": q,
            "answered": answered, "refused": not answered,
            "has_sources": has_src,
            "expected_ok": correct,
            "retrieval_ms": out["retrieval_latency_ms"],
            "generation_ms": out["generation_latency_ms"],
            "answer": out["answer"][:160],
        }
        rows.append(row)
        mark = "ok " if correct else "BAD"
        print(f"  [{mark}] ({bucket}) {q[:48]:<48}  "
              f"{'answer' if answered else 'refuse'}")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # summary
    ans = [r for r in rows if r["bucket"] == "answerable"]
    refuse_set = [r for r in rows if r["bucket"] != "answerable"]
    answer_rate = sum(r["answered"] for r in ans) / len(ans)
    cite_rate = sum(r["has_sources"] for r in ans if r["answered"]) / max(1, sum(r["answered"] for r in ans))
    refusal_rate = sum(r["refused"] for r in refuse_set) / len(refuse_set)
    overall = sum(r["expected_ok"] for r in rows) / len(rows)
    avg_ret = sum(r["retrieval_ms"] for r in rows) / len(rows)
    avg_gen = sum(r["generation_ms"] for r in rows if r["generation_ms"]) / max(1, sum(1 for r in rows if r["generation_ms"]))

    print("\n" + "=" * 52)
    print("  EVAL SUMMARY")
    print("=" * 52)
    print(f"  answerable: answer rate     {answer_rate*100:5.1f}%  ({len(ans)} q)")
    print(f"  answerable: citation rate   {cite_rate*100:5.1f}%")
    print(f"  refusal rate (unans+adv)    {refusal_rate*100:5.1f}%  ({len(refuse_set)} q)  <- anti-hallucination")
    print(f"  overall behaved as expected {overall*100:5.1f}%")
    print(f"  avg retrieval               {avg_ret:5.0f} ms")
    print(f"  avg generation              {avg_gen:5.0f} ms")
    print(f"  total eval time             {time.time()-t_start:5.1f} s")
    print(f"  results -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
