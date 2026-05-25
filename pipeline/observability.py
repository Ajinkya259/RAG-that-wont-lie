"""
Layer 10: Observability everywhere.

Every answered query writes one structured trace line to logs/traces.jsonl:
the retrieval path (BM25 top ids, FAISS top ids, RRF order), the reranked
top-5 with scores, the confidence decision, the final answer, cache status,
and stage latencies. So any answer can be opened up and explained after the
fact: why these chunks, in this order, with this verdict.

Usage:
    from observability import TraceLogger
    tl = TraceLogger()
    tl.log(trace_dict)

View recent traces:
    python3 pipeline/observability.py --tail 5
"""

import os
import sys
import json
import time
import uuid
from pathlib import Path

TRACE_PATH = "logs/traces.jsonl"


class TraceLogger:
    def __init__(self, path: str = TRACE_PATH):
        Path("logs").mkdir(exist_ok=True)
        self.path = path

    def new_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def log(self, trace: dict):
        trace.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S"))
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")


def tail(n: int = 5):
    if not os.path.exists(TRACE_PATH):
        print("no traces yet")
        return
    lines = open(TRACE_PATH, encoding="utf-8").read().splitlines()
    for line in lines[-n:]:
        t = json.loads(line)
        print("=" * 64)
        print(f"[{t.get('ts','')}] query_id={t.get('query_id','')}")
        print(f"  query        : {t.get('query','')}")
        print(f"  cache        : {t.get('cache','')}")
        print(f"  bm25 top     : {t.get('bm25_top', [])[:5]}")
        print(f"  faiss top    : {t.get('faiss_top', [])[:5]}")
        print(f"  rrf top      : {t.get('rrf_top', [])[:5]}")
        print(f"  reranked top : {t.get('reranked_top', [])}")
        print(f"  confidences  : {t.get('confidences', [])}")
        print(f"  best rerank  : {t.get('best_rerank','')}")
        print(f"  fallback     : {t.get('fallback','')}")
        print(f"  latency ms   : retrieval={t.get('retrieval_ms','')} generation={t.get('generation_ms','')}")
        ans = (t.get("answer", "") or "")[:120]
        print(f"  answer       : {ans}")


if __name__ == "__main__":
    os.chdir(Path(__file__).parent.parent)
    n = 5
    if "--tail" in sys.argv:
        try:
            n = int(sys.argv[sys.argv.index("--tail") + 1])
        except Exception:
            pass
    tail(n)
