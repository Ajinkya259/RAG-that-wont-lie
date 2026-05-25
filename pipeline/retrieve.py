"""
Phase 5: Retrieval — question in, ranked relevant chunks out.

Pipeline:
  1. embed query (bge-small-en-v1.5 with query prefix)
  2. FAISS search (meaning) -> top 100
  3. BM25 search (keywords) -> top 100
  4. RRF merge -> top 50 candidates
  5. cross-encoder rerank -> top K
  6. confidence scoring
  7. return chunks with metadata

Usage:
  python3 pipeline/retrieve.py "Who developed the theory of relativity?"
"""

import os
# faiss-cpu and torch each bundle their own OpenMP runtime (libomp). Loaded in
# one process with multiple threads, the two runtimes collide on a thread barrier
# and segfault (confirmed via crash report: __kmp_suspend in libomp.dylib).
# Tolerate the duplicate runtime and force single-threaded OpenMP to avoid it.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
import json
import time
import math
import sqlite3
from pathlib import Path

import numpy as np
import faiss
faiss.omp_set_num_threads(1)   # keep faiss single-threaded (OpenMP conflict guard)
import bm25s
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder

INDEX_DIR = "indexes"
FAISS_PATH = f"{INDEX_DIR}/faiss.index"
FAISS_IDS_PATH = f"{INDEX_DIR}/faiss_ids.json"
BM25_DIR = f"{INDEX_DIR}/bm25"
BM25_IDS_PATH = f"{INDEX_DIR}/bm25_ids.json"
METADATA_PATH = f"{INDEX_DIR}/chunk_metadata.db"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Small models run fine (and stably) on CPU; keeps MPS free for a downstream LLM.
DEVICE = os.environ.get("RETRIEVER_DEVICE", "cpu")

RETRIEVE_N = 100      # candidates from each retriever
RRF_KEEP = 50         # candidates kept after RRF for reranking
RRF_K = 60            # RRF constant
CONFIDENCE_THRESHOLD = 0.4


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class Retriever:
    def __init__(self):
        print("Loading indexes and models (one-time)...")
        t0 = time.time()

        self.faiss_index = faiss.read_index(FAISS_PATH)
        with open(FAISS_IDS_PATH) as f:
            self.faiss_ids = json.load(f)

        self.bm25 = bm25s.BM25.load(BM25_DIR)
        with open(BM25_IDS_PATH) as f:
            self.bm25_ids = json.load(f)

        self.db = sqlite3.connect(METADATA_PATH)

        self.embed_model = SentenceTransformer(EMBED_MODEL, device=DEVICE)
        self.reranker = CrossEncoder(RERANK_MODEL, device=DEVICE)

        print(f"Ready in {time.time() - t0:.1f}s "
              f"({self.faiss_index.ntotal:,} chunks, device={DEVICE})\n")

    def _faiss_search(self, query: str):
        q = self.embed_model.encode(
            [QUERY_PREFIX + query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")
        D, I = self.faiss_index.search(q, RETRIEVE_N)
        return [(self.faiss_ids[row], float(score))
                for row, score in zip(I[0], D[0]) if row != -1]

    def _bm25_search(self, query: str):
        q_tokens = bm25s.tokenize([query], show_progress=False)
        results, scores = self.bm25.retrieve(
            q_tokens, k=RETRIEVE_N, show_progress=False
        )
        return [(self.bm25_ids[int(idx)], float(sc))
                for idx, sc in zip(results[0], scores[0])]

    @staticmethod
    def _rrf(faiss_hits, bm25_hits, k=RRF_K):
        scores, found_by = {}, {}
        for rank, (cid, _) in enumerate(faiss_hits, 1):
            scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank)
            found_by.setdefault(cid, set()).add("faiss")
        for rank, (cid, _) in enumerate(bm25_hits, 1):
            scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank)
            found_by.setdefault(cid, set()).add("bm25")
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return ranked, found_by

    def _get_meta(self, chunk_id: str):
        row = self.db.execute(
            "SELECT article_title, article_url, timestamp, text "
            "FROM chunks WHERE chunk_id=?", (chunk_id,)
        ).fetchone()
        if not row:
            return None
        return {"title": row[0], "url": row[1], "timestamp": row[2], "text": row[3]}

    def search(self, query: str, top_k: int = 5) -> dict:
        t0 = time.time()

        faiss_hits = self._faiss_search(query)
        bm25_hits = self._bm25_search(query)

        ranked, found_by = self._rrf(faiss_hits, bm25_hits)
        candidates = ranked[:RRF_KEEP]

        # Fetch text for reranking
        cand_meta = []
        for cid, _ in candidates:
            meta = self._get_meta(cid)
            if meta:
                cand_meta.append((cid, meta))

        # Rerank
        pairs = [(query, m["text"]) for _, m in cand_meta]
        rel_scores = self.reranker.predict(pairs) if pairs else []

        scored = []
        for (cid, meta), rel in zip(cand_meta, rel_scores):
            consistency = 1.0 if len(found_by[cid]) == 2 else 0.5
            freshness = 0.5  # neutral — no timestamps available
            confidence = (
                0.6 * sigmoid(float(rel))
                + 0.2 * consistency
                + 0.2 * freshness
            )
            scored.append({
                "chunk_id": cid,
                "title": meta["title"],
                "url": meta["url"],
                "text": meta["text"],
                "confidence": round(confidence, 4),
                "rerank_score": round(float(rel), 4),
                "found_by": sorted(found_by[cid]),
            })

        scored.sort(key=lambda x: -x["rerank_score"])
        top = scored[:top_k]

        low_conf = (not top) or (top[0]["confidence"] < CONFIDENCE_THRESHOLD)

        return {
            "query": query,
            "results": top,
            "low_confidence": low_conf,
            "latency_ms": int((time.time() - t0) * 1000),
        }


def main():
    if len(sys.argv) < 2:
        print('Usage: python3 pipeline/retrieve.py "your question here"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    r = Retriever()
    out = r.search(query)

    print(f"Query: {out['query']}")
    print(f"Latency: {out['latency_ms']}ms | low_confidence={out['low_confidence']}\n")
    for i, res in enumerate(out["results"], 1):
        snippet = res["text"][:200].replace("\n", " ")
        print(f"[{i}] {res['title']}  (conf={res['confidence']}, "
              f"rerank={res['rerank_score']}, found_by={res['found_by']})")
        print(f"    {snippet}...")
        print(f"    {res['url']}\n")


if __name__ == "__main__":
    os.chdir(Path(__file__).parent.parent)
    main()
