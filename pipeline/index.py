"""
Phase 4: Build search indexes from the 750K embedded chunks.

Three parts:
  --faiss     : vector index (IndexFlatIP) for semantic search
  --bm25      : keyword index (bm25s) for lexical search
  --metadata  : SQLite store for chunk_id -> text/title/url lookup
  --all       : build all three

Run: python3 pipeline/index.py --all
"""

import os
import sys
import json
import glob
import time
import sqlite3
from pathlib import Path

import numpy as np

CHUNKS_PATH = "data/chunks.jsonl"
EMBED_DIR = "data/embeddings"
INDEX_DIR = "indexes"

FAISS_PATH = f"{INDEX_DIR}/faiss.index"
FAISS_IDS_PATH = f"{INDEX_DIR}/faiss_ids.json"
BM25_DIR = f"{INDEX_DIR}/bm25"
BM25_IDS_PATH = f"{INDEX_DIR}/bm25_ids.json"
METADATA_PATH = f"{INDEX_DIR}/chunk_metadata.db"


def get_embedded_ids() -> list:
    """Ordered list of chunk_ids that were embedded (from ids_*.json)."""
    id_files = sorted(glob.glob(f"{EMBED_DIR}/ids_*.json"))
    all_ids = []
    for fp in id_files:
        with open(fp) as f:
            all_ids.extend(json.load(f))
    return all_ids


def build_faiss():
    import faiss

    print("=== Building FAISS vector index ===")
    npy_files = sorted(glob.glob(f"{EMBED_DIR}/embeddings_*.npy"))
    print(f"Loading {len(npy_files)} embedding files...")

    arrays = [np.load(fp) for fp in npy_files]
    vectors = np.concatenate(arrays, axis=0).astype(np.float32)
    print(f"Loaded vectors: {vectors.shape}")

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    print(f"Added {index.ntotal:,} vectors to IndexFlatIP")

    faiss.write_index(index, FAISS_PATH)

    ids = get_embedded_ids()
    assert len(ids) == index.ntotal, f"ID count {len(ids)} != vector count {index.ntotal}"
    with open(FAISS_IDS_PATH, "w") as f:
        json.dump(ids, f)

    print(f"Saved: {FAISS_PATH} + {FAISS_IDS_PATH}")
    print(f"Vectors indexed: {index.ntotal:,}\n")


def build_bm25():
    import bm25s

    print("=== Building BM25 keyword index ===")
    embedded_ids = set(get_embedded_ids())
    n_target = len(embedded_ids)
    print(f"Target chunks: {n_target:,}")

    texts = []
    ids_order = []
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            if len(ids_order) >= n_target:
                break
            chunk = json.loads(line)
            if chunk["chunk_id"] in embedded_ids:
                texts.append(chunk["text"])
                ids_order.append(chunk["chunk_id"])

    print(f"Collected {len(texts):,} chunk texts")

    print("Tokenizing...")
    tokens = bm25s.tokenize(texts, show_progress=True)

    print("Indexing...")
    retriever = bm25s.BM25()
    retriever.index(tokens)

    retriever.save(BM25_DIR)
    with open(BM25_IDS_PATH, "w") as f:
        json.dump(ids_order, f)

    print(f"Saved: {BM25_DIR}/ + {BM25_IDS_PATH}")
    print(f"Documents indexed: {len(texts):,}\n")


def build_metadata():
    print("=== Building SQLite metadata store ===")
    embedded_ids = set(get_embedded_ids())
    n_target = len(embedded_ids)

    if os.path.exists(METADATA_PATH):
        os.remove(METADATA_PATH)

    con = sqlite3.connect(METADATA_PATH)
    con.execute("""
        CREATE TABLE chunks (
            chunk_id      TEXT PRIMARY KEY,
            article_title TEXT,
            article_url   TEXT,
            timestamp     TEXT,
            chunk_index   INTEGER,
            total_chunks  INTEGER,
            text          TEXT
        )
    """)

    batch = []
    inserted = 0
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            if inserted + len(batch) >= n_target:
                break
            chunk = json.loads(line)
            if chunk["chunk_id"] not in embedded_ids:
                continue
            batch.append((
                chunk["chunk_id"],
                chunk["article_title"],
                chunk["article_url"],
                chunk.get("timestamp", ""),
                chunk["chunk_index"],
                chunk["total_chunks"],
                chunk["text"],
            ))
            if len(batch) >= 10_000:
                con.executemany("INSERT OR IGNORE INTO chunks VALUES (?,?,?,?,?,?,?)", batch)
                inserted += len(batch)
                batch = []

    if batch:
        con.executemany("INSERT OR IGNORE INTO chunks VALUES (?,?,?,?,?,?,?)", batch)
        inserted += len(batch)

    con.commit()
    count = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    con.close()

    print(f"Saved: {METADATA_PATH}")
    print(f"Rows inserted: {count:,}\n")


def main():
    Path(INDEX_DIR).mkdir(exist_ok=True)
    args = sys.argv[1:]
    if not args:
        args = ["--all"]

    t0 = time.time()
    if "--all" in args or "--faiss" in args:
        build_faiss()
    if "--all" in args or "--bm25" in args:
        build_bm25()
    if "--all" in args or "--metadata" in args:
        build_metadata()

    print(f"Total time: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    os.chdir(Path(__file__).parent.parent)
    main()
