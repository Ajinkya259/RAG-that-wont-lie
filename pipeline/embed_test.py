"""
SAFE sanity test — embeds only N chunks with conservative settings.
Use this BEFORE running the full embed.py.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import time
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

N_TEST = 100         # ← only 100 chunks
BATCH_SIZE = 64      # ← small batch

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {DEVICE}")

print("Loading model...")
t = time.time()
model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=DEVICE)
print(f"Model loaded in {time.time() - t:.1f}s")

print(f"Reading {N_TEST} chunks...")
texts, ids = [], []
with open("data/chunks.jsonl") as f:
    for i, line in enumerate(f):
        if i >= N_TEST: break
        c = json.loads(line)
        texts.append(c["text"])
        ids.append(c["chunk_id"])

print(f"Encoding (batch={BATCH_SIZE})...")
t = time.time()
emb = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=False,
    convert_to_numpy=True,
    normalize_embeddings=True,
)
elapsed = time.time() - t

emb16 = emb.astype(np.float16)
rate = N_TEST / elapsed
print(f"\nShape         : {emb16.shape}")
print(f"Dtype         : {emb16.dtype}")
print(f"Self-sim [0]  : {np.dot(emb[0], emb[0]):.4f}  (should be ~1.0)")
print(f"Pair-sim 0,50 : {np.dot(emb[0], emb[50]):.4f}")
print(f"Time          : {elapsed:.1f}s")
print(f"Rate          : {rate:.0f} chunks/sec")
print(f"Full 14M ETA  : {14_000_000 / rate / 3600:.1f} hrs")
