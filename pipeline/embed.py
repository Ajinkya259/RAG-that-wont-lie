"""
Phase 3: Embed chunks → float16 numpy arrays

What this does:
  - Loads chunks from data/chunks.jsonl (13.98M chunks)
  - Embeds in batches of 500K using BAAI/bge-small-en-v1.5
  - Saves embeddings as float16 .npy files (~384 MB each)
  - Saves matching chunk_ids as .json files
  - Checkpoints after every batch — safe to resume if crashed

Run: python3 pipeline/embed.py
"""

import os
# Set thread limits BEFORE importing torch to prevent CPU thread contention
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import time
import numpy as np
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

CHUNKS_PATH = "data/chunks.jsonl"
EMBED_DIR = "data/embeddings"
CHECKPOINT_PATH = "data/embed_checkpoint.json"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 64         # proven-safe setting
SAVE_EVERY = 50_000     # smaller batches — lighter memory, frequent checkpoints
MAX_CHUNKS = 1_000_000  # process only first 1M chunks (set to None for all)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_checkpoint() -> int:
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            data = json.load(f)
            return data.get("chunks_done", 0)
    return 0


def save_checkpoint(chunks_done: int, total: int):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump({"chunks_done": chunks_done, "total_chunks": total}, f)


def count_lines(path: str) -> int:
    with open(path) as f:
        return sum(1 for _ in f)


def main():
    Path(EMBED_DIR).mkdir(parents=True, exist_ok=True)

    print(f"Loading model on device: {DEVICE}")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    print(f"Model loaded: {MODEL_NAME}")

    print("Counting chunks...")
    file_total = count_lines(CHUNKS_PATH)
    total_chunks = min(file_total, MAX_CHUNKS) if MAX_CHUNKS else file_total
    print(f"Chunks in file: {file_total:,}")
    print(f"Will process  : {total_chunks:,}")

    resume_from = load_checkpoint()
    if resume_from > 0:
        print(f"Resuming from chunk #{resume_from:,}")

    batch_num = resume_from // SAVE_EVERY
    texts_buf = []
    ids_buf = []
    chunks_done = resume_from
    t_start = time.time()

    with open(CHUNKS_PATH, encoding="utf-8") as f:
        # Skip already-processed chunks
        for _ in range(resume_from):
            f.readline()

        pbar = tqdm(
            total=total_chunks - resume_from,
            desc="Embedding",
            unit=" chunks",
            dynamic_ncols=True,
        )

        for line in f:
            if chunks_done + len(texts_buf) >= total_chunks:
                break

            line = line.strip()
            if not line:
                continue

            chunk = json.loads(line)
            texts_buf.append(chunk["text"])
            ids_buf.append(chunk["chunk_id"])

            # When buffer hits SAVE_EVERY, embed + save
            if len(texts_buf) >= SAVE_EVERY:
                _embed_and_save(model, texts_buf, ids_buf, batch_num)
                chunks_done += len(texts_buf)
                save_checkpoint(chunks_done, total_chunks)

                elapsed = time.time() - t_start
                rate = chunks_done / elapsed
                remaining = (total_chunks - chunks_done) / rate / 3600
                pbar.set_postfix(
                    batch=batch_num,
                    done=f"{chunks_done:,}",
                    eta=f"{remaining:.1f}h"
                )

                texts_buf = []
                ids_buf = []
                batch_num += 1

            pbar.update(1)

        # Final partial batch
        if texts_buf:
            _embed_and_save(model, texts_buf, ids_buf, batch_num)
            chunks_done += len(texts_buf)
            save_checkpoint(chunks_done, total_chunks)

        pbar.close()

    # Remove checkpoint on full completion
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)

    total_time = (time.time() - t_start) / 3600
    print(f"\nDone.")
    print(f"  Chunks embedded : {chunks_done:,}")
    print(f"  Batches saved   : {batch_num + 1}")
    print(f"  Total time      : {total_time:.2f} hrs")
    print(f"  Output dir      : {EMBED_DIR}/")


def _embed_and_save(model, texts: list, ids: list, batch_num: int):
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    # Save as float16 to halve disk usage
    embeddings = embeddings.astype(np.float16)

    npy_path = f"{EMBED_DIR}/embeddings_{batch_num:02d}.npy"
    ids_path = f"{EMBED_DIR}/ids_{batch_num:02d}.json"

    np.save(npy_path, embeddings)
    with open(ids_path, "w") as f:
        json.dump(ids, f)

    print(f"\n  Saved batch {batch_num:02d}: {len(texts):,} chunks → {npy_path} ({embeddings.nbytes / 1e6:.0f} MB)")


if __name__ == "__main__":
    os.chdir(Path(__file__).parent.parent)
    main()
