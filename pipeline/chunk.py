"""
Phase 2: Filter + chunk wiki_extracted/ articles → data/chunks.jsonl

What this does:
  - Walks all 201 folders in wiki_extracted/
  - Filters out empty pages, stubs, redirects, list/disambiguation pages
  - Splits each article into ~384-token chunks with 50-token overlap
  - Attaches metadata: chunk_id, article_id, title, url, chunk_index, total_chunks
  - Writes one chunk per line to data/chunks.jsonl
  - Uses multiprocessing (8 workers) for speed
  - Checkpoints progress so it can resume if crashed

Run: python3 pipeline/chunk.py
"""

import json
import os
import glob
import multiprocessing as mp
from pathlib import Path

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

WIKI_DIR = "wiki_extracted"
OUTPUT_PATH = "data/chunks.jsonl"
CHECKPOINT_PATH = "data/chunk_checkpoint.json"
NUM_WORKERS = 8
CHUNK_SIZE = 384      # tokens
CHUNK_OVERLAP = 50    # tokens
MIN_CHUNK_TOKENS = 100
MIN_ARTICLE_CHARS = 300

# Titles to skip
BAD_TITLE_PREFIXES = ("list of", "lists of", "index of")
BAD_TITLE_SUBSTRINGS = ("(disambiguation)", ":")


def should_skip(article: dict) -> bool:
    text = article.get("text", "")
    title = article.get("title", "").lower()

    if not text or len(text) < MIN_ARTICLE_CHARS:
        return True
    if any(title.startswith(p) for p in BAD_TITLE_PREFIXES):
        return True
    if any(s in title for s in BAD_TITLE_SUBSTRINGS):
        return True
    return False


def count_tokens(text: str, enc) -> int:
    return len(enc.encode(text))


def process_file(filepath: str) -> list:
    enc = tiktoken.get_encoding("cl100k_base")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=lambda t: len(enc.encode(t)),
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks_out = []

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                article = json.loads(line)
            except json.JSONDecodeError:
                continue

            if should_skip(article):
                continue

            article_id = str(article.get("id", ""))
            title = article.get("title", "")
            url = article.get("url", "")
            text = article["text"]

            splits = splitter.split_text(text)

            # Filter out chunks that are too short
            valid_splits = [
                s for s in splits
                if count_tokens(s, enc) >= MIN_CHUNK_TOKENS
            ]

            total = len(valid_splits)
            for idx, chunk_text in enumerate(valid_splits):
                chunk = {
                    "chunk_id": f"{article_id}_{idx:04d}",
                    "article_id": article_id,
                    "article_title": title,
                    "article_url": url,
                    "timestamp": "",
                    "chunk_index": idx,
                    "total_chunks": total,
                    "text": chunk_text,
                }
                chunks_out.append(json.dumps(chunk, ensure_ascii=False))

    return chunks_out


def load_checkpoint() -> set:
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            return set(json.load(f))
    return set()


def save_checkpoint(done_files: set):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(list(done_files), f)


def main():
    Path("data").mkdir(exist_ok=True)

    # Collect all wiki files
    all_files = sorted(glob.glob(f"{WIKI_DIR}/**/*", recursive=True))
    all_files = [f for f in all_files if os.path.isfile(f)]
    print(f"Total wiki files found: {len(all_files):,}")

    done_files = load_checkpoint()
    remaining = [f for f in all_files if f not in done_files]
    print(f"Already done: {len(done_files):,} | Remaining: {len(remaining):,}")

    mode = "a" if done_files else "w"
    total_chunks = 0

    with open(OUTPUT_PATH, mode, encoding="utf-8") as out_file:
        with mp.Pool(NUM_WORKERS) as pool:
            with tqdm(total=len(remaining), desc="Chunking files", unit="file") as pbar:
                for filepath, chunks in zip(
                    remaining,
                    pool.imap(process_file, remaining, chunksize=4)
                ):
                    for chunk_line in chunks:
                        out_file.write(chunk_line + "\n")

                    total_chunks += len(chunks)
                    done_files.add(filepath)

                    # Checkpoint every 100 files
                    if len(done_files) % 100 == 0:
                        save_checkpoint(done_files)
                        out_file.flush()
                        pbar.set_postfix(chunks=f"{total_chunks:,}")

                    pbar.update(1)

    save_checkpoint(done_files)

    # Clean up checkpoint on full completion
    if len(done_files) == len(all_files):
        os.remove(CHECKPOINT_PATH)
        print("\nAll files processed — checkpoint cleared.")

    print(f"\nDone.")
    print(f"  Total chunks written : {total_chunks:,}")
    print(f"  Output               : {OUTPUT_PATH}")


if __name__ == "__main__":
    os.chdir(Path(__file__).parent.parent)
    main()
