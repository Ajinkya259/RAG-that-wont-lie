"""
Phase 1: Parse raw Wikipedia XML dump → clean articles JSONL.

What this does:
  - Streams the bz2 XML without fully decompressing it (memory safe)
  - Extracts title, text, timestamp, url per article
  - Strips all wikitext markup (templates, infoboxes, tables, refs, etc.)
  - Skips redirects, disambiguation pages, namespace pages, stubs
  - Writes one clean article per line to data/articles.jsonl
  - Checkpoints progress every 100k articles so you can resume

Run: python3 pipeline/parse.py
"""

import bz2
import json
import hashlib
import re
import sys
import os
from pathlib import Path

import mwxml
import mwparserfromhell
from tqdm import tqdm

DUMP_PATH = "enwiki-latest-pages-articles-multistream.xml.bz2"
OUTPUT_PATH = "data/articles.jsonl"
CHECKPOINT_PATH = "data/parse_checkpoint.txt"
MIN_TEXT_LENGTH = 300  # skip stubs shorter than this


def clean_wikitext(raw_text: str) -> str:
    parsed = mwparserfromhell.parse(raw_text)

    # Remove all templates ({{infobox}}, {{cite}}, etc.)
    for template in parsed.filter_templates():
        try:
            parsed.remove(template)
        except Exception:
            pass

    # Get plain text, stripping wikilinks but keeping anchor text
    text = parsed.strip_code(
        normalize=True,
        collapse=True,
        keep_template_params=False,
    )

    # Remove leftover HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove citation markers like [1], [2]
    text = re.sub(r"\[\d+\]", "", text)
    # Remove external link markers
    text = re.sub(r"\[https?://\S+\s*(.*?)\]", r"\1", text)
    # Normalize section headers: == Heading == → \nHeading\n
    text = re.sub(r"={2,}\s*(.*?)\s*={2,}", r"\n\1\n", text)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def should_skip(title: str, text: str) -> bool:
    # Skip namespace pages (Talk:, Category:, Wikipedia:, File:, Template:, etc.)
    if ":" in title:
        return True
    # Skip redirects
    if text.strip().upper().startswith("#REDIRECT"):
        return True
    # Skip disambiguation pages
    if "(disambiguation)" in title.lower():
        return True
    return False


def get_resume_offset() -> int:
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            return int(f.read().strip())
    return 0


def save_checkpoint(count: int):
    with open(CHECKPOINT_PATH, "w") as f:
        f.write(str(count))


def main():
    Path("data").mkdir(exist_ok=True)

    resume_offset = get_resume_offset()
    mode = "a" if resume_offset > 0 else "w"

    if resume_offset > 0:
        print(f"Resuming from article #{resume_offset}")

    seen_hashes = set()
    written = resume_offset
    skipped = 0
    processed = 0

    with bz2.open(DUMP_PATH, "rb") as f:
        dump = mwxml.Dump.from_file(f)

        with open(OUTPUT_PATH, mode, encoding="utf-8") as out_file:
            pbar = tqdm(desc="Parsing articles", unit=" articles", dynamic_ncols=True)

            for page in dump.pages:
                processed += 1

                # Skip to resume point
                if processed <= resume_offset:
                    pbar.update(1)
                    continue

                try:
                    # Get latest revision only
                    revision = next(iter(page))
                    if revision.text is None:
                        skipped += 1
                        pbar.update(1)
                        continue

                    title = page.title or ""
                    raw_text = revision.text
                    timestamp = str(revision.timestamp) if revision.timestamp else ""

                    if should_skip(title, raw_text):
                        skipped += 1
                        pbar.update(1)
                        continue

                    clean_text = clean_wikitext(raw_text)

                    if len(clean_text) < MIN_TEXT_LENGTH:
                        skipped += 1
                        pbar.update(1)
                        continue

                    # Deduplicate by content hash
                    content_hash = hashlib.sha256(clean_text.encode()).hexdigest()
                    if content_hash in seen_hashes:
                        skipped += 1
                        pbar.update(1)
                        continue
                    seen_hashes.add(content_hash)

                    url = "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")

                    article = {
                        "id": page.id,
                        "title": title,
                        "text": clean_text,
                        "url": url,
                        "timestamp": timestamp,
                    }

                    out_file.write(json.dumps(article, ensure_ascii=False) + "\n")
                    written += 1

                    # Checkpoint every 100k articles
                    if written % 100_000 == 0:
                        save_checkpoint(processed)
                        out_file.flush()
                        pbar.set_postfix(written=written, skipped=skipped)

                except Exception as e:
                    skipped += 1

                pbar.update(1)

            pbar.close()

    # Clear checkpoint on successful completion
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)

    print(f"\nDone.")
    print(f"  Articles written : {written:,}")
    print(f"  Articles skipped : {skipped:,}")
    print(f"  Output           : {OUTPUT_PATH}")


if __name__ == "__main__":
    os.chdir(Path(__file__).parent.parent)  # run from RAG/ root
    main()
