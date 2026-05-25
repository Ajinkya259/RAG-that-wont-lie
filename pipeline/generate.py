"""
Phase 6: Generation — grounded, cited answers using local Qwen2.5-3B-Instruct.

Pipeline:
  1. retrieve top-5 chunks (reuses Step 5 Retriever)
  2. hallucination fallback: if best rerank < RERANK_FLOOR -> refuse, skip LLM
  3. build strict context-only prompt
  4. generate with Qwen2.5-3B (temperature 0)
  5. parse [n] citations -> attach real titles/urls
  6. return grounded answer + sources

Usage:
  python3 pipeline/generate.py "Who developed the theory of relativity?"
"""

import os
# See retrieve.py: faiss + torch OpenMP runtime conflict causes a libomp segfault.
# Tolerate the duplicate runtime and force single-threaded OpenMP.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
import re
import time
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from retrieve import Retriever
except ModuleNotFoundError:
    from pipeline.retrieve import Retriever

LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
MAX_NEW_TOKENS = 512
RERANK_FLOOR = 0.5      # below this -> refuse before calling LLM
TOP_K_CONTEXT = 5

# MPS proved unstable for the 3B model on this machine (segfaults on load and on
# repeated generations). CPU is slower per answer but rock-solid. Override with
# LLM_DEVICE=mps if you want to try the GPU.
DEVICE = os.environ.get("LLM_DEVICE", "cpu")

SYSTEM_PROMPT = (
    "You are a factual assistant. Answer the question using ONLY the numbered "
    "context passages provided. Follow these rules strictly:\n"
    "- Use ONLY information in the context. Do not use any outside knowledge.\n"
    "- If the context does not contain the answer, reply exactly: "
    '"Insufficient evidence."\n'
    "- You MUST cite the passage number in square brackets immediately after each "
    "fact you state, e.g. \"The sky is blue [2].\" Every sentence that uses the "
    "context must end with at least one citation like [1] or [3].\n"
    "- Be concise and factual.\n\n"
    "Example:\n"
    "Context:\n"
    "[1] The Eiffel Tower is located in Paris.\n"
    "[2] It was completed in 1889.\n"
    "Question: Where and when was the Eiffel Tower built?\n"
    "Answer: The Eiffel Tower is in Paris [1], completed in 1889 [2]."
)

FALLBACK_MSG = "I don't have enough reliable information to answer that."


class Generator:
    def __init__(self, retriever: Retriever):
        self.retriever = retriever
        print(f"Loading LLM {LLM_MODEL} on {DEVICE}...")
        t0 = time.time()
        # float16 is unsupported on CPU (segfaults); use float32 there.
        dtype = torch.float16 if DEVICE == "mps" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL,
            dtype=dtype,
            device_map=DEVICE,
        )
        self.model.eval()
        print(f"LLM ready in {time.time() - t0:.1f}s\n")

    def _build_prompt(self, query: str, chunks: list) -> str:
        context_lines = []
        for i, c in enumerate(chunks, 1):
            context_lines.append(f"[{i}] {c['text']}")
        context = "\n\n".join(context_lines)
        return f"Context:\n{context}\n\nQuestion: {query}"

    def _generate(self, user_msg: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,            # greedy = deterministic (temp 0)
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()

    def answer(self, query: str) -> dict:
        t0 = time.time()
        retrieval = self.retriever.search(query, top_k=TOP_K_CONTEXT)
        chunks = retrieval["results"]
        ret_ms = retrieval["latency_ms"]

        # Layer 1: rerank-floor fallback (skip LLM for weak retrievals)
        best_rerank = chunks[0]["rerank_score"] if chunks else -999
        if best_rerank < RERANK_FLOOR:
            return {
                "query": query,
                "answer": FALLBACK_MSG,
                "sources": [],
                "fallback": True,
                "model": LLM_MODEL,
                "retrieval_latency_ms": ret_ms,
                "generation_latency_ms": 0,
            }

        # Generate
        user_msg = self._build_prompt(query, chunks)
        g0 = time.time()
        answer_text = self._generate(user_msg)
        gen_ms = int((time.time() - g0) * 1000)

        # Layer 2: LLM self-refusal
        is_refusal = "insufficient evidence" in answer_text.lower()

        # Parse [n] citations -> map to real sources
        cited_nums = sorted(set(int(n) for n in re.findall(r"\[(\d+)\]", answer_text)))
        sources = []
        for n in cited_nums:
            if 1 <= n <= len(chunks):
                c = chunks[n - 1]
                sources.append({"n": n, "title": c["title"], "url": c["url"], "cited": True})

        # Fallback: model gave a real answer but emitted no [n] markers.
        # The answer is constrained to the retrieved context, so attach the top
        # chunks it was given as "sources consulted" — guarantees traceability.
        if not sources and not is_refusal:
            for n, c in enumerate(chunks[:3], 1):
                sources.append({"n": n, "title": c["title"], "url": c["url"], "cited": False})

        return {
            "query": query,
            "answer": answer_text,
            "sources": sources,
            "fallback": is_refusal,
            "model": LLM_MODEL,
            "retrieval_latency_ms": ret_ms,
            "generation_latency_ms": gen_ms,
        }


def main():
    if len(sys.argv) < 2:
        print('Usage: python3 pipeline/generate.py "your question here"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    retriever = Retriever()
    gen = Generator(retriever)
    out = gen.answer(query)

    print("=" * 70)
    print(f"Q: {out['query']}\n")
    print(f"A: {out['answer']}\n")
    if out["sources"]:
        explicit = any(s.get("cited") for s in out["sources"])
        print("Sources:" if explicit else "Sources consulted:")
        for s in out["sources"]:
            print(f"  [{s['n']}] {s['title']} — {s['url']}")
    print(f"\nfallback={out['fallback']} | model={out['model']}")
    print(f"retrieval={out['retrieval_latency_ms']}ms  "
          f"generation={out['generation_latency_ms']}ms")
    print("=" * 70)


if __name__ == "__main__":
    os.chdir(Path(__file__).parent.parent)
    main()
