"""Quick multi-query test of the full RAG pipeline."""
import os, warnings
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
sys.path.insert(0, "pipeline")

from retrieve import Retriever
from generate import Generator

QUERIES = [
    "What is photosynthesis?",
    "asdfghjkl qwerty zxcvbnm nonsense",
    "What did Albert Einstein eat for breakfast?",
]

r = Retriever()
g = Generator(r)
for q in QUERIES:
    out = g.answer(q)
    print("=" * 60)
    print(f"Q: {q}")
    print(f"A: {out['answer']}")
    print(f"fallback={out['fallback']} sources={len(out['sources'])} "
          f"gen={out['generation_latency_ms']}ms")
