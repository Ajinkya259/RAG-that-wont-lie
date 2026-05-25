"""
Layer 09: Caching + memory.

A SQLite-backed query cache. Repeated questions return the stored answer
instantly instead of re-running retrieval + the LLM. Keyed by a normalized
hash of the query so "Who developed relativity?" and "who developed relativity"
hit the same entry.

Usage:
    from cache import QueryCache
    c = QueryCache()
    hit = c.get(query)          # -> dict or None
    c.set(query, result_dict)   # store
"""

import os
import json
import time
import hashlib
import sqlite3
from pathlib import Path

CACHE_DB = "cache/query_cache.db"
DEFAULT_TTL = 24 * 3600  # seconds; None = never expire


def _norm(query: str) -> str:
    return " ".join(query.lower().split())


def _key(query: str) -> str:
    return hashlib.sha256(_norm(query).encode()).hexdigest()


class QueryCache:
    def __init__(self, ttl: int = DEFAULT_TTL):
        Path("cache").mkdir(exist_ok=True)
        self.ttl = ttl
        self.db = sqlite3.connect(CACHE_DB)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                qhash   TEXT PRIMARY KEY,
                query   TEXT,
                result  TEXT,
                created REAL
            )
        """)
        self.db.commit()
        self.hits = 0
        self.misses = 0

    def get(self, query: str):
        row = self.db.execute(
            "SELECT result, created FROM cache WHERE qhash=?", (_key(query),)
        ).fetchone()
        if not row:
            self.misses += 1
            return None
        result_json, created = row
        if self.ttl is not None and (time.time() - created) > self.ttl:
            self.misses += 1
            return None
        self.hits += 1
        result = json.loads(result_json)
        result["cached"] = True
        return result

    def set(self, query: str, result: dict):
        store = {k: v for k, v in result.items() if k != "cached"}
        self.db.execute(
            "INSERT OR REPLACE INTO cache VALUES (?,?,?,?)",
            (_key(query), query, json.dumps(store), time.time()),
        )
        self.db.commit()

    def stats(self) -> dict:
        n = self.db.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        total = self.hits + self.misses
        return {
            "entries": n,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
        }


if __name__ == "__main__":
    # quick self-test, no models needed
    os.chdir(Path(__file__).parent.parent)
    c = QueryCache()
    print("get (miss):", c.get("test query xyz"))
    c.set("test query xyz", {"answer": "hello", "sources": []})
    got = c.get("TEST   Query XYZ")  # different case/spacing -> same key
    print("get (hit, normalized):", got)
    print("stats:", c.stats())
