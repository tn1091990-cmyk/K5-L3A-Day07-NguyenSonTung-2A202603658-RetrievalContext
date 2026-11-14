"""Dense Day 6, BM25 lexical, and reciprocal-rank-fusion retrieval."""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Any


def tokens(text: str) -> list[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())


class BM25:
    def __init__(self, chunks: list[dict], k1: float = 1.5, b: float = .75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.rows = [Counter(tokens(c["text"])) for c in chunks]
        self.lengths = [sum(row.values()) for row in self.rows]
        self.avg_len = sum(self.lengths) / max(1, len(self.lengths))
        df = Counter(term for row in self.rows for term in row)
        self.idf = {term: math.log(1 + (len(chunks) - n + .5) / (n + .5)) for term, n in df.items()}

    def search(self, query: str, top_k: int = 3, filters: dict | None = None) -> list[dict]:
        qterms = set(tokens(query))
        out = []
        for chunk, row, length in zip(self.chunks, self.rows, self.lengths):
            if filters and any(chunk.get("metadata", {}).get(k) != v for k, v in filters.items()):
                continue
            score = 0.0
            for term in qterms:
                freq = row.get(term, 0)
                if freq:
                    denom = freq + self.k1 * (1 - self.b + self.b * length / max(1, self.avg_len))
                    score += self.idf.get(term, 0) * freq * (self.k1 + 1) / denom
            if score:
                out.append({**chunk, "score": score, "strategy": "bm25"})
        return sorted(out, key=lambda c: (-c["score"], c["id"]))[:top_k]


class Day6Dense:
    def __init__(self, store, embedder, strategy: str = "B"):
        self.store, self.embedder, self.strategy = store, embedder, strategy

    def search(self, query: str, top_k: int = 3, filters: dict | None = None) -> list[dict]:
        vector = self.embedder.embed_query(query)
        rows = self.store.search(self.strategy, vector, top_k=top_k, filters=filters)
        return [{"id": r["chunk_id"], "source_id": r["document_id"], "source": r["source"],
                 "page": r["page"], "text": r["text"], "metadata": r["metadata"],
                 "score": r["score"], "strategy": "day6_dense"} for r in rows]


class HybridRRF:
    def __init__(self, dense: Day6Dense, lexical: BM25, rrf_k: int = 60):
        self.dense, self.lexical, self.rrf_k = dense, lexical, rrf_k

    def search(self, query: str, top_k: int = 3, filters: dict | None = None) -> list[dict]:
        pool = max(10, top_k * 4)
        dense = self.dense.search(query, pool, filters)
        lexical = self.lexical.search(query, pool, filters)
        by_id: dict[str, dict[str, Any]] = {}
        for result_set in (dense, lexical):
            for rank, row in enumerate(result_set, start=1):
                current = by_id.setdefault(row["id"], {**row, "rrf_score": 0.0, "rank_sources": []})
                current["rrf_score"] += 1 / (self.rrf_k + rank)
                current["rank_sources"].append(row["strategy"])
        return [{**row, "score": row["rrf_score"], "strategy": "hybrid_rrf"}
                for row in sorted(by_id.values(), key=lambda r: (-r["rrf_score"], r["id"]))[:top_k]]


def pack_context(results: list[dict], budget_chars: int = 2400) -> tuple[str, list[str]]:
    blocks, ids, used = [], [], set()
    for row in results:
        if row["id"] in used:
            continue
        block = f"[{row['id']} | {row['source']} | page {row['page']}]\n{row['text']}"
        proposed = "\n\n".join(blocks + [block])
        if len(proposed) <= budget_chars:
            blocks.append(block)
            ids.append(row["id"])
            used.add(row["id"])
    return "\n\n".join(blocks), ids
