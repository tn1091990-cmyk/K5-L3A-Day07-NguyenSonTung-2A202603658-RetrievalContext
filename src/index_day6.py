"""Rebuild the Day 6 Strategy B vector index in the configured local database."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .corpus import write_chunks
from .run_retrieval import ROOT, configure_day6


def main():
    parser = argparse.ArgumentParser(description="Index the 270 Day 6 Strategy B chunks with its real embedding model.")
    parser.add_argument("--day6-root", type=Path, default=Path(r"C:\Users\DELL\Nextcloud\AIinAction\Day6Lab\K5-L3-Day06-2A202603658-NguyenSonTung-DataIngestionEngineering"))
    parser.add_argument("--force", action="store_true", help="Replace Strategy B chunks for all documents in the Day 6 database.")
    args = parser.parse_args()
    store, embedder = configure_day6(args.day6_root)
    chunks_json = ROOT / "data" / "day6_paragraph_chunks.json"
    chunks = write_chunks(args.day6_root / "starter" / "data", chunks_json)
    if not args.force:
        # Refuse to replace a populated index unless the operator explicitly confirms.
        import psycopg
        with psycopg.connect(store.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT count(*) FROM information_schema.tables WHERE table_schema='public' AND table_name='rag_chunks'")
                table_exists = cursor.fetchone()[0] > 0
                if table_exists:
                    cursor.execute("SELECT count(*) FROM rag_chunks WHERE strategy='B'")
                    existing = cursor.fetchone()[0]
                    if existing:
                        raise RuntimeError(f"Strategy B already contains {existing} rows; pass --force only when replacement is intended.")
    started = time.perf_counter()
    store.create_schema()
    vectors = embedder.embed_documents([chunk["text"] for chunk in chunks])
    if len(vectors) != len(chunks):
        raise RuntimeError("Embedding response count does not match the chunk count")
    from models import Chunk
    models = [Chunk(chunk_id=c["id"], document_id=c["source_id"], source=c["source"], page=c["page"],
                    text=c["text"], metadata=c["metadata"]) for c in chunks]
    store.upsert("B", models, vectors)
    result = {"documents": len({c["source_id"] for c in chunks}), "chunks_indexed": len(chunks),
              "strategy": "B", "embedding_model": embedder.model_name, "dimensions": embedder.dimensions,
              "elapsed_seconds": time.perf_counter() - started, "index_rebuilt": True}
    output = ROOT / "results" / "day6_index_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
