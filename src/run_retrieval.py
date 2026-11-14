"""Run retrieval comparisons and save query-level traces and aggregate scores."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

from .benchmark import write_benchmark
from .corpus import write_chunks
from .retrieval import BM25, Day6Dense, HybridRRF, pack_context

ROOT = Path(__file__).resolve().parents[1]


def configure_day6(day6_root: Path):
    starter = day6_root / "starter"
    if not (starter / ".env").is_file():
        raise FileNotFoundError(f"Day 6 config not found: {starter / '.env'}")
    try:
        from dotenv import load_dotenv
        load_dotenv(starter / ".env", override=False)
    except ImportError as exc:
        raise RuntimeError("Install python-dotenv from Day 6 requirements to load the local Day 6 settings.") from exc
    sys.path.insert(0, str(starter / "src"))
    try:
        from embeddings import MyEmbedder
        from store import PgVectorStore
    except ImportError as exc:
        raise RuntimeError("Install Day 6 requirements (including psycopg and openai) into the active Python.") from exc
    database_url = os.getenv("DATABASE_URL")
    embedding_model = os.getenv("EMBEDDING_MODEL")
    if not database_url or not embedding_model:
        raise RuntimeError("Day 6 .env must define DATABASE_URL and EMBEDDING_MODEL; values are never printed.")
    embedder = MyEmbedder(embedding_model)
    store = PgVectorStore(database_url, embedder.dimensions)
    return store, embedder


def has_evidence(result: dict, gold: list[dict]) -> bool:
    return any(result["id"] == item["chunk_id"] and item["evidence"].casefold() in result["text"].casefold()
               for item in gold)


def evaluate_method(method, questions: list[dict], method_name: str, *, top_k: int, budget: int) -> list[dict]:
    traces = []
    for case in questions:
        started = time.perf_counter()
        results = method.search(case["question"], top_k=top_k, filters=None)
        context, context_ids = pack_context(results, budget)
        answerable = case["answerable"]
        traces.append({"query_id": case["id"], "split": case["split"], "question": case["question"],
                       "strategy": method_name, "top_k": top_k, "context_budget_chars": budget,
                       "filter": None, "results": results, "context": context, "context_ids": context_ids,
                       "hit1": bool(answerable and results and has_evidence(results[0], case["gold"])),
                       "evidence_hit3": bool(answerable and any(has_evidence(r, case["gold"]) for r in results[:3])),
                       "context_evidence": bool(answerable and any(has_evidence(r, case["gold"]) for r in results if r["id"] in context_ids)),
                       "elapsed_ms": (time.perf_counter() - started) * 1000,
                       "answerable": answerable})
    return traces


def summary(traces: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in traces:
        groups[(row["split"], row["strategy"])].append(row)
    out = {}
    for (split, strategy), rows in sorted(groups.items()):
        positives = [r for r in rows if r["answerable"]]
        n = len(positives)
        out[f"{split}/{strategy}"] = {
            "answerable_n": n,
            "Hit@1": {"n": sum(r["hit1"] for r in positives), "N": n},
            "Evidence_Hit@3": {"n": sum(r["evidence_hit3"] for r in positives), "N": n},
            "Context_Evidence_Rate": {"n": sum(r["context_evidence"] for r in positives), "N": n},
            "median_ms": sorted(r["elapsed_ms"] for r in rows)[len(rows) // 2],
            "unanswerable_queries": sum(not r["answerable"] for r in rows),
            "service_cost_usd": 0,
        }
    return out


def run(day6_root: Path, split: str, output: Path):
    chunks = write_chunks(day6_root / "starter" / "data", ROOT / "data" / "day6_paragraph_chunks.json")
    cases = write_benchmark(chunks, ROOT / "data" / "benchmark_day7.json")
    questions = [c for c in cases if c["split"] == split]
    store, embedder = configure_day6(day6_root)
    lexical = BM25(chunks)
    dense = Day6Dense(store, embedder, "B")
    baseline = BM25(chunks, 1.5, .75)
    selected = BM25(chunks, .8, .25)
    if split == "dev":
        methods = [
            (dense, "day6_dense_B_top3", 3, 2400),
            (baseline, "bm25_k1_1.5_b_0.75_top3", 3, 2400),
            (HybridRRF(dense, lexical, 60), "hybrid_rrf60_top3", 3, 2400),
            (BM25(chunks, .8, .25), "experiment_bm25_k1_0.8_b_0.25", 3, 2400),
            (selected, "experiment_bm25_tuned_budget3600", 3, 3600),
            (HybridRRF(dense, lexical, 60), "experiment_hybrid_budget1200", 3, 1200),
        ]
    else:
        methods = [(baseline, "bm25_k1_1.5_b_0.75_top3", 3, 2400),
                   (selected, "selected_bm25_k1_0.8_b_0.25_budget3600", 3, 3600)]
    traces = []
    for method, name, k, budget in methods:
        traces.extend(evaluate_method(method, questions, name, top_k=k, budget=budget))
    output.mkdir(parents=True, exist_ok=True)
    (output / "retrieval_traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result = {"split": split, "corpus_documents": len({c["source_id"] for c in chunks}),
              "chunks": len(chunks), "embedding_model": embedder.model_name,
              "embedding_dimensions": embedder.dimensions, "vector_strategy": "B",
              "methods": summary(traces)}
    (output / "retrieval_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def run_offline_lexical(split: str, output: Path):
    """Run reproducible lexical-only experiments when local services are offline."""
    chunks_path = ROOT / "data" / "day6_paragraph_chunks.json"
    benchmark_path = ROOT / "data" / "benchmark_day7.json"
    if not chunks_path.is_file() or not benchmark_path.is_file():
        raise FileNotFoundError("Build data/day6_paragraph_chunks.json and data/benchmark_day7.json first.")
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    cases = json.loads(benchmark_path.read_text(encoding="utf-8"))
    questions = [c for c in cases if c["split"] == split]
    baseline = BM25(chunks, 1.5, .75)
    tuned = BM25(chunks, .8, .25)
    if split == "dev":
        methods = [
            (baseline, "bm25_k1_1.5_b_0.75_top3", 3, 2400),
            (BM25(chunks, .8, .25), "experiment_bm25_k1_0.8_b_0.25", 3, 2400),
            (BM25(chunks, 1.5, .75), "experiment_bm25_context1200", 3, 1200),
            (tuned, "experiment_bm25_tuned_budget3600", 3, 3600),
        ]
    else:
        methods = [(baseline, "bm25_k1_1.5_b_0.75_top3", 3, 2400),
                   (tuned, "selected_bm25_k1_0.8_b_0.25_budget3600", 3, 3600)]
    traces = []
    for method, name, k, budget in methods:
        traces.extend(evaluate_method(method, questions, name, top_k=k, budget=budget))
    output.mkdir(parents=True, exist_ok=True)
    (output / "retrieval_traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result = {"split": split, "offline_lexical_only": True, "corpus_documents": len({c["source_id"] for c in chunks}),
              "chunks": len(chunks), "methods": summary(traces),
              "note": "Dense and hybrid methods require the real local embedding server and PostgreSQL; they were not replaced by offline estimates."}
    (output / "retrieval_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--day6-root", type=Path, default=Path(os.getenv("DAY6_ROOT", r"C:\Users\DELL\Nextcloud\AIinAction\Day6Lab\K5-L3-Day06-2A202603658-NguyenSonTung-DataIngestionEngineering")))
    parser.add_argument("--split", required=True, choices=("dev", "test"))
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "retrieval-dev")
    parser.add_argument("--offline-lexical-only", action="store_true", help="Run BM25 experiments only; never substitute for dense/hybrid results.")
    args = parser.parse_args()
    if args.offline_lexical_only:
        run_offline_lexical(args.split, args.output)
    else:
        run(args.day6_root, args.split, args.output)


if __name__ == "__main__":
    main()
