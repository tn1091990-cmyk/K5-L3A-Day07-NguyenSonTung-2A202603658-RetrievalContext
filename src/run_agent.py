"""Run the real tool-using agent on the selected split and save full traces."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from .agent import ModelController, run_agent
from .benchmark import write_benchmark
from .corpus import write_chunks
from .retrieval import BM25, Day6Dense, HybridRRF
from .run_retrieval import ROOT, configure_day6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--day6-root", type=Path, default=Path(os.getenv("DAY6_ROOT", r"C:\Users\DELL\Nextcloud\AIinAction\Day6Lab\K5-L3-Day06-2A202603658-NguyenSonTung-DataIngestionEngineering")))
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--method", choices=("dense", "bm25", "hybrid"), default="hybrid")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--budget-chars", type=int, default=3600)
    parser.add_argument("--bm25-k1", type=float, default=.8)
    parser.add_argument("--bm25-b", type=float, default=.25)
    parser.add_argument("--only", action="append", help="Run only the named query ID; repeat to include multiple cases.")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    chunks = write_chunks(args.day6_root / "starter" / "data", ROOT / "data" / "day6_paragraph_chunks.json")
    cases = write_benchmark(chunks, ROOT / "data" / "benchmark_day7.json")
    chosen = [case for case in cases if case["split"] == args.split and (not args.only or case["id"] in args.only)]
    if args.only and len(chosen) != len(set(args.only)):
        raise ValueError("One or more --only IDs do not belong to the selected split")
    output = args.output or ROOT / "results" / f"agent-{args.method}-{args.split}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "cases.json").write_text(json.dumps([{"id": c["id"], "split": c["split"], "answerable": c["answerable"]} for c in chosen], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        store, embedder = configure_day6(args.day6_root)
        dense = Day6Dense(store, embedder, "B")
        lexical = BM25(chunks, args.bm25_k1, args.bm25_b)
        retrievers = {"dense": dense, "bm25": lexical, "hybrid": HybridRRF(dense, lexical, 60)}
        # Construction checks model configuration; every answer still comes from a live model call.
        controller = ModelController()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        traces = [{"query_id": case["id"], "split": args.split, "answerable": case["answerable"],
                   "method": args.method, "status": "error", "error": error,
                   "events": [], "citations": [], "elapsed_ms": 0} for case in chosen]
        (output / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summary = {"split": args.split, "method": args.method, "case_count": len(chosen),
                   "model_calls": 0, "answered": 0, "abstained": 0, "errors": len(traces),
                   "setup_error": error, "manual_review_required": "No answer was produced; resolve service setup and rerun."}
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (output / "reviews_pending.json").write_text("[]\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    traces = []
    for index, case in enumerate(chosen, start=1):
        started = time.perf_counter()
        try:
            result = run_agent(case["question"], controller, retrievers[args.method], top_k=args.top_k, budget_chars=args.budget_chars)
            traces.append({"query_id": case["id"], "split": args.split, "answerable": case["answerable"],
                           "method": args.method, **result})
            print(f"[{index}/{len(chosen)}] {case['id']}: {result['status']} ({result['elapsed_ms']:.0f} ms)", flush=True)
        except Exception as exc:
            traces.append({"query_id": case["id"], "split": args.split, "answerable": case["answerable"],
                           "method": args.method, "status": "error", "error": f"{type(exc).__name__}: {exc}",
                           "events": [], "citations": [], "elapsed_ms": (time.perf_counter() - started) * 1000})
            print(f"[{index}/{len(chosen)}] {case['id']}: error {type(exc).__name__}", flush=True)
    (output / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {"split": args.split, "method": args.method, "model": controller.model,
               "model_calls": controller.calls, "case_count": len(chosen),
               "answered": sum(t.get("status") == "answered" for t in traces),
               "abstained": sum(t.get("status") == "abstain" for t in traces),
               "errors": sum(t.get("status") == "error" for t in traces),
               "median_ms": sorted(t.get("elapsed_ms", 0) for t in traces)[len(traces) // 2],
               "manual_review_required": "Grounded answer, citation support, and abstention labels must be filled after reading traces and source chunks."}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    reviews = []
    for trace in traces:
        if trace.get("status") == "error":
            continue
        review = {"query_id": trace["query_id"], "citation_support": [None for _ in trace.get("citations", [])],
                  "review_status": "pending"}
        if trace["answerable"]:
            review["answer_grade"] = None
        else:
            review["abstention_correct"] = None
        reviews.append(review)
    (output / "reviews_pending.json").write_text(json.dumps(reviews, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
