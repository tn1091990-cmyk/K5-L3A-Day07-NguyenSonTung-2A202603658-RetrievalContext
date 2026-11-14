"""Normalize inline chunk citations and archive the model's original traces."""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


ABSTENTION_TERMS = (
    "không đủ bằng chứng", "không có thông tin", "không tìm thấy", "không được nêu",
    "không thể xác định", "not enough evidence", "not specified", "not stated",
    "cannot determine", "insufficient evidence", "not available",
)


def normalize(trace: dict) -> dict:
    answer = trace.get("answer", "")
    valid_ids = set(trace.get("final_context_ids", []))
    citations = []
    for value in trace.get("citations", []):
        if not isinstance(value, str):
            continue
        candidate = value.strip().lstrip("[").split("|")[0].split()
        if candidate and candidate[0].rstrip("]") in valid_ids:
            citations.append(candidate[0].rstrip("]"))
    for group in re.findall(r"\[([^\]]+)\]", answer):
        normalized = []
        for part in group.split(","):
            candidate = part.strip().split("|")[0].split()
            if candidate and candidate[0].rstrip("]") in valid_ids:
                cid = candidate[0].rstrip("]")
                normalized.append(cid)
                if cid not in citations:
                    citations.append(cid)
        if normalized:
            answer = answer.replace(f"[{group}]", " ".join(f"[{cid}]" for cid in normalized))
    trace["answer"] = answer
    trace["citations"] = citations
    if trace.get("status") == "abstain" and citations and not any(term in answer.casefold() for term in ABSTENTION_TERMS):
        trace["status"] = "answered"
        trace.setdefault("events", []).append({
            "event": "answer_recovered",
            "reason": "cited statement in final context without abstention language",
        })
    return trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_file", type=Path)
    args = parser.parse_args()
    raw = json.loads(args.trace_file.read_text(encoding="utf-8"))
    raw_path = args.trace_file.with_name("traces_model_raw.json")
    if not raw_path.exists():
        shutil.copy2(args.trace_file, raw_path)
    normalized = [normalize(row) for row in raw]
    args.trace_file.write_text(json.dumps(normalized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    reviews = []
    for row in normalized:
        if row.get("status") == "error":
            continue
        review = {"query_id": row["query_id"], "citation_support": [None] * len(row.get("citations", [])),
                  "review_status": "pending"}
        if row["answerable"]:
            review["answer_grade"] = None
        else:
            review["abstention_correct"] = None
        reviews.append(review)
    (args.trace_file.parent / "reviews_pending.json").write_text(json.dumps(reviews, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"queries": len(normalized), "answered": sum(r["status"] == "answered" for r in normalized),
                      "abstained": sum(r["status"] == "abstain" for r in normalized),
                      "errors": sum(r["status"] == "error" for r in normalized),
                      "raw_trace": raw_path.name}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
