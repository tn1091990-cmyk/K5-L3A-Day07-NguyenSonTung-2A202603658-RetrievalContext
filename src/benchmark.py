"""Create and validate the 24-case benchmark using exact Day 6 chunk evidence."""
from __future__ import annotations

import json
import re
from pathlib import Path


# Evidence needles are topic terms, never provided to retrieval or the model.
SPECS = [
    ("D01", "dev", "AMD Ryzen 3 8300G có bao nhiêu nhân?", "4 nhân.", "amd-ryzen-3-8300g", "with 4 cores"),
    ("D02", "dev", "Xung cơ bản của Ryzen 5 7600X là bao nhiêu?", "4.7 GHz.", "amd-ryzen-5-7600x", "operates at 4.7 GHz by default"),
    ("D03", "dev", "Ryzen 5 7500X3D có bao nhiêu cache L3?", "96 MB.", "amd-ryzen-5-7500x3d", "has 96 MB of L3 cache"),
    ("D04", "dev", "Ryzen 5 8600G dùng GPU tích hợp Radeon dòng nào?", "Radeon 760M.", "amd-ryzen-5-8600g", "Radeon 760M"),
    ("D05", "dev", "TDP của Ryzen 7 8700GE là bao nhiêu?", "35 W.", "amd-ryzen-7-8700ge", "TDP of 35 W"),
    ("D06", "dev", "Codename của Ryzen 7 9700X là gì?", "Granite Ridge.", "amd-ryzen-7-9700x", "Granite Ridge"),
    ("D07", "dev", "Ryzen 9 7900X có bao nhiêu nhân?", "12 nhân.", "amd-ryzen-9-7900x", "with 12 cores"),
    ("D08", "dev", "Cache L3 của Ryzen 9 7950X3D là bao nhiêu?", "128 MB.", "amd-ryzen-9-7950x3d", "has 128 MB of L3 cache"),
    ("D09", "dev", "Turbo tối đa của Ryzen 7 9800X3D là bao nhiêu?", "5.2 GHz.", "amd-ryzen-7-9800x3d", "boost up to 5.2 GHz"),
    ("D10", "dev", "Ryzen 9 9950X có bao nhiêu nhân?", "16 nhân.", "amd-ryzen-9-9950x", "with 16 cores"),
    ("D11", "dev", "Ryzen 5 7400 có multiplier mở khóa không?", "Có.", "amd-ryzen-5-7400", "unlocked multiplier"),
    ("D12", "dev", "Ryzen 5 8500G hỗ trợ PCI Express thế hệ nào?", "PCI Express Gen 4.", "amd-ryzen-5-8500g", "PCI-Express Gen 4"),
    ("D13", "dev", "TDP của Ryzen 7 7700 là bao nhiêu?", "65 W.", "amd-ryzen-7-7700", "TDP of 65 W"),
    ("D14", "dev", "Ryzen 9 9900X có Turbo tối đa bao nhiêu?", "5.6 GHz.", "amd-ryzen-9-9900x", "boost up to 5.6 GHz"),
    ("D15", "dev", "Xung cơ bản của Ryzen 5 7500 là bao nhiêu?", "3.7 GHz.", "amd-ryzen-5-7500", "operates at 3.7 GHz by default"),
    ("T01", "test", "Ryzen 5 7600X3D đạt Turbo tối đa bao nhiêu?", "4.7 GHz.", "amd-ryzen-5-7600x3d", "boost up to 4.7 GHz"),
    ("T02", "test", "TDP của Ryzen 7 7700X được ghi là bao nhiêu?", "105 W.", "amd-ryzen-7-7700x", "TDP of 105 W"),
    ("T03", "test", "Ryzen 7 7700X3D có cache L3 bao nhiêu?", "96 MB.", "amd-ryzen-7-7700x3d", "has 96 MB of L3 cache"),
    ("T04", "test", "Ryzen 9 7900X3D có bao nhiêu nhân?", "12 nhân.", "amd-ryzen-9-7900x3d", "with 12 cores"),
    ("T05", "test", "Xung cơ bản của Ryzen 9 7950X là bao nhiêu?", "4.5 GHz.", "amd-ryzen-9-7950x", "operates at 4.5 GHz by default"),
]
ABSTENTIONS = [
    {"id": "D16", "split": "dev", "answerable": False,
     "question": "Thời hạn bảo hành chính thức của Ryzen 7 7700X tại Việt Nam là bao lâu?",
     "expected_behavior": "search_and_abstain", "absence_reason": "CPU specifications corpus contains no warranty terms or Vietnam warranty policy."},
    {"id": "D17", "split": "dev", "answerable": False,
     "question": "Ryzen 9 7900X3D cần bao nhiêu cổng USB trên bo mạch chủ?",
     "expected_behavior": "search_and_abstain", "absence_reason": "The CPU corpus does not specify motherboard USB port counts."},
    {"id": "T06", "split": "test", "answerable": False,
     "question": "Ryzen 5 7600X3D bán lẻ hôm nay có giá bao nhiêu tại Việt Nam?",
     "expected_behavior": "search_and_abstain", "absence_reason": "The corpus has no current Vietnam retail pricing."},
    {"id": "T07", "split": "test", "answerable": False,
     "question": "Ryzen 9 7950X hỗ trợ tối đa bao nhiêu màn hình 8K?",
     "expected_behavior": "search_and_abstain", "absence_reason": "The indexed CPU specification pages do not state supported 8K display counts."},
]


def _quote(chunks: list[dict], source_id: str, needle: str) -> tuple[str, str]:
    for chunk in chunks:
        if chunk["source_id"] == source_id and needle.casefold() in chunk["text"].casefold():
            text = chunk["text"]
            start = text.casefold().index(needle.casefold())
            previous_sentence = text.rfind(". ", 0, start)
            left = 0 if previous_sentence < 0 else previous_sentence + 2
            end = text.find(". ", start + len(needle))
            end = min(len(text), end + 1 if end >= 0 else start + len(needle) + 100)
            evidence = text[left:end].strip()
            return chunk["id"], evidence
    raise ValueError(f"Evidence not found in Day 6 chunk: {source_id!r} / {needle!r}")


def create_benchmark(chunks: list[dict]) -> list[dict]:
    rows = []
    for case_id, split, question, answer, source_id, needle in SPECS:
        chunk_id, evidence = _quote(chunks, source_id, needle)
        rows.append({"id": case_id, "split": split, "answerable": True, "question": question,
                     "expected_answer": answer,
                     "gold": [{"chunk_id": chunk_id, "source_id": source_id, "page": 1, "evidence": evidence}]})
    rows.extend(ABSTENTIONS)
    counts = {(split, answerable): sum(r["split"] == split and r.get("answerable") == answerable for r in rows)
              for split in ("dev", "test") for answerable in (True, False)}
    if counts != {("dev", True): 15, ("dev", False): 2, ("test", True): 5, ("test", False): 2}:
        raise ValueError(f"Benchmark split counts are invalid: {counts}")
    return rows


def write_benchmark(chunks: list[dict], output: Path) -> list[dict]:
    rows = create_benchmark(chunks)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return rows
