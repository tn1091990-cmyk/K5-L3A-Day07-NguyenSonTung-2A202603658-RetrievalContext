"""Rebuild the Day 6 paragraph chunks with the exact IDs used by pgvector.

Uses only the Python standard library so a broken Day 6 virtualenv does not
prevent lexical retrieval. Chunk text and IDs follow starter/src/chunking.py.
"""
from __future__ import annotations

import html
import importlib
import json
import re
import sys
import unicodedata
from html.parser import HTMLParser
from pathlib import Path


class _Text(HTMLParser):
    BLOCK = {"p", "div", "section", "article", "main", "header", "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "li", "table", "tr", "blockquote", "pre"}
    IGNORE = {"script", "style", "noscript", "svg", "iframe", "nav", "footer"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.ignored = 0
        self.title_parts: list[str] = []
        self.in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self.IGNORE:
            self.ignored += 1
        elif not self.ignored:
            if tag == "title":
                self.in_title = True
            if tag in self.BLOCK or tag == "br":
                self.parts.append("\n")
            if tag == "td":
                self.parts.append(" | ")

    def handle_endtag(self, tag):
        if tag in self.IGNORE and self.ignored:
            self.ignored -= 1
        elif not self.ignored and tag in self.BLOCK:
            self.parts.append("\n")
        if tag == "title" and not self.ignored:
            self.in_title = False

    def handle_data(self, data):
        if not self.ignored:
            (self.title_parts if self.in_title else self.parts).append(data)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def paragraph_chunks(document_id: str, source: str, text: str, metadata: dict, max_size: int = 1000) -> list[dict]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[dict] = []
    parts: list[str] = []
    length = 0

    def add(value: str):
        index = len(chunks)
        chunks.append({"id": f"{document_id}-p1-paragraph-{index}", "source_id": document_id,
                       "source": source, "page": 1, "text": value, "metadata": metadata})

    for paragraph in paragraphs:
        if len(paragraph) > max_size:
            if parts:
                add("\n\n".join(parts))
                parts, length = [], 0
            for start in range(0, len(paragraph), max_size):
                add(paragraph[start:start + max_size])
            continue
        proposed = length + (2 if parts else 0) + len(paragraph)
        if proposed > max_size:
            add("\n\n".join(parts))
            parts, length = [paragraph], len(paragraph)
        else:
            parts.append(paragraph)
            length = proposed
    if parts:
        add("\n\n".join(parts))
    return chunks


def _field(text: str, label: str) -> str | None:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        current = line.strip()
        if not re.match(rf"^{re.escape(label)}\s*:", current, re.I):
            continue
        value = current.split(":", 1)[1].strip().lstrip("|").strip()
        if value:
            return value
        for following in lines[i + 1:i + 5]:
            candidate = following.strip().lstrip("|").strip()
            if candidate and candidate != "|":
                return candidate
    return None


def extract_metadata(text: str, title: str, source_path: Path) -> dict:
    """Call Day 6's metadata enrichment so filters match its stored schema."""
    source_dir = source_path.parent.parent / "src"
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    models = importlib.import_module("models")
    metadata_cpu = importlib.import_module("metadata_cpu")
    page = models.DocumentPage(
        document_id=re.sub(r"[^a-z0-9]+", "-", source_path.stem.lower()).strip("-"),
        source=str(source_path), page=1, text=text,
        metadata={"title": title, "source_path": str(source_path),
                  "source_unit_type": "html_document", "character_count": len(text),
                  "extraction_status": "success" if text else "empty"},
    )
    return metadata_cpu.enrich_cpu_metadata(page).metadata


def build_chunks(day6_data: Path) -> list[dict]:
    files = sorted([*day6_data.glob("*.html"), *day6_data.glob("*.htm")])
    chunks = []
    for path in files:
        parser = _Text()
        parser.feed(path.read_text(encoding="utf-8", errors="replace"))
        text = normalize("".join(parser.parts))
        document_id = re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")
        title = normalize("".join(parser.title_parts)) or path.stem
        metadata = extract_metadata(text, title, path)
        chunks.extend(paragraph_chunks(document_id, str(path), text, metadata))
    if not files:
        raise FileNotFoundError(f"No Day 6 HTML files found in {day6_data}")
    if len(files) < 30:
        raise ValueError(f"Expected at least 30 Day 6 documents; found {len(files)}")
    return chunks


def write_chunks(day6_data: Path, output: Path) -> list[dict]:
    chunks = build_chunks(day6_data)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return chunks
