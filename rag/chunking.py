"""Section-based chunking with a context prefix.

A document is cut at its section headings ("## ..."); an e-mail becomes one "Message"
section after its header lines. Long sections are split on paragraph, then sentence,
boundaries. Every chunk carries a prefix (document title, company, date, section) so a
passage such as "Total due: EUR 2,053.35" still says which invoice it belongs to."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

MAX_WORDS = 120


@dataclass
class Document:
    meta: dict
    body: str
    header: dict = field(default_factory=dict)


@dataclass
class Chunk:
    doc_id: str
    company: str
    title: str
    doc_type: str
    source: str
    lang: str
    date: str
    heading: str
    part: int
    text: str

    @property
    def prefixed(self) -> str:
        return (f"[{self.title} | {self.company} | {self.date} | {self.heading}]\n{self.text}")


def parse_document(raw: str) -> Document:
    m = re.match(r"---\n(.*?)\n---\n(.*)", raw, re.S)
    if not m:
        raise ValueError("missing front-matter")
    meta = dict(line.split(": ", 1) for line in m.group(1).splitlines())
    body = m.group(2)
    header = {}
    if meta.get("type") == "email":
        head, _, body = body.partition("\n\n")
        header = dict(line.split(": ", 1) for line in head.splitlines() if ": " in line)
    return Document(meta, body.strip(), header)


def split_sections(doc: Document) -> list[tuple[str, str]]:
    if doc.meta.get("type") == "email":
        return [("Message", doc.body)]
    sections, heading, buf = [], "Preamble", []
    for line in doc.body.splitlines():
        if line.startswith("## "):
            if "".join(buf).strip():
                sections.append((heading, "\n".join(buf).strip()))
            heading, buf = line[3:].strip(), []
        else:
            buf.append(line)
    if "".join(buf).strip():
        sections.append((heading, "\n".join(buf).strip()))
    return sections


def _split_long(text: str, max_words: int = MAX_WORDS) -> list[str]:
    if len(text.split()) <= max_words:
        return [text]
    units = [u for p in re.split(r"\n\s*\n", text) for u in re.split(r"(?<=[.!?])\s+", p)]
    parts, cur = [], []
    for u in units:
        if cur and len(" ".join(cur + [u]).split()) > max_words:
            parts.append(" ".join(cur))
            cur = []
        cur.append(u)
    if cur:
        parts.append(" ".join(cur))
    return parts


def chunk_document(doc: Document) -> list[Chunk]:
    m = doc.meta
    title = m["title"]
    if doc.header.get("From"):
        title = f"{title} (e-mail from {doc.header['From']})"
    out = []
    for heading, text in split_sections(doc):
        for i, part in enumerate(_split_long(text)):
            out.append(Chunk(m["doc_id"], m["company"], title, m["type"], m["source"], m["lang"], m["date"],
                             heading, i, part))
    return out


def load_corpus(root: Path) -> list[Chunk]:
    chunks = []
    for p in sorted(root.glob("*/*.txt")):
        chunks += chunk_document(parse_document(p.read_text(encoding="utf-8")))
    return chunks
