
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    title: str
    source: str
    text: str


@dataclass(frozen=True)
class Corpus:
    documents: list[Document]


def document_from_text(source: str, text: str) -> Document:
    first_line = text.splitlines()[0] if text else ""
    title = first_line.lstrip("#").strip() if first_line.startswith("#") else Path(source).stem
    return Document(title=title, source=source, text=text)


def load_corpus(data_dir: str | Path) -> Corpus:
    data_dir = Path(data_dir)
    documents = [
        document_from_text(path.name, path.read_text(encoding="utf-8"))
        for path in sorted(data_dir.glob("*.md"))
    ]
    return Corpus(documents=documents)