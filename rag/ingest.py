"""Document loading + chunking.

Supports .pdf (pypdf), .md, .markdown, .txt. Chunking is paragraph-aware
with a character budget and overlap — transparent and dependency-light so
the retrieval behaviour is easy to reason about in interviews.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_SUFFIXES = {".pdf", ".md", ".markdown", ".txt"}


@dataclass
class Chunk:
    """One retrievable unit of text."""

    text: str
    source: str  # file name
    chunk_id: int  # index within its source document
    meta: dict = field(default_factory=dict)


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\n\n".join(pages)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_file(path: str | Path) -> str:
    """Return raw text for a single supported file."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        return _read_pdf(path)
    return _read_text(path)


def read_bytes(filename: str, data: bytes) -> str:
    """Extract text from in-memory upload bytes (no disk write — multi-user safe)."""
    import io

    if filename.lower().endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join((p.extract_text() or "") for p in reader.pages)
    return data.decode("utf-8", errors="ignore")


def chunk_uploaded(
    filename: str,
    data: bytes,
    chunk_size: int = 900,
    overlap: int = 150,
) -> list[Chunk]:
    """Read + chunk a single uploaded file held in memory."""
    return chunk_text(read_bytes(filename, data), source=filename, chunk_size=chunk_size, overlap=overlap)


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 900,
    overlap: int = 150,
) -> list[Chunk]:
    """Split text into ~chunk_size-char chunks on paragraph boundaries with overlap.

    Paragraphs longer than chunk_size are hard-split. Overlap is applied as a
    trailing character window carried into the next chunk to preserve context
    across boundaries.
    """
    text = _normalize(text)
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf = ""
    for para in paragraphs:
        # Hard-split paragraphs that exceed the budget on their own.
        while len(para) > chunk_size:
            head, para = para[:chunk_size], para[chunk_size:]
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.append(head)
        if len(buf) + len(para) + 2 <= chunk_size:
            buf = f"{buf}\n\n{para}" if buf else para
        else:
            if buf:
                chunks.append(buf)
            buf = para
    if buf:
        chunks.append(buf)

    # Apply overlap by prefixing each chunk (after the first) with the tail of
    # the previous one.
    out: list[Chunk] = []
    for i, body in enumerate(chunks):
        if i > 0 and overlap > 0:
            tail = chunks[i - 1][-overlap:]
            body = f"{tail} {body}"
        out.append(Chunk(text=body.strip(), source=source, chunk_id=i))
    return out


def load_documents(
    folder: str | Path,
    chunk_size: int = 900,
    overlap: int = 150,
) -> list[Chunk]:
    """Load + chunk every supported document under `folder` (recursive)."""
    folder = Path(folder)
    all_chunks: list[Chunk] = []
    if not folder.exists():
        return all_chunks
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            try:
                raw = read_file(path)
            except Exception as exc:  # noqa: BLE001 - skip unreadable files, keep going
                print(f"[ingest] skip {path.name}: {exc}")
                continue
            all_chunks.extend(chunk_text(raw, source=path.name, chunk_size=chunk_size, overlap=overlap))
    return all_chunks
