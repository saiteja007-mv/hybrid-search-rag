"""Hybrid index: BM25 (sparse) + dense embeddings, persisted to disk.

The dense side uses sentence-transformers (MiniLM by default) with cosine
similarity over L2-normalized vectors — small doc sets don't need a FAISS/ANN
layer, and the brute-force path keeps the math inspectable.
"""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ingest import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenizer shared by BM25 indexing + querying."""
    return _TOKEN_RE.findall(text.lower())


@dataclass
class HybridIndex:
    chunks: list[Chunk]
    embeddings: np.ndarray  # (N, D) float32, L2-normalized
    model_name: str
    _bm25: object = None  # rank_bm25.BM25Okapi

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, chunks: list[Chunk], embedder) -> "HybridIndex":
        """Build from chunks + any embedder exposing ``.encode(list[str]) -> np.ndarray``."""
        from rank_bm25 import BM25Okapi

        if not chunks:
            raise ValueError("Cannot build an index from zero chunks.")

        texts = [c.text for c in chunks]
        emb = np.asarray(embedder.encode(texts), dtype=np.float32)

        bm25 = BM25Okapi([tokenize(t) for t in texts])
        model_name = getattr(embedder, "model_name", None) or "embeddings"
        idx = cls(chunks=chunks, embeddings=emb, model_name=model_name)
        idx._bm25 = bm25
        return idx

    # ------------------------------------------------------------------ scores
    def dense_scores(self, query_vec: np.ndarray) -> np.ndarray:
        """Cosine similarity (vectors are pre-normalized) → (N,) in [-1, 1]."""
        return self.embeddings @ query_vec.astype(np.float32)

    def sparse_scores(self, query: str) -> np.ndarray:
        """BM25 relevance scores → (N,) non-negative."""
        return np.asarray(self._bm25.get_scores(tokenize(query)), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.chunks)

    # ------------------------------------------------------------------ persist
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(
                {
                    "chunks": self.chunks,
                    "embeddings": self.embeddings,
                    "model_name": self.model_name,
                    "bm25": self._bm25,
                },
                fh,
            )

    @classmethod
    def load(cls, path: str | Path) -> "HybridIndex":
        with Path(path).open("rb") as fh:
            d = pickle.load(fh)
        idx = cls(chunks=d["chunks"], embeddings=d["embeddings"], model_name=d["model_name"])
        idx._bm25 = d["bm25"]
        return idx
