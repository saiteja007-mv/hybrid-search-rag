"""Hybrid retrieval: fuse dense + sparse rankings.

Two fusion strategies, both exposed in the UI so the trade-off is visible:

* ``rrf``      — Reciprocal Rank Fusion (rank-based, scale-free, the robust default).
* ``weighted`` — min-max normalize each score list, then ``alpha`` blend
                 (alpha=1.0 → pure dense, 0.0 → pure sparse).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .index import HybridIndex
from .ingest import Chunk


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float  # fused score (higher = better)
    dense_score: float
    sparse_score: float
    dense_rank: int  # 1-indexed rank in the dense list
    sparse_rank: int  # 1-indexed rank in the sparse list


def _ranks(scores: np.ndarray) -> np.ndarray:
    """Return 1-indexed ranks (rank 1 = highest score)."""
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(scores))
    return ranks + 1


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def hybrid_search(
    index: HybridIndex,
    query: str,
    embedder,
    top_k: int = 5,
    method: str = "rrf",
    rrf_k: int = 60,
    alpha: float = 0.5,
) -> list[RetrievedChunk]:
    """Rank chunks by a fusion of dense (semantic) and sparse (BM25) relevance."""
    query_vec = np.asarray(embedder.encode([query]), dtype=np.float32)[0]

    dense = index.dense_scores(query_vec)
    sparse = index.sparse_scores(query)

    dense_rank = _ranks(dense)
    sparse_rank = _ranks(sparse)

    if method == "weighted":
        fused = alpha * _minmax(dense) + (1.0 - alpha) * _minmax(sparse)
    else:  # rrf
        fused = 1.0 / (rrf_k + dense_rank) + 1.0 / (rrf_k + sparse_rank)

    top = np.argsort(-fused, kind="stable")[:top_k]
    return [
        RetrievedChunk(
            chunk=index.chunks[i],
            score=float(fused[i]),
            dense_score=float(dense[i]),
            sparse_score=float(sparse[i]),
            dense_rank=int(dense_rank[i]),
            sparse_rank=int(sparse_rank[i]),
        )
        for i in top
    ]
