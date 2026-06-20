"""Hybrid-search RAG over internal docs (fully cloud / OpenRouter).

Pipeline: ingest (pdf/md/txt) -> chunk -> hybrid retrieval
(BM25 sparse + Nemotron embeddings dense, fused with Reciprocal Rank Fusion)
-> grounded answer from an OpenRouter LLM with inline citations.
"""

from .embed import OpenRouterEmbedder
from .index import HybridIndex
from .ingest import Chunk, chunk_text, chunk_uploaded, load_documents
from .retrieve import RetrievedChunk, hybrid_search

__all__ = [
    "Chunk",
    "HybridIndex",
    "OpenRouterEmbedder",
    "RetrievedChunk",
    "chunk_text",
    "chunk_uploaded",
    "hybrid_search",
    "load_documents",
]
