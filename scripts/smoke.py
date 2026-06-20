"""Headless smoke test: ingest -> embed -> index -> hybrid retrieve -> grounded LLM.

Requires OPENROUTER_API_KEY (embeddings + chat both run on OpenRouter).
Reads it from the env or .streamlit/secrets.toml.

    python scripts/smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Allow running without the env var by sourcing the local secrets file.
try:
    import tomllib

    _sec = ROOT / ".streamlit" / "secrets.toml"
    if _sec.exists():
        import os

        for k, v in tomllib.load(_sec.open("rb")).items():
            os.environ.setdefault(k, str(v))
except Exception:  # noqa: BLE001
    pass

from rag.embed import OpenRouterEmbedder
from rag.index import HybridIndex
from rag.ingest import load_documents
from rag.llm import get_api_key, stream_answer
from rag.retrieve import hybrid_search

QUERIES = [
    "How many approvals does a payments PR need?",
    "How much parental leave do birth parents get?",
    "What is the rollback latency threshold after a deploy?",
]


def main() -> int:
    if not get_api_key():
        print("[smoke] FAIL: OPENROUTER_API_KEY not set (needed for embeddings + chat).")
        return 1

    docs_dir = ROOT / "data" / "docs"
    chunks = load_documents(docs_dir)
    print(f"[smoke] loaded {len(chunks)} chunks from {docs_dir}")
    if not chunks:
        print("[smoke] FAIL: no documents found")
        return 1

    embedder = OpenRouterEmbedder()
    index = HybridIndex.build(chunks, embedder)
    print(f"[smoke] built hybrid index: {len(index)} chunks, embed={index.model_name}\n")

    for q in QUERIES:
        hits = hybrid_search(index, q, embedder, top_k=3, method="rrf")
        print(f"Q: {q}")
        for i, h in enumerate(hits, start=1):
            print(
                f"  [{i}] {h.chunk.source} fused={h.score:.4f} "
                f"dense#{h.dense_rank}({h.dense_score:.3f}) bm25#{h.sparse_rank}({h.sparse_score:.2f})"
            )
        print(f"      top preview: {hits[0].chunk.text[:120].replace(chr(10), ' ')}…\n")

    if get_api_key():
        print("[smoke] OPENROUTER_API_KEY found — testing one grounded answer:\n")
        q = QUERIES[0]
        hits = hybrid_search(index, q, embedder, top_k=4)
        print(f"Q: {q}\nA: ", end="", flush=True)
        try:
            for tok in stream_answer(q, hits):
                print(tok, end="", flush=True)
            print("\n\n[smoke] LLM OK")
        except Exception as exc:  # noqa: BLE001
            print(f"\n[smoke] LLM call failed: {exc}")
            return 2
    else:
        print("[smoke] no OPENROUTER_API_KEY — skipped LLM call (retrieval core verified).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
