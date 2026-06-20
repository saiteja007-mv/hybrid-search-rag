"""OpenRouter embeddings client (cloud-native, no local model / torch).

Uses the NVIDIA Llama-Nemotron embedding model on OpenRouter's free tier.
Embeddings are L2-normalized so cosine similarity is a plain dot product.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import numpy as np

from .llm import OPENROUTER_BASE_URL, get_api_key

DEFAULT_EMBED_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free"
EMBED_DIM = 2048  # llama-nemotron-embed-vl-1b-v2


def get_embed_model() -> str:
    return os.environ.get("OPENROUTER_EMBED_MODEL", DEFAULT_EMBED_MODEL).strip()


class OpenRouterEmbedder:
    """Minimal embeddings client with batching + L2 normalization.

    Exposes ``.encode(list[str]) -> np.ndarray`` so it is a drop-in for the
    rest of the pipeline (HybridIndex / hybrid_search call ``.encode``).
    """

    def __init__(self, api_key: str | None = None, model: str | None = None, batch_size: int = 64):
        self.api_key = api_key or get_api_key()
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set. See the README.")
        self.model_name = model or get_embed_model()
        self.batch_size = batch_size

    def _post(self, inputs: list[str]) -> list[list[float]]:
        body = json.dumps({"model": self.model_name, "input": inputs}).encode()
        req = urllib.request.Request(
            f"{OPENROUTER_BASE_URL}/embeddings",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/saiteja007-mv/hybrid-search-rag",
                "X-Title": "Hybrid Search RAG over Internal Docs",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:300].decode("utf-8", "ignore")
            raise RuntimeError(f"OpenRouter embeddings HTTP {exc.code}: {detail}") from exc
        # Preserve request order (OpenAI-compatible responses carry an index).
        rows = sorted(data["data"], key=lambda d: d.get("index", 0))
        return [r["embedding"] for r in rows]

    def encode(self, texts: list[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        vecs: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            vecs.extend(self._post(texts[i : i + self.batch_size]))
        arr = np.asarray(vecs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms
