"""OpenRouter chat client + grounded-answer prompt construction.

Uses the OpenAI SDK pointed at OpenRouter. The model defaults to the fast,
free Nemotron Nano; override with the ``OPENROUTER_MODEL`` env var.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .retrieve import RetrievedChunk

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"

SYSTEM_PROMPT = (
    "You are a precise assistant that answers questions strictly from the provided "
    "context passages, which are excerpts from the user's internal documents.\n"
    "Rules:\n"
    "1. Use ONLY the context. If the answer is not in the context, say you don't "
    "know based on the available documents — do not invent facts.\n"
    "2. Cite the passages you used with bracketed numbers like [1], [2] that match "
    "the passage numbers. Cite after the claim they support.\n"
    "3. Be concise and direct. Prefer specifics from the documents over generalities."
)


def get_api_key() -> str | None:
    """Resolve the OpenRouter key from env, then Streamlit secrets if available."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key.strip()
    try:
        import streamlit as st

        if "OPENROUTER_API_KEY" in st.secrets:
            return str(st.secrets["OPENROUTER_API_KEY"]).strip()
    except Exception:  # noqa: BLE001 - streamlit absent / no secrets file
        pass
    return None


def get_model() -> str:
    return os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL).strip()


def _client(api_key: str):
    from openai import OpenAI

    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "https://github.com/saiteja007-mv/hybrid-search-rag",
            "X-Title": "Hybrid Search RAG over Internal Docs",
        },
    )


def build_context_block(retrieved: "list[RetrievedChunk]") -> str:
    """Render numbered passages [1..n] for the prompt and for UI citation mapping."""
    lines = []
    for i, r in enumerate(retrieved, start=1):
        lines.append(f"[{i}] (source: {r.chunk.source})\n{r.chunk.text}")
    return "\n\n".join(lines)


def build_messages(query: str, retrieved: "list[RetrievedChunk]", history: list[dict] | None = None):
    context = build_context_block(retrieved)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history[-6:])  # short rolling memory
    messages.append(
        {
            "role": "user",
            "content": f"Context passages:\n\n{context}\n\n---\nQuestion: {query}",
        }
    )
    return messages


def stream_answer(
    query: str,
    retrieved: "list[RetrievedChunk]",
    history: list[dict] | None = None,
    api_key: str | None = None,
    model: str | None = None,
    temperature: float = 0.2,
) -> Iterator[str]:
    """Yield answer tokens from the LLM, grounded in the retrieved passages."""
    api_key = api_key or get_api_key()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set. See the README for where to put it.")
    client = _client(api_key)
    stream = client.chat.completions.create(
        model=model or get_model(),
        messages=build_messages(query, retrieved, history),
        temperature=temperature,
        stream=True,
    )
    for event in stream:
        delta = event.choices[0].delta.content if event.choices else None
        if delta:
            yield delta
