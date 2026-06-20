"""Hybrid Search RAG — chat with your docs (fully cloud, OpenRouter).

Run locally:   streamlit run app.py
Deployed:      Hugging Face Spaces / Streamlit Cloud (set OPENROUTER_API_KEY as a secret)

Uploads are processed in-memory and scoped to your browser session — one
visitor never sees another visitor's documents.
"""

from __future__ import annotations

import streamlit as st

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # noqa: BLE001
    pass

from pathlib import Path

from rag.embed import OpenRouterEmbedder, get_embed_model
from rag.index import HybridIndex
from rag.ingest import chunk_uploaded, load_documents
from rag.llm import get_api_key, get_model, stream_answer
from rag.retrieve import hybrid_search

SAMPLES_DIR = Path(__file__).parent / "data" / "docs"

st.set_page_config(page_title="Hybrid Search RAG", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner=False)
def _embedder() -> OpenRouterEmbedder | None:
    try:
        return OpenRouterEmbedder()
    except Exception:  # noqa: BLE001 - no key yet
        return None


@st.cache_resource(show_spinner="Indexing sample docs…")
def _sample_index() -> HybridIndex | None:
    """Static demo corpus — shared across sessions (read-only, identical for all)."""
    emb = _embedder()
    if emb is None:
        return None
    chunks = load_documents(SAMPLES_DIR)
    return HybridIndex.build(chunks, emb) if chunks else None


def _build_user_index() -> None:
    """(Re)build the session-private index from this visitor's uploaded files."""
    emb = _embedder()
    chunks = st.session_state.get("user_chunks", [])
    st.session_state.user_index = (
        HybridIndex.build(chunks, emb) if (emb and chunks) else None
    )


# --------------------------------------------------------------------------- UI
st.title("📚 Hybrid Search RAG over Internal Docs")
st.caption(
    "Upload your docs and chat with them. Retrieval fuses **BM25** (keyword) with "
    "**Nemotron embeddings** (semantic) via Reciprocal Rank Fusion; answers come from "
    "a free OpenRouter model with inline citations."
)

if "user_chunks" not in st.session_state:
    st.session_state.user_chunks = []
    st.session_state.user_files = []
if "user_index" not in st.session_state:
    st.session_state.user_index = None
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("⚙️ Setup")
    has_key = bool(get_api_key())
    if has_key:
        st.success("OpenRouter key loaded")
        st.caption(f"💬 chat: `{get_model()}`")
        st.caption(f"🔢 embed: `{get_embed_model()}`")
    else:
        st.error("No OpenRouter API key found.")
        st.markdown(
            "Set `OPENROUTER_API_KEY` as an environment variable / Space secret, "
            "or in `.streamlit/secrets.toml`. Get one free at "
            "[openrouter.ai/keys](https://openrouter.ai/keys)."
        )

    st.divider()
    st.subheader("📄 Your documents")
    st.caption("Uploads stay private to your session.")
    uploads = st.file_uploader(
        "Upload PDF / Markdown / TXT",
        type=["pdf", "md", "markdown", "txt"],
        accept_multiple_files=True,
    )
    if uploads and st.button("➕ Index my files", use_container_width=True, disabled=not has_key):
        with st.spinner("Embedding your documents…"):
            have = set(st.session_state.user_files)
            for up in uploads:
                if up.name in have:
                    continue
                st.session_state.user_chunks.extend(chunk_uploaded(up.name, up.getvalue()))
                st.session_state.user_files.append(up.name)
            _build_user_index()
        st.success(f"Indexed {len(st.session_state.user_files)} file(s).")
    if st.session_state.user_files:
        st.write("\n".join(f"- {n}" for n in st.session_state.user_files))
        if st.button("🗑️ Clear my docs", use_container_width=True):
            st.session_state.user_chunks = []
            st.session_state.user_files = []
            st.session_state.user_index = None
            st.rerun()

    st.divider()
    st.subheader("🔎 Retrieval")
    has_uploads = st.session_state.user_index is not None
    corpus = st.radio(
        "Corpus",
        ["My uploaded docs", "Sample docs"],
        index=0 if has_uploads else 1,
        help="Sample docs are a built-in demo. Upload your own to chat with them.",
    )
    top_k = st.slider("Passages (top-k)", 1, 10, 5)
    method = st.radio("Fusion", ["rrf", "weighted"], horizontal=True)
    alpha = st.slider("Dense ⟷ Sparse (alpha)", 0.0, 1.0, 0.5, 0.05) if method == "weighted" else 0.5
    temperature = st.slider("LLM temperature", 0.0, 1.0, 0.2, 0.05)

# Pick the active index
if corpus == "My uploaded docs":
    index = st.session_state.user_index
else:
    index = _sample_index() if has_key else None

if not has_key:
    st.info("👈 Add your OpenRouter API key to start. The app needs it for both embeddings and chat.")
elif index is None:
    if corpus == "My uploaded docs":
        st.info("👈 Upload one or more documents and click **Index my files** to chat with them.")
    else:
        st.warning("Sample index unavailable — check your API key / connectivity.")
else:
    label = "your documents" if corpus == "My uploaded docs" else "the sample docs"
    st.caption(f"Chatting with **{label}** · {len(index)} chunks · embed `{index.model_name}`")

# Replay history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander(f"📎 {len(m['sources'])} sources"):
                for i, s in enumerate(m["sources"], start=1):
                    st.markdown(
                        f"**[{i}] {s['source']}** · fused={s['score']:.4f} · "
                        f"dense#{s['dense_rank']} ({s['dense_score']:.3f}) · "
                        f"bm25#{s['sparse_rank']} ({s['sparse_score']:.2f})"
                    )
                    st.caption(s["preview"])

prompt = st.chat_input("Ask a question about your documents…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not has_key:
            st.warning("Set your OpenRouter API key first.")
            st.stop()
        if index is None:
            st.warning("No documents indexed yet.")
            st.stop()

        with st.spinner("Searching documents…"):
            retrieved = hybrid_search(
                index, prompt, _embedder(), top_k=top_k, method=method, alpha=alpha
            )
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
            if m["role"] in ("user", "assistant")
        ]
        try:
            answer = st.write_stream(
                stream_answer(prompt, retrieved, history=history, temperature=temperature)
            )
        except Exception as exc:  # noqa: BLE001
            answer = f"⚠️ LLM request failed: {exc}"
            st.error(answer)

        sources = [
            {
                "source": r.chunk.source,
                "score": r.score,
                "dense_score": r.dense_score,
                "sparse_score": r.sparse_score,
                "dense_rank": r.dense_rank,
                "sparse_rank": r.sparse_rank,
                "preview": (r.chunk.text[:300] + "…") if len(r.chunk.text) > 300 else r.chunk.text,
            }
            for r in retrieved
        ]
        with st.expander(f"📎 {len(sources)} sources"):
            for i, s in enumerate(sources, start=1):
                st.markdown(
                    f"**[{i}] {s['source']}** · fused={s['score']:.4f} · "
                    f"dense#{s['dense_rank']} ({s['dense_score']:.3f}) · "
                    f"bm25#{s['sparse_rank']} ({s['sparse_score']:.2f})"
                )
                st.caption(s["preview"])

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
