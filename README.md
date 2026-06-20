# Hybrid Search RAG over Internal Docs

A live, **cloud-hosted** chat-with-your-docs web app. Anyone can upload PDFs /
Markdown / text and ask questions — answers are **grounded in the documents
with inline citations**. Retrieval is **hybrid**: it fuses keyword search
(BM25) with semantic search (NVIDIA Nemotron embeddings) so it catches both
exact-term and meaning-based matches. Everything runs in the cloud on
**OpenRouter free models** — no local GPU, no local ML libraries.

🔗 **Live demo:** https://saitejamothukuri-hybrid-search-rag.hf.space
&nbsp;·&nbsp; Space page: https://huggingface.co/spaces/SaitejaMothukuri/hybrid-search-rag

```
┌──────────┐   ┌──────────────┐   ┌──────────────────────────┐   ┌─────────────┐
│ Upload   │──▶│ Chunk +      │──▶│ Hybrid retrieve          │──▶│ OpenRouter  │
│ pdf/md/  │   │ embed (API)  │   │ BM25 ⊕ dense  (RRF / α)  │   │ chat LLM    │
│ txt      │   │ session-only │   │ → top-k passages         │   │ + citations │
└──────────┘   └──────────────┘   └──────────────────────────┘   └─────────────┘
```

| | Model (OpenRouter, free tier) |
|---|---|
| **Embeddings** | `nvidia/llama-nemotron-embed-vl-1b-v2:free` (2048-dim) |
| **Chat** | `nvidia/nemotron-3-nano-30b-a3b:free` |

## Why hybrid?

| Query style | BM25 (sparse) | Embeddings (dense) | Hybrid |
|---|---|---|---|
| Exact term / ID / acronym | ✅ strong | ⚠️ can miss | ✅ |
| Paraphrase / synonym / concept | ⚠️ misses | ✅ strong | ✅ |

The two ranked lists merge with **Reciprocal Rank Fusion** (`score = Σ 1/(k + rank)`),
a scale-free, rank-based combiner. A `weighted` mode (min-max normalize, then an
`alpha` blend of dense vs sparse) is also exposed so the trade-off is visible live.

## Multi-user by design

Uploaded documents are read **in memory** and the index is held in
**`st.session_state`** — scoped to one browser session. Two visitors never see
each other's documents. The built-in "Northwind" sample docs are a shared,
read-only demo corpus so the app is usable the moment it loads.

## Run locally

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# add your OpenRouter key (see below), then:
streamlit run app.py        # http://localhost:8501
```

Verify the pipeline headlessly (embeddings + retrieval + one grounded answer):

```powershell
python scripts/smoke.py
```

### Where to put your OpenRouter API key

Get a free key at <https://openrouter.ai/keys>. Locally, use any one of:

- **`.streamlit/secrets.toml`** (recommended, gitignored):
  ```toml
  OPENROUTER_API_KEY = "sk-or-v1-..."
  ```
- **`.env`** file: `OPENROUTER_API_KEY=sk-or-v1-...`
- env var: `setx OPENROUTER_API_KEY "sk-or-v1-..."`

In the cloud, set it as a platform **secret** (see Deploy). The key is never
committed — `.gitignore` excludes `.streamlit/secrets.toml` and `.env`.

Override models with `OPENROUTER_MODEL` / `OPENROUTER_EMBED_MODEL`.

## Deploy (Hugging Face Spaces, free)

One command — creates the Space (Docker SDK), sets the key as a Space secret,
and uploads the app:

```powershell
python scripts/deploy_hf.py --space-name hybrid-search-rag
```

(Needs a Hugging Face token via `huggingface-cli login` or `HF_TOKEN`.) The
included `Dockerfile` runs Streamlit on port 7860. Streamlit Cloud also works —
point it at this repo and add `OPENROUTER_API_KEY` in the app's secrets.

## How it works

| Stage | File | What it does |
|---|---|---|
| Ingest | `rag/ingest.py` | Read pdf/md/txt (disk **or** in-memory bytes), paragraph-aware chunking with overlap |
| Embed | `rag/embed.py` | OpenRouter embeddings, batched, L2-normalized |
| Index | `rag/index.py` | BM25 (rank-bm25) + dense vectors |
| Retrieve | `rag/retrieve.py` | RRF or weighted fusion → top-k with per-method ranks |
| Generate | `rag/llm.py` | OpenRouter chat, grounded prompt, streamed answer + `[n]` citations |
| UI | `app.py` | Streamlit chat, session-scoped uploads, retrieval controls, source inspector |

Every answer ships with an inspectable **sources** panel showing each passage's
fused score and its dense / BM25 rank. The system prompt forbids answering
outside the retrieved context — if the docs don't cover it, the model says so.

## Project layout

```
hybrid-search-rag/
├── app.py                       # Streamlit chat UI (session-scoped uploads)
├── rag/
│   ├── ingest.py                # load + chunk (disk + in-memory)
│   ├── embed.py                 # OpenRouter embeddings client
│   ├── index.py                 # BM25 + dense hybrid index
│   ├── retrieve.py              # RRF / weighted fusion
│   └── llm.py                   # OpenRouter chat + grounded prompt
├── data/docs/                   # sample demo docs
├── scripts/
│   ├── smoke.py                 # headless pipeline test
│   └── deploy_hf.py             # one-command HF Spaces deploy
├── Dockerfile                   # HF Spaces (docker SDK), Streamlit on :7860
├── requirements.txt             # no torch — fully API-backed
└── .gitignore                   # excludes secrets, .env, venv
```

## License

MIT
