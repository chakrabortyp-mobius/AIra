# 🚀 AIRA — AI Reasoning Agent

> A production-grade agentic AI system built from scratch — combining Retrieval-Augmented Generation (RAG), tool calling, hybrid retrieval, and parameter-efficient finetuning.

The project focuses on **clean architecture**, **extensibility**, and **real-world engineering practices** — not notebook-style experimentation. Every component is independently testable and built to production standards.

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-latest-orange)](https://langchain.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🎯 Project Objectives

- Build a **scalable and modular LLM framework** — from basic chat to full agentic workflows
- Support multiple stages of LLM systems:
  - Basic chat inference
  - Retrieval-Augmented Generation (RAG)
  - Agentic workflows with tool calling
  - Parameter-efficient finetuning (LoRA / QLoRA)
  - High-performance serving (vLLM — upcoming)
- Follow **industry-grade backend & MLOps patterns**:
  - Dependency injection
  - Centralized configuration management
  - Clear separation of concerns
  - API-first design

---

## 🧱 Architecture

```
Client
  │
  │  HTTP (JSON)
  ▼
FastAPI
  │
  ├── /v1/agent/chat ──► AgentExecutor (ReAct Loop)
  │                            │
  │                            ├──► document_search
  │                            │         └── Hybrid Retriever
  │                            │               ├── FAISS top-20  (dense)
  │                            │               ├── BM25  top-20  (sparse)
  │                            │               ├── RRF fusion    top-30
  │                            │               └── CrossEncoder  top-5
  │                            │
  │                            ├──► web_search ──► DuckDuckGo
  │                            │
  │                            └──► calculator ──► AST evaluator
  │
  ├── /v1/rag/chat ──► RAGChain ──► Retriever ──► LLM
  │
  └── /v1/chat ──► BasicChain ──► LLM
```

---

## 📁 Project Structure

```
AIra/
│
├── aira/
│   ├── main.py                  # FastAPI entrypoint + lifespan handler
│   ├── build_faiss.py           # offline script — builds FAISS index from PDFs
│   │
│   ├── api/
│   │   ├── agent.py             # POST /v1/agent/chat
│   │   ├── chat.py              # POST /v1/chat
│   │   ├── rag.py               # POST /v1/rag/chat
│   │   ├── documents.py         # POST /v1/documents/upload
│   │   ├── rerank.py            # POST /v1/rerank/test
│   │   └── health.py            # GET  /health
│   │
│   ├── agents/
│   │   ├── base_agent.py        # AgentExecutor + ReAct prompt + tools
│   │   └── tool_agent.py        # wrapper — clean .run() interface + step formatter
│   │
│   ├── core/
│   │   ├── config.py            # centralized config — models, paths, hyperparams
│   │   ├── dependencies.py      # singletons — LLM, retriever, chains, session store
│   │   ├── llm_loader.py        # loads Qwen3 via HuggingFace pipeline
│   │   └── prompt_manager.py    # prompt templates
│   │
│   ├── chains/
│   │   ├── basic_chain.py       # simple LLM chain
│   │   └── rag_chain.py         # RAG chain — context + question → answer
│   │
│   ├── rag/
│   │   ├── loader.py            # PDF loader
│   │   ├── semantic_chunker.py  # splits at meaning boundaries using embeddings
│   │   ├── embeddings.py        # MiniLM embedding model
│   │   ├── vectorstore.py       # FAISS wrapper (build, save, load)
│   │   ├── retriever.py         # dense FAISS retriever
│   │   ├── bm25_retriever.py    # sparse BM25 retriever
│   │   ├── hybrid_retriever.py  # RRF fusion of FAISS + BM25
│   │   ├── reranker.py          # CrossEncoder reranker
│   │   ├── summarizer.py        # map-reduce document summarizer
│   │   └── kb_metadata.py       # manages document descriptions as JSON
│   │
│   ├── tools/
│   │   ├── rag_tool.py          # RAG pipeline wrapped as LangChain Tool
│   │   ├── search.py            # DuckDuckGo web search tool
│   │   └── calculator.py        # safe AST-based math evaluator
│   │
│   └── finetune/
│       ├── lora_finetune.py     # LoRA/QLoRA on public HuggingFace datasets
│       └── merge_adapter.py     # merges LoRA adapter into base model
│
├── data/
│   ├── faiss/                   # FAISS index (built by build_faiss.py)
│   ├── kb_metadata.json         # base document descriptions
│   └── finetune/
│       └── train.jsonl          # custom finetuning data (optional)
│
├── models/                      # LoRA adapters saved here after finetuning
├── rag/                         # put your base PDFs here
├── .env                         # API keys — never commit this
├── .gitignore
├── requirements.txt
├── README.md
└── PLAYBOOK.md                  # full build log and roadmap
```

---

## ✅ Implemented Features

### 🔹 LLM Inference Core
- Integrated `Qwen/Qwen3` via HuggingFace Transformers
- Clean abstraction for model loading, prompt creation, generation config
- Output post-processing — removes special tokens and `<think>...</think>` reasoning traces
- Automatic CPU/GPU detection via `torch.cuda.is_available()`

### 🔹 FastAPI Serving Layer
- REST API with versioned endpoints
- Request/response validation using Pydantic
- Swagger / OpenAPI docs auto-generated at `/docs`
- Dependency injection for chains, model lifecycle
- Centralized configuration in `config.py`
- FastAPI lifespan handler — startup and graceful shutdown

### 🔹 Hybrid Retrieval Pipeline
- **FAISS** — dense vector similarity search (top-20)
- **BM25** — sparse keyword matching (top-20)
- **RRF (Reciprocal Rank Fusion)** — merges both ranked lists (top-30)
- **CrossEncoder reranker** — final scoring of candidates (top-5)
- **Semantic chunking** — splits documents at topic boundaries using cosine similarity between sentence embeddings, not fixed character counts

### 🔹 Document Management
- Build a permanent FAISS index from PDFs via `build_faiss.py`
- Upload PDFs at runtime via API — no server restart needed
- **Session-scoped uploads** — uploaded docs live in-memory only, wiped on shutdown
- **Map-reduce summarization** — each PDF gets a 4-8 word description for agent awareness
- Dynamic RAG tool description — agent always knows what topics are in the knowledge base

### 🔹 Agentic Workflow
- **ReAct prompting** (Reason + Act loop) — agent thinks before acting
- Three tools: `document_search`, `web_search`, `calculator`
- Agent picks the right tool automatically based on the question
- Full tool call trace returned in every API response
- LangSmith integration for end-to-end observability

### 🔹 Finetuning Pipeline
- **LoRA** — low-rank adapter injection, trains ~1% of parameters
- **QLoRA** — 4-bit NF4 quantization (bitsandbytes) + LoRA, runs on 6GB VRAM
- Supports 4 public HuggingFace datasets out of the box
- Adapter merge script for clean deployment

---

## 🤖 Models Used

| Model | Purpose |
|---|---|
| `Qwen/Qwen3-0.6B` | LLM for answering, summarizing, agent reasoning |
| `sentence-transformers/all-MiniLM-L6-v2` | Document + query embeddings |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | CrossEncoder reranking |

All models download automatically from HuggingFace on first run.

---

## 🔌 API Reference

### Agent Chat
```
POST /v1/agent/chat
```
```json
// Request
{ "question": "What is the structure of a scientific article?" }

// Response
{
  "question": "What is the structure of a scientific article?",
  "answer": "A scientific article follows the AIMReDCaR structure...",
  "tool_calls": [
    {
      "tool": "document_search",
      "input": "structure of a scientific article",
      "output": "Retrieved 5 relevant chunks..."
    }
  ],
  "tools_used": ["document_search"]
}
```

### RAG Chat
```
POST /v1/rag/chat
```
```json
{ "question": "What is AIMReDCaR?" }
```

### Basic Chat
```
POST /v1/chat
```
```json
{ "question": "Explain attention mechanism in simple terms." }
```

### Upload Document
```
POST /v1/documents/upload
```
```bash
curl -X POST http://localhost:8000/v1/documents/upload \
  -F "file=@/path/to/document.pdf"
```
```json
{
  "filename": "document.pdf",
  "chunks_added": 18,
  "description": "machine learning research paper",
  "message": "Successfully added 'document.pdf' to the session knowledge base."
}
```

> **Note:** Uploaded PDFs are session-scoped — fully searchable immediately but wiped on server shutdown by design.

### List Documents
```
GET /v1/documents/list
```

---

## 🧪 Running Locally

### 1. Clone and set up environment

```bash
git clone https://github.com/YOUR_USERNAME/AIRA.git
cd AIRA

python -m venv .venv
source .venv/bin/activate        # Linux / Mac
.venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 2. Configure

Edit `aira/core/config.py`:
```python
MODEL_NAME = "Qwen/Qwen3-0.6B"       # or Qwen3-4B for better quality
RAG_DOC    = "/path/to/your/pdfs"     # folder with your base PDFs
```

Create `.env` in project root:
```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-key
LANGCHAIN_PROJECT=AIRA
```

Get a free LangSmith key at [smith.langchain.com](https://smith.langchain.com)

### 3. Build the FAISS index

```bash
python -m aira.build_faiss
```

Runs once — loads PDFs, chunks semantically, summarizes each document, builds vector index.

### 4. Start the server

```bash
uvicorn aira.main:app --reload
```

### 5. Open Swagger UI

```
http://127.0.0.1:8000/docs
```

---

## 🔧 Finetuning

### Install dependencies
```bash
pip install peft bitsandbytes trl datasets accelerate
```

### Run on a public dataset

```bash
# Alpaca — 52K general instruction examples (recommended first run)
python -m aira.finetune.lora_finetune --mode qlora --dataset alpaca

# Dolly — 15K QA examples (best for RAG use case)
python -m aira.finetune.lora_finetune --mode qlora --dataset dolly

# Custom data
python -m aira.finetune.lora_finetune --mode qlora --dataset custom \
    --data data/finetune/train.jsonl
```

### Supported datasets

| Flag | Dataset | Size | Best for |
|---|---|---|---|
| `alpaca` | tatsu-lab/alpaca | 52K | General instruction following |
| `guanaco` | timdettmers/openassistant-guanaco | 9K | Conversational quality |
| `dolly` | databricks/databricks-dolly-15k | 15K | QA and summarization |
| `oasst` | OpenAssistant/oasst_top1_2023 | ~4K | Helpful assistant behavior |

### Merge adapter for deployment

```bash
python -m aira.finetune.merge_adapter \
    --adapter models/lora_adapter/final_adapter \
    --output  models/merged_model
```

Then update `config.py`:
```python
MODEL_NAME = "models/merged_model"
```

### Hardware requirements

| Model | Mode | Min VRAM |
|---|---|---|
| Qwen3-0.6B | QLoRA | 2GB |
| Qwen3-0.6B | LoRA | 4GB |
| Qwen3-4B | QLoRA | 6GB |
| Qwen3-4B | LoRA | 16GB |

---

## 🧠 Design Philosophy

> **APIs call chains, chains call models**

- No global model state — everything injected via `dependencies.py`
- Loose coupling, high cohesion — every module has one job
- Easy to extend without refactoring — add a new tool, chain, or retriever without touching existing code
- Production-first mindset — logging, error handling, graceful shutdown built in from the start

---

## 🔮 Planned Enhancements

- [ ] **Conversational Memory** — multi-turn chat with session IDs
- [ ] **HyDE** — Hypothetical Document Embeddings for better retrieval
- [ ] **Query Expansion** — rewrite queries into multiple phrasings
- [ ] **React Frontend** — streaming chat UI with source and tool trace panels
- [ ] **Docker** — containerize FastAPI + frontend
- [ ] **AWS Deployment** — EC2 GPU instance with public URL
- [ ] **MLflow** — experiment tracking for retrieval and generation quality
- [ ] **DVC** — version control for FAISS index and datasets
- [ ] **vLLM** — high-throughput GPU inference (10-20x faster than HuggingFacePipeline)
- [ ] **CI/CD** — GitHub Actions for automated test, build, deploy

See [PLAYBOOK.md](PLAYBOOK.md) for the detailed roadmap with implementation order.

---

## 📦 Requirements

```
Python 3.10+
torch
transformers
langchain
langchain-community
langsmith
faiss-cpu
sentence-transformers
rank-bm25
fastapi
uvicorn
loguru
ddgs
peft
bitsandbytes
trl
datasets
accelerate
python-dotenv
```

---

## 📄 License

MIT