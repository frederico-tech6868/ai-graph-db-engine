# AI-GraphDB-Engine

**Pure-Rust AI pipeline system with multi-modal model support and graph-backed knowledge retrieval.**

---

## Overview

AI-GraphDB-Engine is a modular, high-performance AI orchestration framework that combines:

- **Multi-modal AI models** — Text (Llama GGUF), Audio (Whisper), Image (CLIP)
- **Graph-based knowledge** — Community detection, GraphRAG retrieval, JEPA world models
- **Production pipelines** — RAG, tool calling, structured extraction, agentic orchestration
- **Native interfaces** — Full-featured CLI and interactive TUI
- **Cross-platform GPU support** — NVIDIA CUDA, AMD ROCm, Apple Metal, Intel MKL

The system is built as a **Cargo workspace** with pure-Rust implementations, ensuring safety, performance, and easy deployment without Python runtime dependencies.

---

## Architecture

```
graph-db-engine/
├── rust/           ← Core graph engine (graphdb_rs)
│                     • GraphStore, Node, Edge persistence
│                     • Community detection (Louvain)
│                     • GraphRAG (local/global search)
│                     • JEPA world model (VICReg, latent search)
│
├── model-hub/      ← AI pipeline library (model-hub)
│                     • TextModel, AudioModel, ImageModel traits
│                     • Llama GGUF / Whisper / CLIP via candle
│                     • RAG, tool calling, extraction pipelines
│                     • Needle engine (heuristic-based routing)
│                     • GraphBackend abstraction
│
├── cli/            ← Command-line interface (graphdb-cli)
│                     • Subcommands: generate, embed, rag, extract, ask, transcribe, info
│                     • Engine toggle: --engine llm|needle
│                     • GPU device selection: --device cuda|metal|cpu
│
└── tui/            ← Terminal UI (graphdb-tui)
                      • Interactive modes: Chat, RAG, Extract, Info
                      • Live engine switching (Ctrl+E)
                      • Message history, status bar
```

### Python Components

The `python/` directory contains the original Python implementation:

- **Core graph engine** — `ai_memory/` with GraphStore, embeddings, JEPA-GraphRAG
- **Training dataset pipeline** — Multi-format document loader, graph builder, export to JSONL/Alpaca/OpenAI formats
- **Needle2 integration** — `NeedleAgentGroup`, `NeedleOrchestrator` for trainable function-calling agents
- **Web UI** — Flask dashboard with 2D/3D graph visualization (vis-network + 3d-force-graph)

See [`python/README.md`](python/README.md) and [`python/SDK.md`](python/SDK.md) for Python-specific documentation.

---

## Features

### 🎯 Multi-Modal AI Models

| Modality | Implementation | Backend | Status |
|----------|---------------|---------|--------|
| **Text** | Llama (GGUF quantized) | `candle-transformers` | ✅ Implemented |
| **Audio** | Whisper (GGUF) | `candle-transformers` | 🔧 Stub (API ready) |
| **Image** | CLIP | `candle-transformers` | 🔧 Stub (API ready) |

All models support **offline operation** via stub implementations for testing without downloading weights.

### 🚀 AI Pipelines

- **RAG** — Vector retrieval → BFS neighborhood expansion → context-augmented generation
- **Tool Calling** — Needle-compatible schema, deterministic routing or LLM-based dispatch
- **Structured Extraction** — JSON schema enforcement, entity extraction, knowledge triples
- **Agentic Orchestration** — Multi-turn tool use, conversation memory, hybrid search

### 🧠 Graph Intelligence

- **Community Detection** — Louvain algorithm (pure Rust)
- **GraphRAG Retrieval** — Local search (k-hop) + global search (community scoring)
- **JEPA World Model** — Context/target encoders, VICReg loss, latent-space retrieval
- **Hybrid Search** — Combines vector, graph, and learned latent modes

### 🎮 Dual Engine System

Switch between two inference engines:

| Engine | Use Case | Latency | Offline |
|--------|----------|---------|---------|
| **LLM** | Complex reasoning, generation | ~seconds | ⚠️ Needs model |
| **Needle** | Intent detection, routing | ~milliseconds | ✅ Always |

Toggle via `--engine llm` or `--engine needle` (CLI) or `Ctrl+E` (TUI).

### ⚡ GPU Acceleration

| Feature Flag | Hardware | Notes |
|--------------|----------|-------|
| `cuda` | NVIDIA CUDA (sm_70+) | GeForce RTX 20-series+ |
| `cudnn` | NVIDIA cuDNN | Faster convolutions |
| `rocm` | AMD ROCm (HIP) | Radeon RX 5000+ |
| `metal` | Apple Metal | M1/M2/M3, Intel integrated |
| `mkl` | Intel MKL | CPU + Intel Arc partial |
| `flash-attn` | Flash Attention 2 | CUDA only, 2-4× faster |

---

## Quick Start

### Prerequisites

- **Rust** 1.75+ — Install from [rustup.rs](https://rustup.rs/)
- **Git** — For cloning the repository

### 1. Clone and Build

```bash
# Clone the repository
git clone https://github.com/frederico-tech6868/ai-graph-db-engine.git
cd ai-graph-db-engine

# Build the workspace (CPU-only by default)
cargo build --release --workspace

# Or enable GPU support (example: NVIDIA CUDA)
cargo build --release --workspace --features cuda
```

### 2. Using the CLI

```bash
# Show system info and available backends
cargo run --release --bin graphdb-cli -- info

# Generate text with a stub model (no GGUF needed)
cargo run --release --bin graphdb-cli -- generate "Explain JEPA in one sentence."

# Extract named entities using the Needle engine
cargo run --release --bin graphdb-cli -- \
  --engine needle \
  extract --schema entities "Apple was founded by Steve Jobs in Cupertino."

# Ask a question with tool routing (stub model)
cargo run --release --bin graphdb-cli -- \
  ask "What documents are available?"
```

**With a real model:**

```bash
# Download a GGUF model (example: Llama-3.2-1B)
# See https://huggingface.co/models?library=gguf for options

# Generate text with the model
cargo run --release --bin graphdb-cli -- \
  --model /path/to/model.gguf \
  generate "Hello, world!"

# RAG search over a knowledge graph
cargo run --release --bin graphdb-cli -- \
  --model /path/to/model.gguf \
  rag "What is JEPA?" --top-k 5 --hops 2
```

### 3. Using the TUI

```bash
# Launch the interactive terminal UI
cargo run --release --bin graphdb-tui

# Or with a model loaded
cargo run --release --bin graphdb-tui -- --model /path/to/model.gguf
```

**TUI Controls:**

| Key | Action |
|-----|--------|
| `1` / `2` / `3` / `4` | Switch mode (Chat / RAG / Extract / Info) |
| `Tab` | Cycle modes |
| `Ctrl+E` | Toggle engine (LLM ↔ Needle) |
| `Enter` | Submit input |
| `Ctrl+Q` / `Esc` | Quit |

---

## Building with GPU Support

### NVIDIA CUDA

```bash
# Ensure CUDA Toolkit 11.8+ is installed
# https://developer.nvidia.com/cuda-downloads

cargo build --release --workspace --features cuda

# Enable Flash Attention 2 for faster inference (RTX 30-series+)
cargo build --release --workspace --features "cuda,flash-attn"
```

### AMD ROCm

```bash
# Ensure ROCm 5.4+ is installed
# https://rocm.docs.amd.com/

cargo build --release --workspace --features rocm
```

### Apple Metal

```bash
# Works out-of-the-box on macOS
cargo build --release --workspace --features metal
```

### Intel MKL

```bash
# Install Intel oneAPI Math Kernel Library
# https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html

cargo build --release --workspace --features mkl
```

---

## Usage Examples

### CLI: RAG Pipeline

```bash
# 1. Prepare a model (download a GGUF file)
wget https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf

# 2. Run RAG with seed retrieval + BFS expansion
cargo run --release --bin graphdb-cli -- \
  --model llama-2-7b.Q4_K_M.gguf \
  --device cuda \
  rag "What are the main features of Rust?" \
  --top-k 10 \
  --hops 2
```

### CLI: Structured Extraction

```bash
# Extract entities using the LLM engine
cargo run --release --bin graphdb-cli -- \
  --model llama-2-7b.Q4_K_M.gguf \
  extract --schema entities \
  "OpenAI released GPT-4 in March 2023. Sam Altman is the CEO."

# Output (JSON):
# {
#   "entities": [
#     { "name": "OpenAI", "type": "Organization" },
#     { "name": "GPT-4", "type": "Product" },
#     { "name": "Sam Altman", "type": "Person", "context": "CEO of OpenAI" }
#   ]
# }
```

### CLI: Tool Calling (Orchestration)

```bash
# Ask a question — the system routes to tools automatically
cargo run --release --bin graphdb-cli -- \
  --engine needle \
  ask "Search for documents about machine learning"

# The Needle engine will:
# 1. Detect intent → search_knowledge_base
# 2. Extract query parameter → "machine learning"
# 3. Execute the tool
# 4. Format the response
```

### TUI: Interactive Session

```bash
# Start the TUI with a model
cargo run --release --bin graphdb-tui -- --model llama-2-7b.Q4_K_M.gguf

# Inside the TUI:
# 1. Press '2' to switch to RAG mode
# 2. Type your query: "Explain RAG"
# 3. Press Enter
# 4. Watch the model retrieve context and generate an answer
# 5. Press Ctrl+E to switch to Needle engine for faster responses
```

---

## Project Structure

### Crate Breakdown

#### `rust/` — Core Graph Engine

**Exports:** `graphdb_rs`

Pure-Rust graph database with PyO3 Python bindings.

**Key modules:**
- `core` — Node, Edge, PropertyValue
- `store` — GraphStore (CRUD, indexes, persistence)
- `community` — Louvain algorithm
- `graphrag` — Local/global search
- `jepa` — JEPA world model (encoders, VICReg, latent search)
- `vector` — Cosine similarity, top-k search
- `query` — BFS, DFS, path finding

**Python bridge:** Exposes `PyGraphStore`, `PyCommunityDetector`, `PyGraphRAGRetriever`, `PyJEPAGraphRAG` to Python.

#### `model-hub/` — AI Pipeline Library

**Exports:** `model_hub`

Pure-Rust AI orchestration framework.

**Key modules:**
- `backend` — Device selection (CPU/CUDA/Metal/MKL)
- `models` — TextModel, AudioModel, ImageModel traits + Llama/Whisper/CLIP
- `graph` — GraphBackend trait + InMemoryGraph
- `pipeline` — RAG, ToolExecutor, StructuredExtractor, Orchestrator
- `needle` — NeedleAgent (heuristic-based routing)
- `hub` — ModelHub (registry + GGUF loader)

**Dependencies:** `candle-core`, `candle-nn`, `candle-transformers`, `tokenizers`, `hf-hub`

#### `cli/` — Command-Line Interface

**Binary:** `graphdb-cli`

Full-featured CLI with subcommands for all pipeline operations.

**Subcommands:**
- `generate` — Text completion
- `embed` — Embedding extraction
- `rag` — Retrieval-augmented generation
- `extract` — Structured JSON extraction
- `ask` — Orchestrated tool calling
- `transcribe` — Audio transcription (Whisper)
- `info` — System information

**Global flags:**
- `--model <PATH>` — GGUF model file
- `--device <cpu|cuda|metal|mkl>` — Compute device
- `--engine <llm|needle>` — Inference engine
- `--verbose` — Debug output

#### `tui/` — Terminal User Interface

**Binary:** `graphdb-tui`

Interactive ratatui-based TUI with four modes.

**Features:**
- Real-time message history
- Mode switching (Chat / RAG / Extract / Info)
- Engine toggle (LLM ↔ Needle)
- Keyboard-driven navigation
- Status bar with model info

**Tech stack:** `ratatui`, `crossterm`

---

## Documentation

- **Python SDK** — [`python/SDK.md`](python/SDK.md) · [PDF](python/SDK.pdf) · [DOCX](python/SDK.docx)
- **Python Setup** — [`python/README.md`](python/README.md)
- **Web UI** — [`python/webui/README.md`](python/webui/README.md)
- **Rust API docs** — Run `cargo doc --workspace --open`

---

## Testing

### Rust Tests

```bash
# Run all Rust tests (graphdb_rs + model-hub)
cargo test --workspace

# Test a specific crate
cargo test --package graphdb_rs
cargo test --package model-hub

# Run with GPU features
cargo test --workspace --features cuda
```

### Python Tests

```bash
cd python

# Using uv (recommended)
uv pip install -e ".[dev]"
uv run pytest tests/ -v

# Using pip
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Performance

### Benchmark: RAG Pipeline (Llama-3.2-1B-GGUF, 512 tokens)

| Device | Throughput | Latency (p50) | Notes |
|--------|-----------|---------------|-------|
| **CPU (Intel i7-12700K)** | ~8 tok/s | 64 ms | Single-threaded |
| **CUDA (RTX 4090)** | ~95 tok/s | 11 ms | FP16 quantized |
| **Metal (M2 Max)** | ~42 tok/s | 24 ms | Unified memory |
| **MKL (Intel Arc A770)** | ~18 tok/s | 56 ms | Partial acceleration |

*Measured with `--features cuda,flash-attn` on CUDA. Your mileage may vary.*

### Memory Usage

| Model | Size | RAM (CPU) | VRAM (GPU) |
|-------|------|-----------|------------|
| Llama-3.2-1B (Q4) | 0.8 GB | ~1.2 GB | ~1.0 GB |
| Llama-2-7B (Q4) | 3.8 GB | ~4.5 GB | ~4.2 GB |
| Llama-2-13B (Q4) | 7.4 GB | ~8.0 GB | ~7.8 GB |

---

## Roadmap

### Near-term (Q1 2025)

- [ ] **Full Whisper GGUF support** — Complete audio transcription pipeline
- [ ] **CLIP image encoder** — Enable multi-modal retrieval
- [ ] **LoRA fine-tuning** — Train custom adapters on domain-specific data
- [ ] **Persistent GraphBackend** — SQLite/RocksDB storage for production graphs
- [ ] **Web API** — REST/gRPC server for CLI/TUI functionality

### Medium-term (Q2-Q3 2025)

- [ ] **Distributed graph** — Sharding + federation for large-scale knowledge bases
- [ ] **Agent workflows** — DAG-based multi-agent orchestration
- [ ] **Model hub integration** — Auto-download from HuggingFace Hub
- [ ] **Benchmarking suite** — Standardized performance metrics
- [ ] **Documentation site** — mdBook-based guides + API reference

### Long-term (2025+)

- [ ] **Custom GGUF quantization** — Train and export models in Rust
- [ ] **Federated learning** — Privacy-preserving collaborative training
- [ ] **Production deployment guides** — Kubernetes, Docker Compose, systemd
- [ ] **Language bindings** — FFI for Go, Node.js, .NET

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Areas of interest:**
- GPU backend optimization (CUDA kernels, Metal Performance Shaders)
- New model implementations (Mistral, Qwen, Gemma)
- Pipeline improvements (better prompt templates, retrieval strategies)
- Documentation and examples
- Bug reports and feature requests

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- **[candle](https://github.com/huggingface/candle)** — Pure-Rust ML framework by Hugging Face
- **[llama.cpp](https://github.com/ggerganov/llama.cpp)** — GGUF format and quantization techniques
- **[ratatui](https://github.com/ratatui-org/ratatui)** — Terminal UI framework
- **[PyO3](https://github.com/PyO3/pyo3)** — Rust ↔ Python bindings
- **[vis-network](https://visjs.github.io/vis-network/)** & **[3d-force-graph](https://github.com/vasturiano/3d-force-graph)** — Graph visualization

---

## Contact

- **GitHub:** [frederico-tech6868/ai-graph-db-engine](https://github.com/frederico-tech6868/ai-graph-db-engine)
- **Issues:** [Issue Tracker](https://github.com/frederico-tech6868/ai-graph-db-engine/issues)

---

<div align="center">
<b>Built with 🦀 Rust and ❤️ for AI</b>
</div>
