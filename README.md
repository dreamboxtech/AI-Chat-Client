# Local AI Chat Desktop App (PyQt + llama.cpp)

This project is a cross-platform desktop chat application for running **open-source large language models locally** using **llama.cpp**.
It is designed to work well on **average CPUs (≈8 GB RAM)**, with **GPU acceleration as an optional advantage**, not a requirement.

The focus is on **performance, stability, and clean architecture first**, before adding advanced features such as RAG, browsing, or multimodal input.

---

## Goals

- Run local GGUF models efficiently on CPU
- Keep the application **model-agnostic**
- Fast perceived response time (streaming, warm load)
- Clean separation of concerns (UI / engine / workers / settings)
- Cross-platform (Windows, Linux, macOS)

---

## Current Features

- **Desktop UI (PyQt6)**
  - Chat list (left)
  - Active chat view (right)
  - Multiple chat sessions
- **Local inference via llama.cpp**
  - Uses `llama-cpp-python`
  - Supports single and split `.gguf` files
- **Performance-oriented runtime**
  - Streaming token output
  - Model warm-loading
  - History trimming to prevent slowdown
- **Settings dialog**
  - Model selection (GGUF)
  - Context size, threads, batch size
  - GPU layers (optional)
  - mmap / mlock toggles
- **Model-agnostic design**
  - Uses the model’s embedded chat template when available
  - No hard-coded model behavior

---

## Supported Models (Current Focus)

The app works best with **CPU-friendly GGUF models**, especially Q4 / Q5 quantizations.

Tested models:
- MiniCPM-V-4.5 (Q4_0, GGUF)
- Qwen2.5-7B-Instruct (Q4 / Q5, GGUF)

> Not all GGUF files are compatible with llama.cpp. Only curated and verified models are recommended.

---

## Model Files (Important)

### Single vs Split GGUF

Some models are split into multiple files:

```
model-00001-of-00002.gguf
model-00002-of-00002.gguf
```

Rules:
- Do **not** merge the files
- Place all parts in the **same directory**
- Select **only** `00001-of-XXXX.gguf`
- Splitting does **not** affect performance

---

## Installation

### Recommended: Conda (conda-forge)

```bash
conda create -n localchat python=3.11
conda activate localchat
conda install -c conda-forge llama-cpp-python
pip install pyqt6
```

> Pip builds on Windows often fail due to missing C++ toolchains. Conda-forge provides prebuilt binaries.

---

## Running the App

```bash
python app.py
```

---

## Recommended Default Settings (CPU-First)

For an average PC (≈8 GB RAM, no GPU):

| Setting            | Value |
|--------------------|------:|
| Quantization       | Q4_K_M or Q4_0 |
| Context (`n_ctx`)  | 2048 |
| Batch (`n_batch`)  | 256 |
| Threads            | CPU cores − 1 |
| GPU layers         | 0 |
| Streaming          | ON |
| Keep last messages | 16 |

These defaults prioritize **fast first token**, **low memory usage**, and **stable performance**.

---

## Why Streaming Matters

Even on CPU, generation can take seconds.
Streaming:
- Reduces perceived latency
- Keeps the UI responsive
- Matches modern chat UX expectations

Streaming is enabled by default.

---

## Architecture Overview

```
app.py
├─ UI (PyQt)
│  ├─ MainWindow
│  ├─ SettingsDialog
│
├─ Engine
│  └─ LlamaCppEngine
│     ├─ load_if_needed()
│     ├─ chat_blocking()
│     └─ chat_stream()
│
├─ Workers (QThread)
│  ├─ GenerateWorker
│  └─ StreamWorker
│
└─ Models
   ├─ AppSettings
   └─ ChatSession
```

- Model is loaded once and reused
- Inference runs outside the UI thread
- Chat history is trimmed to avoid slowdown over time

---

## Known Limitations (Intentional)

- No Transformers backend (yet)
- No RAG or browsing (yet)
- No multimodal input UI (image input later)
- No packaging (.exe / .app) yet

---

## Roadmap

1. CPU speed auto-tuning presets
2. GPU capability detection and safe offload presets
3. Chat persistence (SQLite)
4. RAG (local document ingestion)
5. Controlled web browsing
6. Multimodal input support
7. Application packaging

---

## Design Philosophy

- Assume **no GPU**
- Curate models rather than supporting everything blindly
- Prioritize performance and reliability over features

Everything else builds on top of this.
