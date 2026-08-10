<p align="center">
  <img src="assets/logo.png" alt="BeeSearch logo" width="160">
</p>

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-orange)](https://langchain-ai.github.io/langgraph/)
[![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-green)](https://ollama.ai)
[![License: PolyForm NC](https://img.shields.io/badge/License-PolyForm%20NC-blue)](https://polyformproject.org/licenses/noncommercial/1.0.0)

---

## Table of contents

- [What is BeeSearch?](#what-is-beesearch)
- [The four modes](#the-four-modes)
- [Get started in 3 steps](#get-started-in-3-steps)
- [GPU acceleration](#gpu-acceleration)
- [Managing Docker](#managing-docker)
- [Choosing an AI model](#choosing-an-ai-model)
- [Using the web interface](#using-the-web-interface)
- [Adjusting AI responses](#adjusting-ai-responses)
- [Settings reference](#settings-reference)
- [For developers](#for-developers)
- [Output files](#output-files)
- [Documentation](#documentation)
- [License](#license)

---

## What is BeeSearch?

BeeSearch is a free AI research tool that runs entirely on your own computer — no internet connection required after setup, no subscriptions, no data leaving your machine. It uses a local AI model to help you search published research papers, analyse documents you upload, and answer research questions with cited sources.

Everything runs locally via [Ollama](https://ollama.ai), an open-source tool that runs AI models on your hardware. BeeSearch automatically picks the right model for your computer.

---

## The four modes

| Mode | What it does |
|------|-------------|
| **1 — Systematic Review** | Searches Google Scholar, arXiv, Semantic Scholar, and CrossRef for papers on your topic; screens them; and produces a formatted review report (Word/PDF) with risk-of-bias ratings, contradiction summaries, citation graphs, trend analysis, and plain-language summaries. |
| **2 — Research Notebook** | Upload your own PDFs, Word docs, or web pages and chat with them. Get cross-document summaries, Q&A, mind maps, knowledge graphs, audio scripts, and more — all grounded in your documents with cited page references. |
| **3 — AI Research Assistant** | Ask a research question in plain English and get a cited answer drawn from published papers — no files to upload, no formal review workflow. |
| **4 — Paper Discovery** | Explore the academic neighborhood of any paper. **Similarity Graph** builds a Connected Papers–style force-directed map using bibliographic coupling and co-citation. **Discovery Network** grows a persistent collection incrementally via references, citations, recommendations, and author networks. (Web interface only; no Ollama required.) |

---

## Get started in 3 steps

The fastest way to run BeeSearch is with Docker — it handles everything automatically (Python, Node.js, the AI model). You only need Docker installed.

**Step 1 — Install Docker**

Download [Docker Desktop](https://docs.docker.com/get-started/get-docker/) for your platform and start it. That's the only prerequisite.

**Step 2 — Download BeeSearch**

```bash
git clone https://github.com/sarder-abedin/BeeSearch.git
cd BeeSearch
cp .env.example .env
```

The `.env` file holds your settings. The defaults work fine to get started.

**Step 3 — Start the app**

```bash
./scripts/start.sh --build
```

The script detects your GPU automatically (NVIDIA, AMD, or Apple Silicon) and uses the right configuration. The first run downloads the AI model (~2 GB) and builds the app — this takes 5–10 minutes. After that, starts take under a minute. The app opens at **http://localhost:8000**. Press **Ctrl-C** to stop.

> **macOS (Apple Silicon M1/M2/M3):** Docker cannot run Ollama natively on Apple Silicon. Install [Ollama](https://ollama.com/download) first — it starts automatically after installation. Pull your models once (`ollama pull llama3.2:3b && ollama pull nomic-embed-text`), then run `./scripts/start.sh --build`.

> **No Docker?** See [Local install (no Docker)](#local-install-no-docker) in the For developers section.

---

## GPU acceleration

BeeSearch runs on CPU by default. Adding a GPU lets Ollama offload model layers to VRAM, making responses significantly faster. If the model doesn't fully fit in VRAM, Ollama automatically splits it — GPU layers run on the card, the rest run on CPU+RAM, with no manual configuration required.

**GPU is detected automatically.** `./scripts/start.sh` checks for NVIDIA, AMD, and Apple Silicon in turn and picks the right Docker Compose configuration. All you need to do is install the prerequisites for your hardware.

### NVIDIA GPU

Install [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html), then run:

```bash
./scripts/start.sh --build
```

### AMD Radeon GPU (ROCm)

Install [ROCm drivers](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/). Then check which Docker variant you are running — it determines the setup path:

```bash
docker context show
# default        → Docker Engine  (full GPU passthrough)
# desktop-linux  → Docker Desktop (runs in a VM — native Ollama required)
```

**Docker Engine** (`default` context):

Add your user to the `docker` and `video` groups, then log out and back in:

```bash
sudo usermod -aG docker $USER
sudo usermod -aG video $USER
```

Then start BeeSearch:

```bash
./scripts/start.sh --build
```

> **Integrated GPU warning** — if you see `dropping ROCm device — no rocblas support for gfx target` in the logs, your CPU's integrated graphics is being skipped. It is harmless; Ollama uses your discrete Radeon card automatically.

> **Older AMD cards (Polaris / Vega / pre-RDNA)** — some cards need a GFX version hint. Check your card with `rocminfo | grep gfx`, then uncomment `HSA_OVERRIDE_GFX_VERSION` in `docker-compose.gpu-amd.yml`.

**Docker Desktop on Linux** (`desktop-linux` context):

Docker Desktop runs inside a VM and cannot pass `/dev/kfd` through. Install Ollama natively instead — ROCm is auto-detected if drivers are already installed:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

Then run `./scripts/start.sh --build`. GPU work happens inside native Ollama on the host — BeeSearch's web container just calls its API.

**GPU + RAM sharing** — Ollama automatically fits as many model layers as possible into VRAM and runs the rest on CPU+RAM. No manual tuning is needed. To reserve VRAM headroom for your desktop, set `OLLAMA_GPU_OVERHEAD` (bytes) in your `.env`:

```env
OLLAMA_GPU_OVERHEAD=2147483648   # reserve 2 GB for the OS/desktop
```

---

## Managing Docker

### Check container status

```bash
docker compose ps
```

Watch it update in real time (refreshes every 2 seconds):

```bash
watch docker compose ps
```

The `STATUS` column shows each container's state:

| Status | Meaning |
|--------|---------|
| `starting` | Health check hasn't passed yet — normal during the first ~30 seconds |
| `healthy` | Container is ready |
| `unhealthy` | Health check failed — check logs (see below) |

### View logs

Stream live logs from all containers:

```bash
docker compose logs -f
```

Stream logs from a single container:

```bash
docker compose logs -f ollama    # Ollama AI server
docker compose logs -f web       # BeeSearch web app
```

### Inspect a failing health check

If the ollama container is stuck on `unhealthy`, check the health check log directly:

```bash
docker inspect beesearch-ollama --format='{{json .State.Health}}' | python3 -m json.tool
```

This shows the last 5 health check attempts with exit codes and output, making it easy to see what's failing.

### Pull models manually

After the stack is healthy, pull any model into the running Ollama container:

```bash
docker compose exec ollama ollama pull nemotron3:33b
docker compose exec ollama ollama pull nomic-embed-text
```

Models are stored in `~/.ollama` on the host (bind-mounted into the container), so they survive restarts and are available to native Ollama as well.

### List available models

```bash
docker compose exec ollama ollama list
```

### Check model storage

Models persist in your home directory:

```bash
ls ~/.ollama/models/manifests/registry.ollama.ai/library/
```

### Stop and restart

```bash
docker compose down        # stop and remove containers
./scripts/start.sh         # start again (GPU is re-detected automatically)
```

`down` does **not** delete your models — they are in `~/.ollama` on the host.

---

## Choosing an AI model

BeeSearch automatically picks the best model for your computer based on available RAM. You can view what's recommended for your hardware with:

```bash
python main.py --check-system        # hardware summary + recommended model
python main.py --list-models         # table of all pulled models with RAM and context info
```

Or change it at any time in the **Settings panel (⚙)** → **LLM Model** dropdown. BeeSearch shows every pulled model and automatically adjusts context size and chunk settings when you pick one.

**Recommended models by RAM:**

| RAM | Model | Context | Notes |
|-----|-------|---------|-------|
| 4 GB+ | `llama3.2:3b` | 32k | Fastest, lowest memory use |
| 8 GB+ | `llama3.1:8b` | 32k | Reliable all-rounder |
| 12 GB+ | `mistral-nemo:12b` | 128k | Best context window |
| 16 GB+ | `qwen3:14b` | 32k | Excellent quality |
| 20 GB+ | `nemotron3:33b` | 4k | NVIDIA large model |
| 20 GB+ | `qwq:32b` | 32k | Deep reasoning |
| 20 GB+ | `qwen3:32b` | 32k | Top-tier quality |

To pull a model (Docker):

```bash
docker compose exec ollama ollama pull mistral-nemo:12b
```

To pull a model (local install / native Ollama):

```bash
ollama pull mistral-nemo:12b
```

Then select it in **Settings → LLM Model** — no restart needed.

**Embedding model** — used for document search in Research Notebook. `nomic-embed-text` is the default and works well for most use cases. To switch, pull an alternative and select it in **Settings → Embedding Model**:

```bash
docker compose exec ollama ollama pull mxbai-embed-large    # highest accuracy
docker compose exec ollama ollama pull bge-m3               # best for non-English docs
docker compose exec ollama ollama pull qwen3-embedding:0.6b # compact multilingual
```

---

## Using the web interface

Open **http://localhost:8000** after starting the app.

### Mode 1 — Systematic Review

Click **Systematic Review** on the home page. Enter your research goal, inclusion criteria (e.g. "peer-reviewed studies, human participants"), and exclusion criteria (e.g. "animal studies"). BeeSearch searches multiple databases, screens the results, and produces a full review report.

Output files are saved to the `outputs/` folder in the repository.

### Mode 2 — Research Notebook

Click **Research Notebook** on the home page. Create a notebook, upload your sources (PDFs, Word docs, web URLs), then use the tabs:

| Tab | What it does |
|-----|-------------|
| **Chat** | Ask questions; answers are grounded in your documents with cited page references |
| **Sources** | Upload and manage your documents |
| **Summary** | Cross-document synthesis; drill into any document section by section |
| **FAQ** | Auto-generated Q&A pairs across all your sources |
| **Literature Review** | Academic-style narrative synthesis of your documents |
| **Mind Map** | Visual concept map of the key ideas |
| **Knowledge Graph** | Entity-relationship diagram |
| **Citation Timeline** | Papers cited in your documents, organised by year |
| **Study Comparison** | Side-by-side comparison table of studies |
| **Pipeline** | Runs a 7-step automated analysis (summary → knowledge graph → study guide → podcast script, etc.) |
| **Research Report** | Structured report grounded in your documents, optionally enriched with web or arXiv sources |
| **Explain** | Plain-language explanations of your sources; automatically adapts if you rephrase or say you don't understand |
| **Reviewer** | IEEE-style peer review of a selected paper, grounded in the document *and* backed by external scientific literature. Searches arXiv and Semantic Scholar *before* writing the review; the resulting papers are cited inline as [E1]–[E9] within the critique text. Sections: Summary, Strengths, Weaknesses, and a Detailed Critique covering Novelty, Technical Soundness (mathematical correctness, logical validity, misleading claims — each backed by [En] citations), Experimental Evaluation, Related Work, Writing Quality, and a Recommendation (Accept / Minor Revision / Major Revision / Reject). Includes a follow-up chat that also references the same [En] papers. |

### Mode 3 — AI Research Assistant

Click **AI Research Assistant** on the home page. Type your question and click **Ask**. BeeSearch searches Google Scholar, arXiv, and Semantic Scholar, reads the results, and writes a cited answer.

### Mode 4 — Paper Discovery

Click **Paper Discovery** on the home page. No Ollama model is required — this mode queries the [Semantic Scholar Academic Graph API](https://api.semanticscholar.org/) directly.

**Similarity Graph** tab — enter a Semantic Scholar paper ID or title and click **Build Graph**. BeeSearch maps the paper's academic neighborhood using:
- **Bibliographic coupling** (Kessler, 1963) — two papers share references
- **Co-citation** (Small, 1973) — two papers are frequently cited together

The result is a force-directed graph where nodes are papers (colour = year, size = citation count). Click any node to see its abstract and metadata; click **Set as new origin** to re-centre the graph on that paper. Use the sliders to tune the balance between the two similarity measures and the number of candidates to consider.

**Discovery Network** tab — add one or more seed paper IDs or titles and click **Create Collection**. BeeSearch fetches the seed papers from Semantic Scholar and displays them as an initial graph. Click any node, then choose a relationship type and click **Expand**:

| Relationship | What it fetches |
|---|---|
| Earlier work (references) | Papers cited by this paper |
| Later work (citations) | Papers that cite this paper |
| Similar papers (recommended) | Semantic Scholar recommendations |
| Author network | Other papers by the same authors |

The collection grows with each expand — edges between newly added papers and existing ones are built automatically.

> **API rate limits** — the free Semantic Scholar tier allows roughly 100 requests/minute. For faster graphs or collections, add your key to `.env`: `SEMANTIC_SCHOLAR_API_KEY=your_key_here`. Keys are free at [semanticscholar.org](https://www.semanticscholar.org/product/api).

---

## Adjusting AI responses

You can change how BeeSearch writes its answers using the **Response Tuning** setting in the Settings panel (⚙ button). This applies to Research Notebook answers, summaries, and all analysis tools.

| Setting | What you get |
|---------|-------------|
| **Precise** | The same question always gives the same answer, word for word. Good for reproducible research. |
| **Focused** *(default)* | Answers stay close to your source material with minimal variation. Recommended for most users. |
| **Balanced** | More natural, varied phrasing while still grounded in your sources. |
| **Creative** | The most varied and exploratory answers. Useful for brainstorming, podcast scripts, and mind maps. |

You can change this setting at any time — it takes effect on your very next question without restarting.

---

## Settings reference

Copy `.env.example` to `.env` before starting. Most users don't need to change anything — BeeSearch picks sensible defaults based on your hardware.

```env
# Address of the Ollama AI server (default works with Docker setup)
OLLAMA_BASE_URL=http://localhost:11434

# Which AI model to use (BeeSearch auto-selects based on your RAM if not set)
OLLAMA_MODEL=llama3.1:8b

# Embedding model for document search in Research Notebook
EMBED_MODEL=nomic-embed-text

# How many pages before switching to a lighter PDF parser (lower on machines with < 8 GB RAM)
LARGE_DOC_PAGE_THRESHOLD=50

# Default answer style: precise | focused | balanced | creative
TEMPERATURE_LEVEL=focused

# Vision model for figure captioning in uploaded PDFs (optional)
# Docker: llava:7b is pulled automatically on first start — no manual steps.
# Local install: pull first (ollama pull llava:7b) then set the value below.
# Leave empty to skip figure extraction entirely (no errors, no overhead).
VISION_MODEL=llava:7b

# GPU type hint — set this when running in Docker so the Settings panel shows
# the correct accelerator instead of "CPU only". Set automatically by the GPU
# compose files; only needed if you run a custom setup.
# Values: nvidia | amd | apple_silicon | cpu
# GPU_TYPE=amd
```

Optional settings for higher API rate limits (leave blank if you don't have these):

```env
SEMANTIC_SCHOLAR_API_KEY=
CROSSREF_EMAIL=your@email.com
```

Optional — LLM observability with [Langfuse](https://langfuse.com) (traces every AI call with prompts, latency, and token counts):

```env
# 1. Start self-hosted Langfuse: docker compose -f docker-compose.langfuse.yml up -d
# 2. Open http://localhost:3000 → create account → Settings → API Keys
# 3. Paste the keys below. Leave blank to disable tracing (no overhead, no errors).
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=http://localhost:3000
```

---

## For developers

This section covers the CLI, manual installation, running the backend and frontend separately, and other developer tools. If you just want to use BeeSearch, the sections above are all you need.

### Local install (no Docker)

Requires Python 3.10+, Node.js 20+, and [Ollama](https://ollama.ai) installed and running.

```bash
# Pull the AI models
ollama pull llama3.1:8b
ollama pull nomic-embed-text     # required for document search in Research Notebook

# Create a virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate.bat
pip install -r requirements.txt

# Start the web interface (React + backend)
python -m uvicorn backend.app.main:app --reload --port 8000
# Then in a separate terminal:
cd frontend && npm install && npm run dev
# React app at http://localhost:5173 (proxies /api/* to the backend)
```

**Windows — PowerShell:**

```powershell
python -m venv .venv
# If you see an execution-policy error, run this once then re-open PowerShell:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn backend.app.main:app --reload --port 8000
```

> **Optional — Mind Map / Knowledge Graph rendering:** needs the system `dot` binary:
> `apt install graphviz` (Linux), `brew install graphviz` (macOS), or the
> [Graphviz Windows installer](https://graphviz.org/download/).
> Docker users get this automatically — it's already in the image.

### Using vision models for figure captioning

BeeSearch can caption figures (charts, diagrams, photographs) extracted from uploaded PDFs using a multimodal Ollama model. Captions are indexed as `[FIGURE]` text chunks and cited in answers like any other chunk.

**Docker** — no manual steps needed. The `vision-init` service pulls `llava:7b` automatically on first start, and `VISION_MODEL=llava:7b` is pre-set in the container:

```bash
docker compose up --build   # llava:7b is pulled automatically
```

**Local install** — pull a vision model first, then set `VISION_MODEL` in your `.env`:

```bash
# Recommended: llava:7b (~4 GB, works on 8 GB RAM)
ollama pull llava:7b

# Alternatives (larger models give better captions)
ollama pull llava:13b            # ~8 GB
ollama pull llama3.2-vision:11b  # ~7 GB, Meta's vision model
```

Then in `.env`:
```env
VISION_MODEL=llava:7b
```

**Opting out** — leave `VISION_MODEL` empty (the default) to skip figure extraction entirely. No errors, no overhead. Text and table extraction are unaffected.

**Supported model families** — any Ollama model that accepts image input works: `llava` (7b, 13b, 34b), `llama3.2-vision`, `llava-llama3`, `moondream`, `bakllava`. Run `ollama list` to see what you have installed.

---

### Web interface — manual startup

```bash
# Backend (from repo root, with virtualenv active)
python -m uvicorn backend.app.main:app --reload --port 8000

# Stub out AI calls for UI development (no Ollama needed)
BEESEARCH_MOCK_LLM=1 python -m uvicorn backend.app.main:app --reload --port 8000

# Frontend (in a separate terminal)
cd frontend
npm install
npm run dev
# Opens at http://localhost:5173 — proxies /api/* to the backend automatically
```

### Production build

```bash
cd frontend
npm run build     # type-checks + builds to frontend/dist/
npm run preview   # serves the built output at http://localhost:4173
```

### Tests

```bash
# Full backend suite (root tests/ + backend/tests/), from the repository root
python -m pytest -q

# Frontend
cd frontend
npm run test       # component tests (Vitest)
npm run lint       # ESLint
npx tsc --noEmit   # type-check only
npm run e2e        # Playwright E2E — auto-starts a mock backend and preview server
```

### CLI reference

#### System and model info

```bash
python main.py --check-system    # hardware summary + recommended model
python main.py --list-models     # table of all pulled models with RAM, context, and quality
```

#### Systematic Literature Review

```bash
# Basic review
python main.py --systematic-review \
  --goal "What is the effect of sleep deprivation on working memory?" \
  --inclusion "Peer-reviewed empirical studies" "Human participants" \
  --exclusion "Animal studies" "Review papers only"

# Generate Word + PDF reports with author info
python main.py --systematic-review --goal "..." \
  --sr-docx --sr-pdf \
  --sr-author "Dr. Jane Smith" --sr-institution "University of Oxford"

# Plain-language summaries (patient / policy / press / all)
python main.py --systematic-review --goal "..." --sr-plain-language all

# Trend analysis + preprint tracking + concept drift
python main.py --systematic-review --goal "..." \
  --sr-trends --sr-preprints --sr-concept-drift

# Print risk-of-bias and contradiction results
python main.py --systematic-review --goal "..." --sr-quality

# Full combined run
python main.py --systematic-review \
  --goal "Efficacy of CBT for treatment-resistant depression" \
  --inclusion "RCTs" "Adult patients" \
  --exclusion "Children" "Open-label studies" \
  --sr-docx --sr-pdf \
  --sr-author "Dr. Smith" --sr-institution "MIT" \
  --sr-plain-language all \
  --sr-trends --sr-preprints --sr-concept-drift
```

#### AI Research Assistant

```bash
# Ask a question, get a literature-grounded answer with citations
python main.py --ask "Does intermittent fasting improve insulin sensitivity in adults?"

# Academic sources only (skip web search)
python main.py --ask "Transformer scaling laws" --no-web
```

#### Research Notebook

```bash
# New notebook
python main.py --notebook --notebook-name "Antibiotic Resistance"

# Open existing notebook
python main.py --notebook --notebook-id <id>

# Add files when opening
python main.py --notebook --notebook-id <id> --files paper.pdf notes.txt

# Document parsing options
python main.py --notebook --files paper.pdf                     # default (Docling)
python main.py --notebook --files paper.pdf --ocr               # Docling + OCR (scanned PDFs)
python main.py --notebook --files paper.pdf --no-docling        # always use pdfplumber
python main.py --notebook --files big.pdf --large-doc-threshold 30

# List all notebooks
python main.py --list-notebooks

# Advanced analysis (one-shot, by notebook ID)
python main.py --notebook-summary <id>
python main.py --notebook-faq <id>
python main.py --notebook-review <id>
python main.py --notebook-audio <id>
python main.py --notebook-mindmap <id>
python main.py --notebook-graph <id>
python main.py --notebook-compare <id> --compare-docs A.pdf B.pdf
python main.py --notebook-timeline <id>
python main.py --notebook-study-table <id>
python main.py --notebook-pipeline <id>

# Response tuning
python main.py --notebook --notebook-id <id> --temperature-level balanced
```

#### Section-by-Section Breakdown

```bash
python cli.py sections <notebook-id> --source paper.pdf
python cli.py sections <notebook-id> --source paper.pdf --level novice
python cli.py sections <notebook-id> --source paper.pdf --review
python cli.py sections <notebook-id> --source paper.pdf --review -o breakdown.md
```

| Flag | Default | Description |
|------|---------|-------------|
| `--source FILENAME` | interactive | Filename substring to match |
| `--level {novice,intermediate,expert}` | `intermediate` | Explanation depth |
| `--review` | off | Add expert reviewer critique per section |
| `-o / --output FILE` | none | Save output to a Markdown file |

#### Interactive notebook slash commands

While in `--notebook` mode:

```
/add <file>            Add a local document
/url <url>             Add a web page
/sources               List all sources
/summary               Cross-document summary
/faq                   FAQ generation
/review                Literature review
/audio                 Audio script + WAV synthesis
/mindmap               Mind map (DOT + PNG + SVG)
/graph                 Knowledge graph
/compare               Compare two sources
/timeline              Citation timeline
/study-table           Study comparison table
/temperature [level]   Show or change response tuning
/quit                  Exit
```

#### Running the CLI inside Docker

```bash
docker compose exec web python main.py --notebook --notebook-name "My Research"
docker compose exec web python main.py --list-notebooks
docker compose exec web bash   # open a shell
```

### MCP Server (optional)

`mcp_servers/research_tools_server.py` exposes BeeSearch's search and notebook tools over the [Model Context Protocol](https://modelcontextprotocol.io), so external MCP clients (Claude Code, Claude Desktop) can call them directly.

```bash
python mcp_servers/research_tools_server.py

# Or with the MCP inspector UI
mcp dev mcp_servers/research_tools_server.py
```

---

## Output files

All outputs are saved to `outputs/`:

| File | Contents |
|------|---------|
| `systematic_review_<id>.md` | Full review report in Markdown |
| `prisma_report_<id>.docx` | Review report as a Word document |
| `prisma_report_<id>.pdf` | Review report as a PDF |
| `summary_patient_<id>.txt` | Plain-language summary for patients |
| `summary_policy_<id>.txt` | Policy brief |
| `summary_press_<id>.txt` | Press release |
| `pipeline_study_guide_<name>.md/docx/pdf` | Study guide from the 7-agent pipeline |
| `pipeline_podcast_<name>.txt` | Podcast script |
| `knowledge_graph_<id>.dot/png/svg` | Knowledge graph |
| `mindmap_<id>.dot/png/svg` | Mind map |
| `citation_timeline_<id>.md` | Papers cited in your documents by year |
| `<name>_sections_<id>.md` | Section-by-section breakdown (CLI `--output`) |

---

## Documentation

Deeper technical documentation:

| Doc | Contents |
|-----|---------|
| [`docs/architecture.md`](docs/architecture.md) | Full pipeline diagrams, state field lists, file map, tech stack |
| [`docs/overview.md`](docs/overview.md) | Condensed architecture overview |
| [`docs/tutorial.md`](docs/tutorial.md) | Step-by-step walkthrough |
| [`docs/FAQ.md`](docs/FAQ.md) | Frequently asked questions |

---

## License

[PolyForm Noncommercial License 1.0.0](LICENSE) — free for personal, academic, and non-commercial use.
