# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

BeeSearch (repo: ResearchBuddy / BeeSearch) is a local-first AI research app with four
user-facing modes, built on LangGraph + Ollama (no cloud LLM, no API keys required):

- **Mode 1 — Systematic Literature Review**: PRISMA pipeline (Google Scholar, arXiv,
  Semantic Scholar, CrossRef → screening → PICO evidence extraction → quality assessment
  (risk of bias RoB 2/ROBINS-I, GRADE certainty, contradiction detection) → synthesis →
  PRISMA DOCX/PDF + plain-language summaries, trends, citation network with Smart Citation
  stance classification + citation-context snippets, concept drift).
- **Mode 2 — Research Notebook**: NotebookLM-style grounded chat over uploaded
  documents, plus an "Explain" storyteller tab, a Research Report tab, and a
  7-agent analysis pipeline (study guide, podcast, knowledge graph, etc.).
- **Mode 3 — AI Research Assistant**: stateless free-form question answering grounded in
  published literature with code-rebuilt inline citations (`agents/research_assistant.py`,
  `ui/tabs/research_assistant.py`, `main.py --ask`) — no upload, no PRISMA workflow.
- **Mode 4 — Paper Discovery** (web-only, no CLI): two sub-features backed by the
  Semantic Scholar Academic Graph API. **Similarity Graph** builds a Connected Papers–style
  force-directed map from a single origin paper using bibliographic coupling (Kessler, 1963)
  and co-citation (Small, 1973). **Discovery Network** lets the user grow a persistent paper
  collection by incrementally exploring earlier work (references), later work (citations),
  similar papers (S2 recommendations), or author networks.

All modes are reachable from the CLI (`main.py`, plus `cli.py` for the section-by-section
breakdown tool) and from the React + FastAPI web app (`frontend/` + `backend/`) — see
"React + FastAPI web app" under Architecture below.

For deep dives beyond this file: `docs/architecture.md` (full pipeline diagrams,
state field lists, file map, tech stack), `docs/overview.md` (condensed version),
`README.md` (install/usage/CLI reference/React+FastAPI web app), `docs/FAQ.md`,
`docs/tutorial.md`.

## Commands

```bash
# Setup
pip install -r requirements.txt
cp .env.example .env
ollama pull llama3.1:8b
ollama pull nomic-embed-text          # required for Hybrid RAG in Research Notebook

# Run — CLI
python main.py --check-system                          # hardware-aware model recommendation
python main.py --notebook --notebook-name "My Notes"   # Research Notebook session
python main.py --systematic-review --goal "..." \
  --inclusion "Peer-reviewed" --exclusion "Animal studies"
python cli.py sections <notebook-id> --source paper.pdf # section-by-section breakdown

# Run — React + FastAPI web app (two separate processes, local dev)
python -m uvicorn backend.app.main:app --reload --port 8000   # backend, http://localhost:8000
cd frontend && npm install && npm run dev                      # frontend, http://localhost:5173
BEESEARCH_MOCK_LLM=1 python -m uvicorn backend.app.main:app --reload --port 8000  # backend w/ stubbed LLM+search, no Ollama needed

# Docker (single command — React + FastAPI + Ollama, GPU auto-detected)
./scripts/start.sh --build   # all platforms — opens http://localhost:8000
```

### Tests

```bash
python -m pytest -q                                             # full suite: root tests/ + backend/tests/ together
python -m pytest tests/ -q                                      # core pipeline tests only (excludes backend/tests/)
python -m pytest backend/tests/ -q                              # FastAPI backend tests only
python -m pytest tests/test_temperature_levels.py -q            # one file
python -m pytest tests/test_temperature_levels.py::test_precise_forces_full_determinism -q  # one test
python -m py_compile path/to/file.py                             # syntax check, no deps needed
cd frontend && npm run test                                      # frontend component tests (Vitest)
cd frontend && npm run e2e                                       # frontend E2E (Playwright; auto-starts backend + preview server)
```

`python -m pytest tests/ -q` alone is NOT the full suite — it misses
`backend/tests/` (FastAPI service/API tests). Always use the no-path form
(`python -m pytest -q`, from repo root) when you mean "run everything."

`.github/workflows/tests.yml` runs the Python suite, the frontend suite, and
`tsc --noEmit` on every PR. It installs `requirements-ci.txt` — the subset of
`requirements.txt` the tests actually need — rather than the full file, because
docling pulls torch and streamlit/chromadb/faiss add gigabytes more, none of
which the suite needs (all are imported lazily behind working fallbacks). **If a
test needs a package not in `requirements-ci.txt`, CI fails with
`ModuleNotFoundError` at collection — add it there.** ESLint runs advisory-only
(`continue-on-error`) while two pre-existing `react-hooks/set-state-in-effect`
errors remain in `ReviewerPanel.tsx` and `SettingsContext.tsx`; fix those and
the flag can be dropped to make lint a real gate. Playwright E2E is not in CI.

`rich` may not be installed in some sandboxes even though it's in `requirements.txt`. If
so, `py_compile` is enough for a syntax check; to exercise `main.py`'s argparse logic,
stub `sys.modules["rich"]` (and submodules) with `MagicMock()` before importing `main`.

## Architecture

### Entry points and dispatch

- `main.py` → `--systematic-review` / `--notebook` (+ one-shot `--notebook-*` flags) /
  `--ask` (Mode 3) drive the core logic. SR adds `--sr-quality` to print the
  risk-of-bias / GRADE / contradiction results.
- `app.py` → `ui/landing.py` → `projects/{mode1_systematic_review,mode2_notebook,mode3_research_assistant}.py::run(settings)` —
  a Streamlit UI that still works locally (`streamlit run app.py`) but is not part of the
  Docker setup; registered in `projects/__init__.py::PROJECT_REGISTRY`.
  `ui/tabs/notebook.py` is the large tab container for all of Mode 2 (Chat, Sources, Summary,
  FAQ, Literature Review, Mind Map, Knowledge Graph, Citation Timeline, Study Comparison,
  Pipeline, Research Report, Explain). `ui/tabs/research_assistant.py` is the single-screen
  Mode 3 tab.

### React + FastAPI web app

A REST API (`backend/`) and a React + TypeScript SPA (`frontend/`) are the primary
web interface. They expose the same core logic as the CLI — `main.py` and `app.py`
are unmodified. Current coverage: all four modes are fully covered. Mode 1 and
Mode 3 are complete; Mode 2 covers the core notebook workflow (create/rename/delete,
source upload, grounded chat with citations), the 7-agent analysis pipeline, the
standalone advanced tools (cross-document summary, FAQ, literature review, mind map,
audio summary, source comparison, knowledge graph, citation timeline, study
comparison), the Explain tab, and the Research Report workflow
(`backend/app/routers/notebook_report.py` + `notebook_report_service.py`,
`frontend/src/components/notebook/ResearchReportTab.tsx`). Mode 4 is web-only
(no Streamlit/CLI equivalent) — see "Mode 4 — Paper Discovery" below.

- `backend/app/main.py` — FastAPI app factory; CORS via `BEESEARCH_CORS_ORIGINS`
  (defaults to the Vite dev ports); mounts `backend/app/routers/{health,
  research_assistant,systematic_review,notebook}.py`, each a thin layer over a
  `backend/app/services/*_service.py` that calls straight into the existing
  `agents/*` / `projects/*` modules — no duplicated business logic.
  `backend/app/schemas/` holds the Pydantic request/response models.
- `backend/app/jobs.py` — in-memory, thread-based background job runner
  (`Job` dataclass + `run_in_background`) reused by every long-running chat
  endpoint across Modes 1–3; the frontend polls `GET /api/*/jobs/{id}` on a
  700ms interval until `status` is `done` or `error` (`pollUntilTerminal` in
  `frontend/src/api/*.ts`).
- `backend/app/mock_llm.py` / `mock_search.py` — dev/test-only stubs, installed
  at the top of `backend/app/main.py` (not the CLI's `main.py`) when
  `BEESEARCH_MOCK_LLM=1` is set, **before** any `agents.*` import (`ChatOllama`
  is bound into other modules' namespaces at import time, so patching later
  wouldn't take effect). Used by the Playwright E2E `webServer` config
  (`frontend/playwright.config.ts`) so tests need neither a reachable Ollama
  server nor network access.
- `backend/tests/` — FastAPI backend test suite (service-layer + API-layer),
  separate from the root `tests/` directory — see Tests above for the command
  that runs both together.
- `frontend/src/api/client.ts` — thin `fetch` wrapper (`apiFetch`/`apiFetchBlob`/
  `apiFetchText`); base URL defaults to `""` (relative), so the Vite dev/preview
  proxy (`frontend/vite.config.ts`) routes `/api/*` to the backend with zero
  frontend env configuration.
- `frontend/src/pages/{LandingPage,SystematicReviewPage,NotebookPage,AskPage,PaperDiscoveryPage}.tsx`
  — one page per mode (`AskPage` = Mode 3, `PaperDiscoveryPage` = Mode 4);
  `App.tsx` does `?mode=mode1|mode2|mode3|mode4` query-param routing, no router dependency.
- **Docker**: the root `Dockerfile` is multi-stage -- a `node:20-alpine` stage runs
  `npm run build` for `frontend/`, then `COPY --from=` copies the built static assets
  into the final `python:3.11-slim` stage at `frontend/dist`. `backend/app/main.py`
  mounts that directory with `StaticFiles(html=True)` at `/` (after all API routers,
  so it only catches unmatched paths), so the FastAPI process serves the React SPA
  directly — no nginx, no separate frontend container. The Dockerfile CMD runs
  `uvicorn backend.app.main:app` directly; `docker compose up --build` is the single
  command (Linux/Windows). Apple Silicon uses `docker-compose.mac.yml` (native Ollama);
  GPU uses `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up`.
  The CLI runs ad hoc via `docker compose exec web python main.py ...`.
  The standalone `frontend/Dockerfile` + `frontend/nginx.conf` (multi-stage `npm run
  build` → nginx, reverse-proxying `/api/*`) still work standalone (`docker build -t
  beesearch-frontend ./frontend`) but are no longer referenced by the default Compose
  files.

### Mode 4 — Paper Discovery

Web-only (no Streamlit/CLI); does not call Ollama. All data comes from the
[Semantic Scholar Academic Graph API](https://api.semanticscholar.org/) — no API key
required on the free tier (rate-limited; set `SEMANTIC_SCHOLAR_API_KEY` in `.env` for
higher limits).

**`paper_graph/` package** (pure backend logic, no FastAPI coupling):
- `s2_client.py` — `SemanticScholarClient` with tenacity exponential backoff (429/5xx),
  `lru_cache` per-process on `get_paper`/`get_references`/`get_citations`, a
  `_meta_cache` dict for batch-populated entries, and `PaperGraphDataSource` Protocol
  as a swap point for OpenAlex. `get_client()` returns a module-level singleton.
- `similarity.py` — pure, zero-I/O functions: `bibliographic_coupling()` (reference-set
  intersection), `co_citation()` (shared-citers count from a pre-built citing index),
  `combined_score()` (min-max normalised, configurable bc/cc weights), `rank_candidates()`.
- `graph_builder.py` — `GraphData` / `GraphEdge` / `PaperNode` dataclasses and two
  assembly helpers: `build_similarity_graph()` and `build_discovery_graph()`.
- `collection_store.py` — `CollectionStore` (thread-safe `threading.Lock`, in-memory,
  lost on restart — same trade-off as `jobs.py`). `get_store()` returns a module-level
  singleton.

**Backend layer** (`backend/app/`):
- `schemas/paper_graph.py` — Pydantic request/response models for all endpoints.
- `services/paper_graph_service.py` — `_resolve_paper_id()` (40-hex S2 ID or title
  search fallback), `run_similarity_graph()` (5-stage pipeline: resolve → fetch refs →
  score → batch-fetch metadata → build graph), `create_collection()`,
  `expand_collection()` (handles all 4 relationship types).
- `routers/paper_graph.py` — prefix `/api/paper-graph`; POST triggers return 202 +
  `job_id`, GET polls follow the standard job pattern from `backend/app/jobs.py`.

**Frontend** (`frontend/src/`):
- `api/paperGraphTypes.ts` / `paperGraph.ts` — TypeScript types and API helpers;
  `pollSimilarityGraphJob` / `pollExpandJob` reuse `pollUntilTerminal`.
- `components/paper-graph/ForceGraph.tsx` — `react-force-graph-2d` canvas wrapper;
  `yearToColor()` year→colour gradient, `nodeRadius()` log-scaled by citation count,
  `paintNode` draws labels for selected/large nodes, directional arrows on
  reference/citation edges.
- `components/paper-graph/PaperDetailPanel.tsx` — node inspector; "Set as new origin"
  button (Feature 1) or relationship selector + "Expand" button (Feature 2).
- `components/paper-graph/SimilarityGraphPanel.tsx` — Feature 1 UI: paper ID/title
  input, top-N and BC/CC weight sliders, job polling, `ResizeObserver`-based width.
- `components/paper-graph/DiscoveryNetworkPanel.tsx` — Feature 2 UI: seed-paper input
  with chip list, collection creation, incremental expand with job polling.
- `pages/PaperDiscoveryPage.tsx` — tab switcher; wired into `App.tsx` as `mode4`.

### Internal "Mode N" numbering vs. user-facing modes

Docstrings and comments use internal mode numbers that don't match the README's
"Mode 1 / Mode 2":

- **Mode 7** = user-facing Mode 1 (Systematic Literature Review) — `agents/systematic_review_*.py`
- **Mode 8** = user-facing Mode 2 (Research Notebook) — `agents/notebook_*.py`
- **Mode 5** = old "Research Partner" (storytelling) — `agents/story_*.py`, now surfaced
  as Mode 2's **Explain** tab
- `agents/graph.py` + `agents/state.py` = a separate "Research Report" workflow, also a
  tab inside `ui/tabs/notebook.py`. Both Explain and Research Report degrade gracefully
  (warn + hide the tab) if their modules are missing.

### Per-pipeline file pattern

Every pipeline (SR, Notebook Q&A, Notebook 7-agent pipeline, Explain/story, Research
Report) follows the same layout under `agents/`:

- `*_state.py` — `TypedDict` + `create_*_state(...)` factory that sets all defaults
  (including `temperature_level` for Notebook-related states)
- `*_nodes.py` — node functions (or inlined in `*_graph.py` for smaller pipelines like
  Research Report); each module has a private `_llm` / `_make_llm(...)` ChatOllama factory
- `*_graph.py` — `build_*_graph()` and a `run_*_turn()`/`run_*()` entry point that
  assembles and invokes the LangGraph `StateGraph`
- `*_memory.py` (where persistence applies) — SQLite read/write helpers

When adding a feature to one pipeline, the analogous files in another pipeline are the
best template.

### LLM response tuning (temperature levels)

`tools/temperature_levels.py::apply_temperature_level(base_temperature, level)` is the
single source of truth for the user-tunable "Response Tuning" feature (Precise /
Focused / Balanced / Creative). It's called from the `_llm`/`_make_llm` factories in
`agents/notebook_nodes.py`, `agents/story_nodes.py`, `agents/notebook_advanced.py`, and
`agents/notebook_pipeline_nodes.py`. `level` flows from `state["temperature_level"]` /
`settings["temperature_level"]`, set via the sidebar "Response Tuning" control or
`/temperature <level>` in the CLI. Calls with `base_temperature <= 0.0` (grading /
faithfulness checks) are always forced to `0.0` regardless of level — this is
deliberate, not a bug.

### Anti-AI writing style enforcement

`tools/writing_style.py` exports three constants injected into prose-generating LLM
prompts across all agents:

- `ANTI_AI_TELL_INSTRUCTION` — strict variant (Chat, Summaries, Literature Review, SR
  synthesis, study guide, Research Report). Bans generic AI vocabulary, formulaic openers,
  and lazy paragraph starters.
- `ANTI_AI_TELL_NARRATIVE_INSTRUCTION` — softer variant for audio/podcast content that
  allows natural spoken transitions (First, Then, Finally) while still banning the
  forbidden vocabulary.
- `ANTI_AI_TELL_REVIEWER_INSTRUCTION` — strictest variant for the Reviewer tool
  (`generate_paper_review` / `reviewer_chat` in `agents/notebook_advanced.py`). Drawn
  from the Wikipedia *Signs of AI Writing* guidelines. Extends the vocabulary ban
  (adds: underscore, crucial, enhance, landscape, realm, interplay, garnered, bolstered,
  impactful, innovative, key-as-adjective) and additionally bans AI structural habits:
  compliment sandwich, hourglass structure, "not X but Y" manufactured contrast,
  "faces challenges / despite these challenges" formula, rule-of-three padding, and
  uniform paragraph length. Requires specific evidence citation (equation numbers,
  section headings, quoted passages) — never abstract references to "the methodology".

**Reviewer pipeline** — `generate_paper_review` runs in three steps so that both the
uploaded paper and external literature ground the critique: (1) extract 3 search queries
from the document (topic, methodology, claimed novelty); (2) search arXiv + Semantic
Scholar, assign `[E1]`–`[E9]` reference numbers; (3) generate the full review with both
the paper text and external abstracts in context — the LLM is instructed to cite `[En]`
inline wherever external evidence supports or contradicts a critique point (missing
baselines, novelty overlaps, mathematical errors, unjustified assumptions, incorrect
claims). `_build_external_ref_block()` formats the numbered block for the prompt.
`reviewer_chat` receives the same `external_refs` list so the follow-up chat can
reference the same papers by `[En]` label.

All three are injected via direct `from tools.writing_style import ...` imports (not
through `tools/__init__.py`'s lazy `__getattr__`) and are also re-exported in `_EXPORTS`
for any code that imports via the `tools` namespace. Do **not** inject these into
JSON-output prompts (FAQ, mind map, knowledge graph, PICO extraction, RoB, GRADE,
screening) — the style rules corrupt structured output parsing.

### Citation grounding

Notebook Chat (`notebook_nodes.py::_build_context_block`), Literature Review
(`notebook_advanced.py::_build_numbered_excerpts`), and the Explain tab
(`story_nodes.py::build_numbered_doc_context`) all follow the same pattern: number
every individual chunk (not document) with its real page tag, bake the tag into the
context string handed to the LLM, then after generation regex-rebuild the References
list in code from whichever numbers the LLM actually cited — never trust the LLM's
own self-written references. When adding citations to a new pipeline, follow this
pattern rather than letting the LLM free-write its own References section.

### SR reference checking, Smart Citations, and Mode 3

- **SR `quality_assessment_node`** (`agents/systematic_review_nodes.py`, wired between
  `evidence_extraction` and `synthesis` in `systematic_review_graph.py`) runs three formerly
  dead-code modules and writes `rob_table` / `grade_results` / `contradictions` to state:
  `agents/risk_of_bias.py` (RoB 2 / ROBINS-I per paper), `agents/grade_assessment.py` (GRADE
  certainty of the whole body), `agents/contradiction_detector.py` (cross-paper conflicts +
  0–100 consensus). Each is independently try/except-wrapped — any failure degrades to an
  empty result, never blocks the pipeline (same "safe no-op" philosophy as self-reflective
  RAG). `synthesis_node` feeds these plus PICO fields into its narrative prompt. Surfaced in
  the UI's Explore → *Risk & Certainty* tool and on the CLI via `--sr-quality`. Paper caps are
  configurable via state (`max_evidence_papers`/`max_synthesis_papers`/`max_rob_papers`).
- **Smart Citations** (`tools/citation_network.py::classify_citation_stances`) optionally
  labels each citation-network edge Supporting/Contrasting/Mentioning from the two papers'
  abstracts (`_parse_stance` defaults to neutral Mentioning on any parse failure);
  `network_to_pyvis_html` colours edges by stance. `tools/citation_context.py` is best-effort,
  open-access-only citing-sentence extraction (`find_citation_mentions` is the pure, tested
  core; `_fetch_fulltext` is the only networked part). Both fail safe.
- **Mode 3 — AI Research Assistant** (`agents/research_assistant.py`) is stateless like the SR
  pipeline (no `*_memory.py`, no graph): `run_research_assistant()` does search → number
  sources into one `[n]` namespace → ground LLM answer → rebuild citations in code from the
  `[n]` actually cited. `build_numbered_sources`/`build_citations`/`_strip_llm_references_section`
  are pure and unit-tested; the search backends and ChatOllama are the only external deps.

### Explain tab: repeated-clarification detection + concept visualization

The Explain pipeline (`agents/story_graph.py`) runs `context_loader → repetition_tracker
→ source_router → storyteller → concept_visualizer → memory_saver → story_eval`.
`repetition_tracker_node` and `concept_visualizer_node` (`agents/story_nodes.py`) detect
when a user re-asks the same question or signals confusion ("I don't understand",
"still lost", …), and respond with both a different explanation style and an
interactive concept map — always on, no UI toggle, matching the tab's existing
automatic online-search behavior.

- **Detection is zero-LLM-call and deterministic**: Jaccard word-overlap similarity
  (stopword-stripped, threshold `0.4`, calibrated against real paraphrase pairs) between
  the current question and recent prior user questions, OR a match against
  `_CONFUSION_PHRASES`. Requires at least one prior assistant turn — a session's first
  message can never be "a repeat." An embeddings-based approach was considered and
  rejected: the added latency/fallback-handling would change the node's character for a
  problem word-overlap already solves well enough.
- **Style rotation reuses existing styles** (`simple`, `analogy`, `walkthrough`,
  `debate`) rather than inventing new categories — `_next_explanation_strategy` honors
  the user's current radio selection unless it matches what was already tried last turn,
  in which case it rotates to the next style in `_STYLE_ROTATION`, wrapping around. If
  the previous turn's `explanation_style` is unknown (sessions saved before this feature
  existed), the node keeps the user's current selection rather than guessing.
- **Concept visualization mirrors `tools/citation_network.py::network_to_pyvis_html`**:
  same Pyvis constructor/styling pattern, simplified to a single hub-and-spoke star graph
  (one central concept + up to 6 related nodes) since no graph algorithms run on it.
  Only triggers on a detected repeat — most turns skip its LLM extraction call entirely.
- **Fails safe like the rest of the codebase**: any failure in extraction, JSON parsing,
  or a missing `pyvis` import is caught and never blocks the primary explanation already
  produced by `storyteller_node` — same philosophy as `self_reflective_rag`'s "any
  grading failure is a safe no-op."
- **`concept_visual_html` is ephemeral**, like the pre-existing `online_results`/
  `source_decision` fields — available in `StoryState` for the current turn's UI render
  only, never persisted to `StorytellerMemory` (avoids SQLite bloat from Pyvis HTML
  blobs). `explanation_style` *is* persisted per assistant turn (`StorytellerMemory.
  add_turn(..., explanation_style=...)`) so future turns know what was already tried.

### Hybrid RAG + Self-Reflective RAG

- `tools/hybrid_store.py::HybridStore` — dense FAISS (`IndexFlatIP`, in-memory,
  per-session, no training) + sparse BM25 (`rank-bm25`) + ChromaDB (persistent
  embedding cache with MD5-based invalidation), fused via Reciprocal Rank Fusion (k=60).
  Falls back to BM25-only if the embedding model isn't pulled.
- `agents/self_reflective_rag.py` — post-retrieval LLM grading (`grade_chunks` for
  Notebook, `grade_papers` for SR), always `temperature=0.0`. Notebook retrieval gets up
  to 2 cycles with query rewrite if fewer than 3 chunks pass grading. Any grading
  failure is a safe no-op (all items kept).

### Config and lazy imports

- `config/settings.py::get_settings()` — `lru_cache`'d Pydantic `BaseSettings`
  singleton reading `.env`. New settings need a `Field(default, alias="ENV_VAR_NAME")`.
- `tools/__init__.py` — `__getattr__`-based lazy re-exports (`_EXPORTS` dict); importing
  `tools` does not pull in `faiss`/`chromadb`/`langchain_ollama` until a specific name is
  accessed. Add new public tool functions to `_EXPORTS` rather than importing the
  submodule eagerly.

### Memory

Research Notebook sessions persist in `outputs/memory/sessions.db` (SQLite, WAL mode):
`notebooks` table (metadata + conversation + `concepts_covered`) and `notebook_chunks`
(chunk text, loaded separately so listing notebooks stays cheap). Embeddings are cached
in ChromaDB under `outputs/chroma_db/`. The SR pipeline (Mode 1) is stateless — no DB
writes; outputs go to `outputs/`.
