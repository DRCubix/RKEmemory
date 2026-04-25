<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License">
  <img src="https://img.shields.io/badge/Qdrant-1.17%2B-orange.svg" alt="Qdrant">
  <img src="https://img.shields.io/badge/FalkorDB-compatible-purple.svg" alt="FalkorDB">
</p>

<h1 align="center">🧠 RKEmemory — Recursive Knowledge Engine</h1>

<p align="center">
  <strong>Unified multi-agent memory system with RLM routing, LLM-Wiki compilation,<br>
  Qdrant vector search, and FalkorDB knowledge graph.</strong>
</p>

<p align="center">
  <em>Agents never see full context. They explore knowledge programmatically.</em>
</p>

> ⚠️ **Status: alpha, v0.2.0** ([release notes](https://github.com/DRCubix/RKEmemory/releases/tag/v0.2.0)).
> The core (CLI, config, wiki manager, vector store, graph store, RLM
> router, ingestion pipelines, agent integration) plus five v0.2 memory
> primitives — Whoosh inverted index, LLM entity extraction, lifecycle/TTL,
> bi-temporal graph, and a LangChain-compatible chat memory adapter — are
> implemented, unit-tested, and validated end-to-end against live Qdrant
> + FalkorDB. **Expect breaking changes** until v1.0. Feedback and PRs
> welcome.

---

## What Is RKE?

RKE (Recursive Knowledge Engine) is an **open-source memory architecture for AI agents** that solves the fundamental problem of context management in multi-agent systems. Instead of stuffing growing amounts of context into every agent prompt — causing context rot, token bloat, and degraded reasoning — RKE gives agents a **programmatic knowledge interface** they can explore, query, and refine on demand.

It combines four subsystems into a single, cohesive memory layer:

| Layer | Technology | Purpose |
|---|---|---|
| **RLM Router** | Recursive Language Models (Zhang & Khattab, 2025) | Context-as-code REPL — agents `peek()`, `grep()`, `partition()` knowledge without seeing it all |
| **LLM-Wiki** | Git-tracked markdown + Karpathy pattern | Compounded knowledge base that grows richer with every ingestion and query |
| **Vector Store** | Qdrant + BGE-M3 (local, 1024-dim) | Semantic search across wiki, code, conversations, and project embeddings |
| **Knowledge Graph** | FalkorDB (Cypher queries) | Entity-relationship graph connecting APIs, agents, decisions, patterns, and projects |

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Agent Layer                              │
│     Claude Code  │  Codex  │  Gemini CLI  │  Hermes  │  Custom  │
└────────────────────────┬────────────────────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │  Agent Integration  │
              │   Context Builder   │
              │                     │
              │  • gather_context() │
              │  • format_for_agent │
              │  • role formatting  │
              └──────────┬──────────┘
                         │
     ┌───────────────────┼───────────────────┐
     │                   │                   │
┌────▼────┐      ┌───────▼───────┐    ┌──────▼──────┐
│ Honcho  │      │  RLM Router   │    │ LLM-Wiki    │
│ (Who/   │      │  & Environment│    │ (KB)        │
│  Why)   │      │               │    │             │
│         │      │ peek()        │    │ create_page │
│ profiles│      │ grep()        │    │ query_wiki  │
│ patterns│      │ partition()   │    │ lint()      │
│ memory  │      │ sub_rlm()     │    │ ingest()    │
└────┬────┘      └───────┬───────┘    └──────┬──────┘
     │                   │                   │
     └───────────────────┼───────────────────┘
                         │
            ┌────────────▼────────────┐
            │   Knowledge Queries     │
            │                         │
            │  Parallel search across │
            │  wiki + vectors + graph │
            └────────────┬────────────┘
                         │
          ┌──────────────┼──────────────┐
          │                             │
   ┌──────▼──────┐              ┌───────▼───────┐
   │  Qdrant     │              │  FalkorDB     │
   │  Vectors    │              │  Knowledge    │
   │             │              │  Graph        │
   │ • semantic  │              │               │
   │ • hybrid    │              │ • entities    │
   │ • keyword   │              │ • relations   │
   │ • BM25      │              │ • GraphRAG    │
   │             │              │ • Cypher      │
   └─────────────┘              └───────────────┘
```

### Data Flow: Agent Query

```
Agent asks: "How do we handle OAuth token switching?"
                    │
    ┌───────────────┼────────────────┐
    │               │                │
    ▼               ▼                ▼
  Wiki            Vectors           Graph
 (keyword)      (semantic)       (entities)
    │               │                │
    ▼               ▼                ▼
┌────────────────────────────────────────┐
│         Context Synthesis              │
│                                        │
│  Wiki: "multi-account-oauth" page      │
│  Vectors: 5 matching chunks (0.72 avg) │
│  Graph: OAuth API → Token → Account    │
│                                        │
│  → Formatted context block             │
│  → Agent-specific instructions         │
│  → Injected into agent prompt          │
└────────────────────────────────────────┘
                    │
                    ▼
           Agent responds with
          full knowledge access,
         zero context overflow
```

## Key Advantages

### 1. **No Context Rot — Ever**
Traditional RAG systems dump retrieved context into agent prompts. As knowledge grows, prompts balloon, reasoning degrades, and agents lose focus. RKE uses the **Recursive Language Model pattern** — agents interact with knowledge through a REPL environment with `peek()`, `grep()`, `partition()`, and `sub_rlm()` operations. They never see the full context; they explore it like a filesystem.

### 2. **Knowledge Compounds Over Time**
The LLM-Wiki isn't a static knowledge base — it's a **living, self-improving system**. Every ingestion adds structured pages. Every query enriches the index. A built-in linter finds contradictions, detects orphaned pages, and suggests cross-links. The wiki is git-tracked, so every change has full version history.

### 3. **Cross-Agent Shared Memory**
Claude Code, Codex, Gemini, Hermes — all agents read from and write to the **same knowledge base**. When one agent learns something, every other agent benefits. No more re-teaching each agent your project's architecture, decisions, or patterns.

### 4. **Local-First, Privacy-Respecting**
Embeddings use **BGE-M3 running locally on your GPU** (or CPU). No API calls, no data leaving your machine, no per-token costs. The vector store and graph database run locally too. Your knowledge stays yours.

### 5. **Multi-Modal Knowledge Ingestion**
RKE ingests from diverse sources:
- **Files & directories** — Markdown, code, docs
- **Google Drive** — Docs, Sheets, Slides (with OAuth)
- **Git repositories** — Local and GitHub repos
- **Direct wiki pages** — Create knowledge manually
- **Agent conversations** — Capture decisions and patterns

### 6. **Three-Layer Search Strategy**
Every query runs in parallel across three layers:
- **Keyword search** — Fast, exact matching on the wiki index
- **Semantic search** — Vector similarity with hybrid keyword boosting
- **Graph search** — Entity relationships extracted via Cypher
Results are synthesized into a single context block for the agent.

### 7. **Measured, Reproducible Performance**
On a 200-doc synthetic corpus (CPU only, MiniLM-L6-v2 384d) the
shipped `scripts/benchmark.py` reports:

| Metric | Value |
|---|---|
| Embedding throughput | **104 docs/sec** |
| End-to-end ingestion | **25 pages/sec** |
| Vector search p50 / p95 | **9.9 / 12.8 ms** |
| Cypher 1-hop p50 (FalkorDB) | **0.2 ms** |
| Recall@1 / Recall@5 | **100% / 100%** |

With the v0.2 Whoosh inverted index, combined `wiki + vector + graph`
query p95 drops from ~200ms (v0.1 linear scan) to ~10–20ms.

### 8. **Open-Source Stack, Zero Vendor Lock-In**
Every component is open-source and self-hostable:
- **Qdrant** — Vector database (Apache 2.0)
- **FalkorDB** — Knowledge graph (Server Side Public License)
- **BGE-M3** — Embedding model (MIT)
- **RKE itself** — MIT License

## What's New in v0.2

Five additional memory primitives, each opt-in and tenant-scoped:

| Module | What it does |
|---|---|
| `rke.wiki.search_index` | **Whoosh BM25F inverted index** for `query_wiki()`. Composite `(category, slug)` keys keep same-slug pages from different categories distinct. Optional `[search]` extra. |
| `rke.wiki.tantivy_index` | **Tantivy (Rust) BM25F backend** — drop-in alternative with the same API. Measured **17–21× faster** than Whoosh on query latency (0.05 ms p50 vs 0.88 ms on a 200-doc corpus). Install via `[search-tantivy]`. |
| `rke.knowledge.extractor` | **Auto-extracts entities and relations** from new wiki pages and writes them to FalkorDB. Three backends: `regex` (offline default), `anthropic`, `openai` (`[llm]` extra). |
| `rke.wiki.lifecycle` | **TTL / LRU eviction** — `set_expiry`, `touch`, `expired_pages`, `evict_expired`, `lru_candidates`, `evict_lru`, `AccessTracker`. Pages without lifecycle metadata stay immortal. |
| `rke.graph_temporal` | **Bi-temporal graph layer** — `TemporalRelation` carries `valid_from / valid_to / recorded_at`; `query_at(t)` injects the temporal predicate; `history()`, `invalidate(at)`, `fact_changes_between()` ship. Composes over `GraphStore` (no subclassing). |
| `rke.adapters.chat_memory` | **LangChain-compatible chat history**, persisted as wiki threads with `summarize_and_archive()` rollover. Routes through `KnowledgeBase` so `search_long_term()` finds embedded conversation content. |

Hardened through 11 rounds of automated Codex review (30+ blockers caught and fixed; see [CHANGELOG.md](CHANGELOG.md)).

## Quick Start

### Option 1: Docker (Recommended)

```bash
git clone https://github.com/DRCubix/RKEmemory.git
cd RKEmemory

# One-line setup
bash scripts/setup.sh

# Start Qdrant + FalkorDB
docker compose up -d

# Check status
rke status
```

### Option 2: Bare-Metal

```bash
git clone https://github.com/DRCubix/RKEmemory.git
cd RKEmemory

# Setup
bash scripts/setup.sh

# Install Qdrant
curl -sL https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-gnu.tar.gz | tar -xzf -
./qdrant &

# Install FalkorDB
pip install falkordb-bin
python3 -c "
import pathlib, falkordb_bin, subprocess
pkg = pathlib.Path(falkordb_bin.__file__).parent / 'bin'
subprocess.run([str(pkg/'redis-server'), '--loadmodule', str(pkg/'falkordb.so'), '--port', '6379', '--daemonize', 'yes'])
"

# Check status
rke status
```

## CLI Usage

```bash
# System status
rke status

# Semantic vector search
rke search "oauth token switching patterns"

# Multi-layer knowledge query (wiki + vectors + graph)
rke query "How does our authentication system work?"

# Cypher query on knowledge graph
rke graph "MATCH (a:API)-[:USED_BY]->(p:Project) RETURN a.name, p.name"

# Seed the graph with initial knowledge
rke graph-seed

# Ingest a document
rke ingest document.md

# Create a wiki page
rke wiki-page "My Topic" "# Content goes here"

# Wiki health check (find orphans, contradictions)
rke lint

# Query through RLM router
rke rlm "What patterns do we use for authentication?"
```

## Python API

```python
from rke.wiki.manager import WikiManager
from rke.vector_store import VectorStore
from rke.graph_store import GraphStore
from rke.rlm.router import RLMRouter
from rke.agent_integration import gather_context, format_context_for_agent

# ── Wiki: create, query, lint ───────────────────────────
wiki = WikiManager()
wiki.create_page("OAuth Patterns", "# OAuth\n\nContent here...", "entities")
result = wiki.query_wiki("How does token switching work?")

# ── Vector Store: semantic + hybrid search ─────────────
store = VectorStore()
results = store.hybrid_search("authentication patterns", limit=5)

# ── Knowledge Graph: Cypher + GraphRAG ─────────────────
graph = GraphStore()
graph.connect()
graph.graphrag_query("What APIs does the project use?")

# ── RLM Router: recursive agent completion ─────────────
router = RLMRouter()
result = router.complete(agent="gemini", query="Design the auth architecture")

# ── Agent Integration: inject RKE context ──────────────
context = gather_context("Build the auth module")
prompt = format_context_for_agent(context, "claude")  # or "codex", "gemini"
```

### v0.2 features

```python
# ── Whoosh inverted index ── (`pip install -e ".[search]"`) ───
from rke.wiki.manager import WikiManager
from rke.wiki.search_index import WhooshIndex
wm = WikiManager()
WhooshIndex.attach(wm, index_dir="data/.wiki_index")
wm.query_wiki("oauth refresh tokens")  # now BM25F-ranked

# ── LLM entity extraction ── (`pip install -e ".[llm]"` for LLM backends) ─
from rke.knowledge.extractor import EntityExtractor, attach_to_wiki
from rke.graph_store import GraphStore
graph = GraphStore()
# Tenant-scoped: only this wm's pages feed `graph`. Regex backend is
# offline; provider="anthropic"|"openai" needs the [llm] extra + creds.
attach_to_wiki(graph, wiki=wm, extractor=EntityExtractor(provider="regex"))
wm.create_page("RKE Stack", "RKE uses Qdrant. Qdrant depends on Rocksdb.")
# → graph now has Project(RKE) -[USES]-> API(Qdrant) -[DEPENDS_ON]-> API(Rocksdb)

# ── Lifecycle / TTL / LRU ─────────────────────────────────────
from rke.wiki.lifecycle import set_expiry, evict_expired, AccessTracker
set_expiry(wm, "old-experiment", days=30)            # expire in 30 days
evicted = evict_expired(wm)                          # delete past-due
with AccessTracker(wm):                              # auto-touch on read
    page = wm.get_page("oauth-patterns")             # last_accessed_at updated

# ── Bi-temporal graph ─────────────────────────────────────────
from rke.graph_temporal import TemporalGraphStore, TemporalRelation
from datetime import datetime, timezone
tgs = TemporalGraphStore(graph)
tgs.add_relation(TemporalRelation(
    "Project", "RKE", "USES", "API", "Qdrant",
    valid_from=datetime(2025, 1, 1, tzinfo=timezone.utc),
))
rows = tgs.query_at(
    "MATCH (p:Project)-[r:USES]->(a:API) RETURN a.name",
    t=datetime(2025, 6, 1, tzinfo=timezone.utc),
)

# ── Chat memory adapter (LangChain-style) ─────────────────────
from rke.adapters.chat_memory import ChatMemory
from rke.wiki.knowledge_base import KnowledgeBase
kb = KnowledgeBase()
mem = ChatMemory(thread_id="user-42", kb=kb,
                 buffer_size=20, summarize_threshold=50)
mem.add_user_message("How do refresh tokens work?")
mem.add_assistant_message("They rotate when users switch accounts.")
mem.history()                  # last N messages, chronological
mem.to_prompt_string()         # rendered for LLM injection
mem.search_long_term("refresh tokens")   # KB-routed, vector-backed
```

## Ingestion Pipeline

```bash
# Ingest all markdown files from configured sources
python3 -m rke.ingestion.knowledge all --parallel

# Ingest Google Drive documents
python3 -m rke.ingestion.drive

# Scan and ingest local git repositories
python3 -m rke.ingestion.git_repos /path/to/repos
```

Configure sources in `config/sources.yaml`:

```yaml
sources:
  my-project:
    name: "My Project Docs"
    paths:
      - "/path/to/docs"
      - "/path/to/code"
    category: "projects/my-project"
    tags: ["project", "docs"]

  research:
    name: "Research Papers"
    paths:
      - "/path/to/papers"
    category: "entities"
    tags: ["research"]
```

## Configuration

All configuration lives in `config/rke.yaml`. Copy from `config/rke.yaml.example` and customize:

```yaml
wiki:
  path: data/wiki

qdrant:
  host: localhost
  port: 6333
  grpc_port: 6334

falkordb:
  host: localhost
  port: 6379

embedding:
  model: BAAI/bge-m3
  model_path: models/BAAI/bge-m3
  device: "cuda:0"   # Use "cpu" for CPU-only
  dimensions: 1024

rlm:
  max_depth: 5
  max_iterations: 20
  max_cost_usd: 1.00
  timeout_seconds: 300
```

Environment variables (`RKE_*`) override YAML values. See `.env.example` for all options.

## Project Structure

```
RKEmemory/
├── src/rke/
│   ├── __init__.py              # Package version
│   ├── __main__.py              # Entry point
│   ├── cli.py                   # CLI orchestrator (rke command)
│   ├── config.py                # Configuration (defaults < YAML < env)
│   ├── vector_store.py          # Qdrant + sentence-transformers
│   ├── graph_store.py           # FalkorDB knowledge graph
│   ├── graph_temporal.py        # ★ v0.2 — bi-temporal facade
│   ├── agent_integration.py     # Agent context injection
│   ├── rlm/
│   │   ├── router.py            # RLM Router (peek/grep/partition)
│   │   └── environment.py       # REPL environment for agents
│   ├── wiki/
│   │   ├── manager.py           # Wiki CRUD + extension hooks
│   │   ├── knowledge_base.py    # Wiki + vector chunked indexer
│   │   ├── search_index.py      # ★ v0.2 — Whoosh BM25F accelerator
│   │   ├── tantivy_index.py     # ★ v0.2.1 — Tantivy BM25F (Rust, ~20× faster)
│   │   └── lifecycle.py         # ★ v0.2 — TTL / LRU / AccessTracker
│   ├── knowledge/
│   │   └── extractor.py         # ★ v0.2 — entity / relation extractor
│   ├── adapters/
│   │   └── chat_memory.py       # ★ v0.2 — LangChain-compat chat adapter
│   └── ingestion/
│       ├── knowledge.py         # Parallel local ingestion
│       ├── drive.py             # Google Drive ingestion ([drive])
│       └── git_repos.py         # Git repo scanning
├── config/
│   ├── rke.yaml.example         # Main config template
│   └── sources.yaml.example     # Knowledge sources template
├── scripts/
│   ├── setup.sh                 # One-line setup (venv + dirs + config)
│   ├── start-services.sh        # Start Qdrant + FalkorDB
│   ├── feature_test.py          # ★ 52-stage v0.1 feature smoke
│   ├── integration_v02.py       # ★ 18-stage v0.2 live integration
│   └── benchmark.py             # ★ latency / throughput / recall@K
├── tests/                       # 84 unit tests across 8 files
├── .github/workflows/ci.yml     # pytest 3.10/3.11/3.12 + ruff
├── docker-compose.yml           # Qdrant + FalkorDB
├── pyproject.toml               # extras: [search] [llm] [drive] [llamaindex] [dev]
└── README.md                    # This file
```

## Requirements

- **Python 3.10+**
- **Docker & Docker Compose** (for services) — or install Qdrant/FalkorDB manually
- **NVIDIA GPU** (optional, for GPU-accelerated embeddings)
- **~3 GB disk space** (BGE-M3 default model; ~100 MB if you swap to MiniLM-L6 via `RKE_EMBEDDING_MODEL`)

### Optional dependency extras

```bash
pip install -e ".[search]"          # Whoosh inverted index for query_wiki()
pip install -e ".[search-tantivy]"  # Tantivy BM25F (Rust, ~20× faster)
pip install -e ".[llm]"             # anthropic + openai for LLM extraction / RLM
pip install -e ".[drive]"           # Google Drive ingestion
pip install -e ".[llamaindex]"      # LlamaIndex integration
pip install -e ".[dev]"             # pytest + ruff + mypy + benchmark deps
```

## License

MIT — see [LICENSE](LICENSE) for details.

## Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Credits

- Built on the [Recursive Language Models](https://arxiv.org/abs/2512.24601) paper (Zhang & Khattab, 2025)
- Wiki pattern inspired by [Karpathy's LLM-Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)
- Developed by the [CubixAI](https://github.com/CubixAI) team
