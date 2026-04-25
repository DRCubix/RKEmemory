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

### 7. **Production-Tested on Real Infrastructure**
RKE runs on bare-metal Ubuntu with:
- 82 wiki pages, 541 vector chunks, 59 graph nodes
- Qdrant (v1.17+), FalkorDB, BGE-M3 (1024-dim embeddings)
- Sub-second search across the entire knowledge base
- Docker Compose or bare-metal deployment

### 8. **Open-Source Stack, Zero Vendor Lock-In**
Every component is open-source and self-hostable:
- **Qdrant** — Vector database (Apache 2.0)
- **FalkorDB** — Knowledge graph (Server Side Public License)
- **BGE-M3** — Embedding model (MIT)
- **RKE itself** — MIT License

## Quick Start

### Option 1: Docker (Recommended)

```bash
git clone https://github.com/DrCubix/RKEmemory.git
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
git clone https://github.com/DrCubix/RKEmemory.git
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
│   ├── config.py                # Configuration (env + YAML)
│   ├── vector_store.py          # Qdrant + BGE-M3 embeddings
│   ├── graph_store.py           # FalkorDB knowledge graph
│   ├── agent_integration.py     # Agent context injection
│   ├── rlm/
│   │   ├── router.py            # RLM Router (peek/grep/partition)
│   │   └── environment.py       # REPL environment for agents
│   ├── wiki/
│   │   ├── manager.py           # Wiki CRUD + ingest + query + lint
│   │   └── knowledge_base.py    # LlamaIndex RAG integration
│   └── ingestion/
│       ├── knowledge.py         # Parallel domain ingestion
│       ├── drive.py             # Google Drive ingestion
│       └── git_repos.py         # Git repo scanning
├── config/
│   ├── rke.yaml.example         # Main config template
│   └── sources.yaml.example     # Knowledge sources template
├── scripts/
│   ├── setup.sh                 # One-line setup (venv + dirs + config)
│   └── start-services.sh        # Start Qdrant + FalkorDB
├── docker-compose.yml           # Docker services
├── pyproject.toml               # Python package definition
└── README.md                    # This file
```

## Requirements

- **Python 3.10+**
- **Docker & Docker Compose** (for services) — or install Qdrant/FalkorDB manually
- **NVIDIA GPU** (optional, for GPU-accelerated embeddings)
- **50GB+ disk space** (for BGE-M3 model + vector data)

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
