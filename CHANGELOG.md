# Changelog

All notable changes will be documented here. This project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and uses
[SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-04-24

First real, runnable release. The 1.0.0 tag in the initial commit was a
placeholder; all source files were empty stubs. That has been replaced with
a working scaffold and the version reset to 0.1.0 (alpha) until end-to-end
validation against Qdrant + FalkorDB + BGE-M3 is complete.

### Added
- `rke.config` — layered loader (defaults < `config/rke.yaml` < `RKE_*` env vars).
- `rke.vector_store` — Qdrant client wrapper with sentence-transformers
  embeddings, `search()`, and `hybrid_search()` (semantic + keyword overlap).
- `rke.graph_store` — FalkorDB wrapper with `add_entity`, `add_relation`,
  raw Cypher `query()`, and a naive `graphrag_query()`.
- `rke.wiki.manager` — markdown wiki with frontmatter, CRUD, keyword query,
  and `lint()` (orphans, broken links, duplicates, empty pages).
- `rke.wiki.knowledge_base` — `KnowledgeBase` ties wiki and vector store
  together, with character-level chunking and `reindex_all()`.
- `rke.rlm.environment` — REPL with `peek`, `grep`, `partition`, `sub_rlm`.
- `rke.rlm.router` — RLM loop. Deterministic fallback when no LLM provider
  is configured; pluggable Anthropic / OpenAI clients otherwise.
- `rke.ingestion.knowledge` — parallel local file ingestion driven by
  `config/sources.yaml`.
- `rke.ingestion.git_repos` — scan or clone git repos and ingest.
- `rke.ingestion.drive` — Google Drive ingestion (optional dep
  `pip install -e ".[drive]"`).
- `rke.agent_integration` — `gather_context()` + `format_context_for_agent()`
  with role framing for Claude / Codex / Gemini / generic.
- `rke.cli` — Typer CLI: `status`, `search`, `query`, `graph`, `graph-seed`,
  `ingest`, `wiki-page`, `wiki-list`, `lint`, `rlm`, `reindex`, `version`.
- Tests for config, wiki, and chunking (24 deterministic tests, all passing).

### Fixed
- `LICENSE`: replaced placeholder text with full MIT license body.
- `pyproject.toml`: real package metadata, dependencies, and `[project.scripts]`
  entry-point so `rke` is on PATH after `pip install -e .`.
- `docker-compose.yml`: working Qdrant + FalkorDB services with persistent
  bind-mounts under `./storage/` (gitignored).
- `.env.example`, `config/rke.yaml.example`, `config/sources.yaml.example`,
  `scripts/setup.sh`, `scripts/start-services.sh`: real, documented templates.
- README clone URLs (`DrCubix` → `DRCubix`).

### Security
- Audit confirmed no secrets, OAuth tokens, personal paths, internal IPs, or
  session memories in the repo or git history. `.gitignore` covers `.env`,
  `data/`, `models/`, `logs/`, `storage/`, `snapshots/`, OAuth tokens, and
  `credentials.json`. No GitHub Actions secrets, webhooks, or deploy keys
  configured at the repo level.

## [1.0.0] - 2026-04-24 (yanked — placeholder release)

Original commit. All Python source, configs, and scripts were 156-byte
placeholder stubs. Superseded by 0.1.0.
