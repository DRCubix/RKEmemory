# Changelog

All notable changes will be documented here. This project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and uses
[SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-04-24

End-to-end validation against live Qdrant + FalkorDB completed. Two real
compatibility bugs surfaced and were fixed.

### Fixed
- `rke.vector_store`: switched to `client.query_points()` because
  `qdrant-client` 1.10+ removed `client.search()`. Falls back to legacy
  `.search()` when `query_points` is absent.
- `rke.vector_store`: tolerate `CollectionInfo` without `vectors_count`
  (the field was removed in newer qdrant-client versions).
- `rke.vector_store`: future-proof embedding-dim getter against the
  `sentence-transformers` ≥3.0 rename
  (`get_sentence_embedding_dimension` → `get_embedding_dimension`).
- `docker-compose.yml`: pinned Qdrant to `v1.17.1` (the previously-listed
  `v1.17.4` tag does not exist on Docker Hub).

### Added
- `scripts/e2e_smoke.py`: 7-stage end-to-end test exercising wiki create+
  index, vector search, hybrid search, FalkorDB entity/relation/cypher,
  combined KB query, agent context formatting, deterministic RLM router,
  and lint. Verified passing against live Qdrant 1.17.1 + FalkorDB latest.

### Security / Repo posture
- Branch protection enabled on `master` (requires `pytest (py3.10)`,
  `pytest (py3.11)`, `pytest (py3.12)`, and `ruff` to pass).
- Dependabot security updates enabled (auto-PRs on CVEs).
- Commit history rewritten so all author/committer emails use the GitHub
  noreply form (`<id>+DRCubix@users.noreply.github.com`); the prior
  non-noreply author address no longer appears in any reachable commit.

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
