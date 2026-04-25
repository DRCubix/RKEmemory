# Changelog

All notable changes will be documented here. This project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and uses
[SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-04-25

Five new memory-system primitives, built by a parallel sub-agent fleet
and hardened through 11 rounds of automated Codex audit (30+ blockers
caught and fixed).

### Added — five v0.2 features

- **`rke.wiki.search_index`** (optional `[search]` extra) — Whoosh-backed
  inverted index that accelerates `WikiManager.query_wiki()` from a
  ~200ms p95 linear scan to BM25F lookup. Composite `(category, slug)`
  uniqueness key keeps same-slug-different-category pages distinct.
  Tenant-aware via `WhooshIndex.attach(wm, ...)`.

- **`rke.knowledge.extractor`** — pluggable entity / relation extractor
  with three backends: `regex` (offline default, zero deps), `anthropic`
  (Claude), `openai`. `attach_to_wiki(graph, wiki=wm)` registers a
  tenant-scoped post-create hook that auto-feeds the FalkorDB graph;
  failures never propagate out of page creation. Optional deps via the
  new `[llm]` extra.

- **`rke.wiki.lifecycle`** — Letta-style memory pressure / TTL layer:
  `set_expiry`, `touch`, `is_expired`, `expired_pages`, `evict_expired`,
  `lru_candidates`, `evict_lru`, `AccessTracker` (no-recursion installer).
  All API points accept a `category=` argument to disambiguate same-slug
  pages.

- **`rke.graph_temporal`** — Zep-style bi-temporal facade over
  `GraphStore`. `TemporalRelation` carries `valid_from / valid_to /
  recorded_at`; `query_at(t)` injects the temporal predicate;
  `invalidate(at)` is idempotent; `history()` and
  `fact_changes_between()` ship.

- **`rke.adapters.chat_memory`** — LangChain-`ConversationBufferMemory`-
  compatible adapter persisting each thread as a wiki page, with
  metadata-preserving line format, summarize/archive rollover, and
  KB-routed writes so `search_long_term()` stays in sync with the
  vector index.

### Added — supporting infrastructure

- Extension hooks in `WikiManager`: `register_post_create`,
  `register_post_delete`, `set_query_backend`, `clear_hooks`.
  `WikiPage` gained optional `expires_at` and `last_accessed_at`
  frontmatter fields.
- `KnowledgeBase` now registers a tenant-scoped `post_delete` hook that
  drops the page's vector chunks from Qdrant on delete or eviction.
- `VectorStore.delete_by_filter(match, min_chunk_index=...)` evicts
  stale chunks left over from longer past revisions.
- `pyproject.toml` extras: `[search]` (whoosh), `[llm]` (anthropic +
  openai), `[dev]` updated to include whoosh + psutil.
- New scripts: `scripts/integration_v02.py` (18-stage live integration
  smoke against Qdrant + FalkorDB), `scripts/feature_test.py`
  (52-stage v0.1 regression suite), `scripts/benchmark.py` (latency,
  throughput, recall@K). All three auto-isolate to per-run /tmp scratch
  namespaces respecting both `RKE_*` and `RKE_*__*` env-var aliases —
  safe to run on a developer machine without clobbering real data.

### Fixed (caught by Codex audit)

This release was hardened through 11 rounds of automated review. Notable
fixes the unit tests had not surfaced:
- AccessTracker recursion that triggered ~1000 file rewrites per single
  read (Python's recursion-limit error was being swallowed).
- `created_at` timestamp corruption when a page was edited (lifecycle
  rewrites went through `create_page(overwrite=True)`).
- `qdrant-client` 1.10+ API (`query_points` replaces deprecated
  `search`, `CollectionInfo.vectors_count` removed).
- `sentence-transformers` ≥3.0 method rename
  (`get_sentence_embedding_dimension` → `get_embedding_dimension`).
- Whoosh slug collision across categories (composite key fix).
- Bound-method identity vs equality (`is` vs `==`) in detach paths.
- ChatMemory metadata persistence round-trip.
- Hook accumulation on repeated attach (now idempotent across all
  feature modules).
- KB cross-tenant vector deletion (now scoped to wiki root).
- Validation scripts wiping user data (now auto-isolate to /tmp).

### Known limitations (v0.3 roadmap)

- **Concurrent multi-tenant in a single Python process** — extractor
  and search_index installations are scoped to one (graph, wm) pair at
  a time via module-global state. Side-by-side tenants in the SAME
  process work for KnowledgeBase but not for extractor or
  search_index. For most deployments (one process per tenant) this is
  irrelevant. A registry-per-tenant redesign is on the v0.3 list.

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
