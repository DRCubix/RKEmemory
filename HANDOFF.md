# RKE Handoff — Benchmark Underperformance Investigation

> **Status as of 2026-04-25, master `f06f6c4`+** — RKE v0.3 (post-`30d09dd` ChatMemory fixes + `f06f6c4` eval harness) is **measurably behind** published competitors on agent-memory benchmarks. The next session's job is to **research and propose ranked improvements** — not to ship code, just to figure out what to change.

## Where RKE actually sits today

Real J-scores measured at `n=20` with `qwen3.5-plus` (Alibaba DashScope) as both the answer-composer (`RKE_LLM_PROVIDER=openai`) and the judge:

| Benchmark | RKE v0.3 | Best published | Gap | Hardest competitor on this benchmark |
|---|---|---|---|---|
| **DMR** (MSC-Self-Instruct, 20 of ~500) | **85.0%** | 94.8% (Zep) | **-10pt** | Zep 94.8%, Letta 93.4% |
| **LongMemEval-oracle** (n=20) | **55.0%** | 96.6% (MemPalace, disputed) | **-35 to -41pt** | Honcho 90.4%, MemPalace 96.6% |
| **LOCOMO** (conv-30, n=20) | **30.0%** | 89.9% (Honcho) | **-60pt** | Honcho 89.9%, Zep 75.14%, Mem0 ~66% |

**Caveats already understood:**
- n=20, not n=200–500. Wide CI.
- Single LOCOMO conversation, not the full 10-conv set.
- MiniLM-L6 384d embeddings, not BGE-M3 1024d.
- Judge = qwen3.5-plus, not Claude/GPT-4 (judge variance is real).
- Sample data: `/tmp/lo20j.json`, `/tmp/lme20j.json`, `/tmp/dmr20j.json`.

## What v0.3 already shipped (don't redo)

| Layer | Status | File |
|---|---|---|
| Whoosh BM25F (legacy) | shipped, in `[search]` extra | `src/rke/wiki/search_index.py` |
| Tantivy BM25F (Rust, ~20× faster) | shipped, in `[search-tantivy]` extra | `src/rke/wiki/tantivy_index.py` |
| LLM entity extraction (regex/Anthropic/OpenAI) | shipped, tenant-scoped | `src/rke/knowledge/extractor.py` |
| TTL/LRU lifecycle | shipped, category-aware | `src/rke/wiki/lifecycle.py` |
| Bi-temporal graph (`valid_from`/`valid_to`/`recorded_at`/`query_at`) | shipped, **not yet wired into `mem.answer()`** | `src/rke/graph_temporal.py` |
| ChatMemory adapter | shipped + v0.3 per-turn chunk fix | `src/rke/adapters/chat_memory.py` |
| `ChatMemory.answer(question)` via RLM router | shipped (deterministic + LLM modes) | `src/rke/adapters/chat_memory.py` |
| Eval harness (LOCOMO/LongMemEval/DMR with substring/anthropic/openai judges) | shipped | `scripts/eval_*.py` |
| Audit history | 12 rounds of Codex review across v0.2 + v0.3, 30+ blockers patched | see `CHANGELOG.md` |

## Where each benchmark is leaking score (best guesses, need validation)

### LOCOMO -60pt (worst gap)
LOCOMO is multi-session **temporal** reasoning ("when did Jon lose his job?", "which event was first?"). RKE has bi-temporal graph (`graph_temporal.py`) that could answer these directly via `query_at(t)` — but `ChatMemory.answer()` routes to RLM router which uses semantic vector retrieval, ignoring the graph entirely. Hypothesis: **wiring the graph into the answer pipeline closes most of the LOCOMO gap.**

### LongMemEval -35pt
LongMemEval's "oracle" subset gives only the relevant sessions; the gap is purely retrieval/composition quality. Hypothesis: **larger embedding model (BGE-M3 1024d) + a re-ranking step closes most of this gap.** Honcho's 90% is reachable; MemPalace's 96.6% is not (independent reviewers note it's basically a ChromaDB score, methodology questionable).

### DMR -10pt (best gap)
Closest to parity. DMR is multi-session conversational recall. The 85% likely caps out around 90% with the same LLM judge but a better composer model (qwen3.5-plus may underperform Claude/GPT-4 on free-form composition). **Less leverage here — focus on LOCOMO + LongMemEval first.**

## Research questions the next session should answer

For each leading competitor, gather **specific, citation-backed answers** to these 5 questions. Then synthesize into a ranked change list for RKE.

1. **What retrieval architecture does it use?** Vector-only? Vector+graph? Vector+keyword fusion (BM25+dense)? Re-ranking (Cohere/cross-encoder)?
2. **How does it compose answers from retrieved memory?** Single LLM call with stuffed context? Iterative agent loop? Specific prompt template?
3. **How does it handle temporal reasoning?** Bi-temporal facts? Relative-time NL parsing? Session-ordered ranking?
4. **What benchmarks did it tune for?** What hyperparameters/embedding model?
5. **What is the ONE technique it claims is its differentiator?** (e.g. Zep = Graphiti, Letta = paged context, Mem0 = LLM extraction at write-time)

Specifically read:
- **Zep paper** ([arXiv 2501.13956](https://arxiv.org/abs/2501.13956)) — Graphiti temporal KG construction. Likely most relevant for closing the LOCOMO gap.
- **Mem0 paper** ([arXiv 2504.19413](https://arxiv.org/abs/2504.19413)) — write-time LLM fact extraction; latency tradeoff.
- **MemGPT/Letta paper** ([arXiv 2310.08560](https://arxiv.org/abs/2310.08560)) — paged context as OS-style memory hierarchy.
- **Honcho's eval methodology** ([evals.honcho.dev](https://evals.honcho.dev/)) — what about their identity/ToM model maps to LongMemEval-S 90.4%?
- **LongMemEval paper** ([arXiv 2410.10813](https://arxiv.org/abs/2410.10813)) — to understand what categories of question are being lost.
- **LOCOMO paper** ([Snap Research](https://snap-research.github.io/locomo/)) — same.

## Concrete next-session deliverable

Produce `RESEARCH.md` (don't write code, just analysis) containing:

1. **Per-system breakdown** — for each of {Zep, Mem0, Letta, Honcho}, a half-page summary answering the 5 questions above with citations.
2. **Gap analysis table** — for each of LOCOMO/LongMemEval/DMR, list the techniques used by the leader on that benchmark that RKE does NOT currently implement.
3. **Ranked recommendation list** — 5–8 concrete code changes RKE should make, in order of expected J-score lift, with: (a) which benchmark each helps, (b) effort estimate (hours/days), (c) which file(s) would change.
4. **Specific quick wins** — at the top of the list, identify the 2 changes that would move the most J-score per hour of effort. These become the v0.3.1 priorities.

## Operational facts the next session needs

- **Repo:** https://github.com/DRCubix/RKEmemory (master `f06f6c4` + uncommitted `30d09dd` ChatMemory). Clone fresh.
- **Local checkout used this session:** /tmp/security-audit/RKEmemory (.venv has all deps installed; reuse if available).
- **Docker services for benchmarks:** Qdrant on 6433, FalkorDB on 6390. Restart with the commands in `scripts/start-services.sh` (point at alternate ports via `RKE_QDRANT_PORT`).
- **Datasets used this session** (re-clone if missing): /tmp/locomo (Snap LOCOMO), /tmp/longmemeval (xiaowu0162/LongMemEval), /tmp/msc.jsonl (HF MemGPT/MSC-Self-Instruct).
- **Eval scripts:** scripts/eval_locomo.py, scripts/eval_longmemeval.py, scripts/eval_dmr.py — all support `--limit N --judge {substring,anthropic,openai} --out PATH`.
- **API keys are NOT in this handoff.** The user will supply at session start. The eval scripts honor `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_JUDGE_MODEL` (or the Anthropic equivalents). Alibaba DashScope works at `https://coding-intl.dashscope.aliyuncs.com/v1` with model `qwen3.5-plus`.
- **CI:** GitHub Actions, master branch protected, all tests in `tests/` (95+ unit, 18 integration).
- **Known unfixed v0.3 limitations** (from Codex audit logs):
  - Concurrent multi-tenant for extractor and search_index in same process (singleton hooks).
  - BGE-M3 not yet head-to-head benchmarked (agent timed out mid-ingest).

## Prompt to start the next session

Copy-paste this verbatim into the next conversation:

> I'm continuing work on RKE (https://github.com/DRCubix/RKEmemory), an open-source AI agent memory system at v0.3. I just measured it against three published benchmarks and we're underperforming: DMR 85.0% (target 94.8%), LongMemEval 55.0% (target 90.4%), LOCOMO 30.0% (target 89.9%). Read `HANDOFF.md` in the repo first — it contains the full benchmark methodology, the architecture we already have, and concrete research questions.
>
> Your job in this session is research, not coding. Produce a single deliverable, `RESEARCH.md`, that:
> 1. For each of Zep, Mem0, Letta, Honcho: a half-page summary citing their paper(s), answering: retrieval architecture, answer composition, temporal handling, benchmark tuning, and their differentiating technique.
> 2. A gap-analysis table per benchmark naming the techniques used by the leader that RKE does NOT currently implement.
> 3. A ranked list of 5–8 concrete code changes RKE should make, sorted by expected J-score lift per engineering hour. For each: which benchmark it helps, effort estimate, which file(s).
> 4. A clearly-marked "two highest-leverage changes" section at the top of the ranked list — these become the v0.3.1 priorities.
>
> Use WebSearch / WebFetch / Hive Mind aggressively for paper analysis. Do NOT make up numbers — cite everything. The cleanest sources are the arXiv papers; treat marketing blogs as low-trust. The end of the session should be `RESEARCH.md` committed to the repo with a clear "ship the top 2 in v0.3.1" recommendation. Do not write code yet.
