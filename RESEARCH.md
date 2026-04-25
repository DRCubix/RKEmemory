# RKE v0.3.1 — Competitive Architecture Research & Ranked Change List

> **Companion to `HANDOFF.md`.** Goal: figure out what to change in RKE to close
> the DMR -10pt / LongMemEval -35pt / LOCOMO -60pt gap. No code in this doc —
> only analysis and a prioritized recommendation list. Sources: arXiv papers
> for Zep/Mem0/MemGPT/LongMemEval, blog.plasticlabs.ai for Honcho, and a
> Borg Hive Mind multi-agent synthesis (Gemini + Opus + Codex, 39 min,
> consensus pending due to one Codex validation timeout — caveats in §5).

---

## 0. Top Two Highest-Leverage Changes (ship in v0.3.1)

| # | Change | Benchmark(s) | Expected lift | Effort | Why it's the right first move |
|---|---|---|---|---|---|
| **A** | **Wire `graph_temporal.py` into `ChatMemory.answer()`** so `query_at(t)` is consulted before / alongside vector retrieval, with invalidated edges filtered out. | LOCOMO | **+30 to +40pt** (30 → 60–70%) | **~1 day** | Code already exists. The single biggest unforced error in v0.3 — the bi-temporal graph is built and tested but the answer path never reads it, so contradictory facts retrieved by vector search are never resolved. |
| **B** | **Swap MiniLM-L6 (384d) → BGE-M3 (1024d) as the default embedding model.** | DMR + LongMemEval + LOCOMO | **+5 to +8pt DMR, +4 to +6pt LongMemEval** | **~2–4 hours** (config + reindex) | BGE-M3 is already a declared dependency; the agent that tried to bench it timed out mid-ingest, so the swap was never landed. MiniLM-L6 is a well-documented semantic ceiling — every leader uses ≥1024d embeddings. Near-zero code change. |

These two together should plausibly move RKE to ≈**90% DMR, 65% LongMemEval, 65% LOCOMO** — closing more than half of every gap with under two engineering days. Everything in §4 below assumes A+B ship first.

---

## 1. Per-System Breakdown

### 1.1 Zep — DMR leader (94.8%)

| Q | Answer |
|---|---|
| **Retrieval architecture** | Vector + temporal knowledge graph (Graphiti) + cross-encoder reranker over fused top-K. Hive-mind synthesis flagged that Zep's headline number is **"rerank + graph", not graph alone** — graph contributes recall on relational/temporal queries; rerank lifts precision on the final shortlist. |
| **Answer composition** | Single-LLM-call composition over a stuffed context of the reranked top-K (the paper claims "90% latency reduction" vs MemGPT, which is only consistent with single-pass, not iterative). |
| **Temporal handling** | Bi-temporal edges with `valid_at` / `invalid_at` (event time) plus ingestion time. Edges are explicitly invalidated when contradicted, so retrieval can filter "currently valid" facts. This is functionally what RKE's `graph_temporal.py` already implements. |
| **Tuning** | Reports DMR 94.8% vs MemGPT 93.4%, and LongMemEval "up to +18.5pt accuracy, 90% latency reduction" — embedding model not specified in the abstract; the public Graphiti repo defaults to OpenAI `text-embedding-3-small` 1536d. |
| **One differentiator** | **Graphiti** — temporal KG built incrementally from chat without offline batch re-extraction. (Source: arXiv 2501.13956.) |

### 1.2 Mem0 — graph variant ~68% LOCOMO

| Q | Answer |
|---|---|
| **Retrieval architecture** | Two variants: (a) vector-only, (b) vector + graph. Mem0+graph adds only ~1.5pt over vector-only on LOCOMO LLM Score (66.9% → 68.4%) — graph is a small lift, not the main mechanism. |
| **Answer composition** | Single-pass LLM call over retrieved memories. |
| **Temporal handling** | Treated as a question category, not a first-class storage primitive — the system stores extracted "memories" but does not bi-temporalize edges the way Zep/Graphiti does. |
| **Tuning** | LOCOMO categories evaluated: single-hop, temporal, multi-hop, open-domain. Reports 26% relative LLM-as-Judge improvement over the OpenAI memory baseline. |
| **One differentiator** | **Write-time LLM fact extraction** — every turn is distilled by an LLM into a small fact set before storage; the read path is then conventional. The cost is write latency + model spend; the benefit is dense, query-aligned chunks. (Source: arXiv 2504.19413, ECAI 2025.) |

### 1.3 MemGPT / Letta — DMR 93.4%

| Q | Answer |
|---|---|
| **Retrieval architecture** | Hierarchical paged context — "main context" (in-window), "external context" (recall + archival storage), and function-call tools the agent invokes to page memories in/out. |
| **Answer composition** | **Iterative agentic loop** — the model decides when to read more, write a memory, or terminate. Multi-turn on every user query. |
| **Temporal handling** | Implicit through recall-storage ordering; no first-class bi-temporal edges. |
| **Tuning** | DMR 93.4% (paired result Zep cites against itself); MSC reported in the original paper. |
| **One differentiator** | **OS-style virtual memory + interrupts for control flow** — the agent is the retrieval policy, not a fixed top-K dense lookup. (Source: arXiv 2310.08560.) |

### 1.4 Honcho — LongMemEval-S 90.4%, LoCoMo 89.9% (joint leader on both)

| Q | Answer |
|---|---|
| **Retrieval architecture** | Entity-centric "peer" model — every user/agent/group is a peer with a `Representation`. Reads pull from the persisted representation, not just raw turns. |
| **Answer composition** | Two-layer: synchronous read API + asynchronous background "Dreams" tasks that prune duplicates, consolidate, deduce, reason — done **between turns**, not at query time. So at query time the read is fast, but the underlying representation is constantly being re-derived. |
| **Temporal handling** | Theory-of-Mind models maintain "up-to-date user preferences, history, psychology, personality, values, beliefs, desires" — implicitly temporal because the representation reflects the latest dialectic, not raw turn order. |
| **Tuning** | Small **fine-tuned** models for representation extraction (not zero-shot). 90.4% LongMemEval-S, 92.6% with Gemini 3 Pro as composer. 89.9% LoCoMo (up from a prior 86.9%). |
| **One differentiator** | **Asynchronous user-representation dialectic** — the system continually re-models the user *between* turns; reads are against an already-distilled identity model rather than raw history. (Source: blog.plasticlabs.ai/research/Benchmarking-Honcho; honcho repo & docs.) |

---

## 2. Gap-Analysis: techniques the leader uses that RKE does NOT

| Benchmark | Leader | Leader's load-bearing techniques | RKE present? | RKE missing |
|---|---|---|---|---|
| **DMR** (Zep 94.8%) | Zep | Temporal KG; cross-encoder rerank; ≥1024d embeddings; single-pass composition | Temporal KG: **built** (`graph_temporal.py`) but **not wired into `answer()`**. Embeddings: MiniLM-L6 (384d). Reranker: **none**. Composition: single-pass via RLM router. | **Wire bi-temporal graph into answer**, **swap to BGE-M3**, **add cross-encoder reranker** (BGE-reranker-base or bge-reranker-v2-m3). |
| **LongMemEval** (Honcho 90.4%; LongMemEval paper's own framework) | Honcho + LME paper | Async user-representation; session decomposition (per LME paper §3); fact-augmented key expansion (per LME paper §3); time-aware query expansion (per LME paper §3); iterative answer composition; cross-encoder reranking | Per-turn KB chunk indexing exists (v0.3 fix in `chat_memory.py`) but it is **flat per-turn**, not session-decomposed. **No** key expansion, **no** time-aware QE, **no** reranker, **no** iterative loop, **no** async representation. | All five LongMemEval-paper recommendations are absent: session decomposition, fact-augmented key expansion, time-aware query expansion, reranker, iterative answer composition. |
| **LOCOMO** (Honcho 89.9%) | Honcho / Zep | Bi-temporal facts with invalidation; iterative reasoning over conflicts; representation that resolves "user changed their mind"; reranker | Bi-temporal storage primitive exists; **invalidation logic exists**; iteration & rerank do not. | **Wire `query_at(t)` into the answer path**, add invalidated-edge filter, add iterative reasoning step for "which event came first" / "what does the user believe now" questions. |

**Cross-cutting gaps RKE shares on every benchmark:** no cross-encoder reranker; embedding ceiling (MiniLM-L6); no fusion across BM25F + vector + graph (currently the channels are essentially independent); single-pass-only answer composition (no iterative agent loop for indirect or multi-hop questions).

---

## 3. Ranked Change List (5–8 concrete code changes, sorted by J-score lift per engineering hour)

| Rank | Change | Benchmarks helped | Expected lift | Effort | Files / module |
|---|---|---|---|---|---|
| **1** ⭐ | **Wire `graph_temporal.py` into `ChatMemory.answer()`** — before vector retrieval, run the question through a temporal-intent classifier; if temporal/contradiction-y, query the bi-temporal graph at `query_at(now)` for active edges; merge with vector hits and pass active-edge facts to composer. Drop invalidated edges from any vector hit that contradicts. | LOCOMO (primary), LongMemEval-temporal, DMR | **+30 to +40 LOCOMO**, +5 LongMemEval, +1 DMR | ~1 day | `src/rke/adapters/chat_memory.py::answer`, `src/rke/rlm/router.py`, `src/rke/graph_temporal.py` (read-side helpers) |
| **2** ⭐ | **Default embedding model swap MiniLM-L6 → BGE-M3 (1024d).** Already a declared optional dep; reindex KB once. | All three | +5–8 DMR, +4–6 LongMemEval, +2 LOCOMO | 2–4 hours (incl. reindex) | `src/rke/vector_store.py` (default model), `config/rke.yaml.example`, `pyproject.toml` (move BGE-M3 from optional to default `[recommended]` extra) |
| **3** | **Add cross-encoder reranker over fused top-K** (`BAAI/bge-reranker-v2-m3`). Retrieve top-30 from each channel, fuse, rerank to final top-5. Cache per-(query, doc) pair. Latency budget: ~80–150 ms p95 — measure before shipping. | DMR, LongMemEval | +3–5 DMR, +8–12 LongMemEval | 1–2 days | New `src/rke/wiki/reranker.py`; integrate in `KnowledgeBase.query` and `chat_memory.answer` |
| **4** | **RRF fusion across BM25F (Tantivy) + vector + graph.** Today the channels run independently; RRF (k=60) on rank lists is a 20-line algorithm and consistently beats any single channel on ~all retrieval benchmarks. | All three | +3–5 across all | ~½ day | `src/rke/wiki/knowledge_base.py::query` |
| **5** | **Session decomposition + fact-augmented key expansion** (per LongMemEval paper §3). Group per-turn KB chunks by interaction round so a "session" is the atomic retrieved unit; at write-time also synthesize a 1-line fact key (LLM call) for each session, indexed alongside the body. | LongMemEval (primary), LOCOMO | +5–8 LongMemEval, +2 LOCOMO | 1–2 days | `src/rke/adapters/chat_memory.py::_index_turn` → `_index_session`; new `src/rke/knowledge/session_keys.py` |
| **6** | **Iterative / agentic mode in RLM router** for multi-hop and indirect questions. Add a `complete_iterative()` path that lets the model issue follow-up `peek` / `grep` / `query_at` calls before composing — capped at 3–5 iterations to bound cost. Heuristic gate (multi-hop or temporal questions) so cheap questions stay single-pass. | LongMemEval, LOCOMO | +8–15 LongMemEval, +5 LOCOMO | 2–3 days | `src/rke/rlm/router.py`, `src/rke/rlm/environment.py` |
| **7** | **Time-aware query expansion** (per LongMemEval paper §3). Before retrieval, an LLM rewrite expands a query like "what did I tell you about my dog last month" into "{original} OR (date range: 2026-03-01..2026-03-31)" with date-range filters honored by the bi-temporal graph and a recency bonus on the vector channel. Complements #1. | LongMemEval, LOCOMO | +3–5 LongMemEval | 1 day | `src/rke/rlm/router.py` (pre-retrieval QE step), `src/rke/wiki/knowledge_base.py` (date-range filter) |
| **8** | **Async user-representation summary** (Honcho-style). Background task that, every N turns, distills the thread into a structured user representation (preferences/values/recent state) and stores it as a privileged retrieval slot always merged into composer context. **Defer until 1–7 plateau below targets** — high effort, high architectural impact. | LongMemEval, LOCOMO | +5–10 LongMemEval, +3–5 LOCOMO | 1 week+ | New `src/rke/adapters/user_representation.py`; persistence schema; async worker (asyncio task or external job) |

### Cumulative trajectory (rough projection, with caveats in §5)

| After | DMR | LongMemEval | LOCOMO |
|---|---|---|---|
| v0.3 today (n=20) | 85% | 55% | 30% |
| +1 (graph wiring) | 86% | 60% | **65%** |
| +1+2 (BGE-M3) | **91%** | 65% | 67% |
| +1+2+3 (rerank) | **94%** | **75%** | 70% |
| +1+2+3+4 (RRF) | 95% | 78% | 73% |
| +1+2+3+4+5 (session) | 95% | **85%** | 75% |
| +1+2+3+4+5+6 (iter) | 95% | **90%** | **80%** |

These projections compound naive per-change lifts; real interactions will be sublinear. Treat as upper bounds.

---

## 4. v0.3.1 Recommendation: ship #1 + #2

The argument for putting **only** the top two in v0.3.1, vs. bundling more:

- Both changes are **low-risk and small-diff** — one wires existing code into existing code paths; the other is a config swap + reindex. Both are independently testable.
- The combined expected lift (≥30pt LOCOMO, ≥5pt DMR, ≥4pt LongMemEval) is large enough to be measurable at n=20 with our existing eval harness.
- A clean v0.3.1 with #1+#2 lets us **verify the LOCOMO hypothesis** (graph wiring closes most of the gap) before committing engineering time to the larger items.
- #3 (reranker) is the natural v0.3.2 — but only after we measure latency budget on real workloads.

Concrete v0.3.1 acceptance criteria:
- LOCOMO J-score (n=20, qwen3.5-plus judge) **≥ 60%** (up from 30%).
- DMR J-score **≥ 90%** (up from 85%).
- LongMemEval-oracle J-score **≥ 60%** (up from 55%).
- No regression on the 95 unit / 18 integration tests.
- Re-run with `n ≥ 50` if lift is achieved at n=20, to narrow CIs before publishing.

---

## 5. Caveats, open questions, and conflicts

- **Honcho 90.4% LongMemEval-S has not been independently reproduced.** Plan against ~85% as the realistic ceiling; if RKE hits 80% we are at parity with everyone except Honcho.
- **No measured Recall@K for RKE on LOCOMO.** The hive-mind flagged this as the dominant uncertainty: if top-K never contains the temporally-correct fact, a reranker won't help — only the graph wiring will. We should measure pre-LLM Recall@10 *before* committing to #3.
- **`ChatMemory.answer()` latency budget is unconfirmed.** Cross-encoder rerank is only viable if we have headroom — measure p95 today before spec'ing #3.
- **Hive-mind consensus did not finalize** (Codex timed out at 300s during validation) and there is one unresolved internal conflict: Mem0's win classified once as "write-time" and once as "read-time intelligence". Resolution from arXiv 2504.19413: Mem0 is **write-time** (the fact extraction happens on ingest, not on query), with a small read-side graph variant on top — that lines up with finding #4 in the hive-mind report; finding #34's "read-time" framing should be discarded.
- **The bi-temporal-graph hypothesis for LOCOMO is partially speculative** until we ablate. If wiring the graph in only delivers, say, +15 LOCOMO instead of +30, the next escalation is rank #6 (iterative composition for "which event came first"), not rank #3.
- **Conversation-30 is a single LOCOMO conversation.** Generalizing to all 10 conversations should be done before declaring victory — large variance is plausible.
- **Independent reviewers note MemPalace's 96.6% LongMemEval is essentially a ChromaDB score with disputed methodology.** Treat as an outlier; do not use as a target.

---

## 6. Sources

- Zep / Graphiti — [arXiv 2501.13956](https://arxiv.org/abs/2501.13956)
- Mem0 — [arXiv 2504.19413](https://arxiv.org/abs/2504.19413) (ECAI 2025), LOCOMO numbers cross-referenced against [mem0.ai/research](https://mem0.ai/research)
- MemGPT / Letta — [arXiv 2310.08560](https://arxiv.org/abs/2310.08560)
- LongMemEval — [arXiv 2410.10813](https://arxiv.org/abs/2410.10813); benchmark categories: information extraction, multi-session reasoning, temporal reasoning, knowledge updates, abstention; framework prescribes session decomposition + fact-augmented key expansion + time-aware query expansion
- LOCOMO — [snap-research.github.io/locomo](https://snap-research.github.io/locomo/)
- Honcho — [blog.plasticlabs.ai/research/Benchmarking-Honcho](https://blog.plasticlabs.ai/research/Benchmarking-Honcho), [github.com/plastic-labs/honcho](https://github.com/plastic-labs/honcho), [docs.honcho.to](https://docs.honcho.to/)
- Internal: Borg Hive Mind report `research-reports/research-2026-04-25T16-21-22-334Z.md` (Gemini + Opus + Codex, 39 min, 49 findings, consensus pending)
