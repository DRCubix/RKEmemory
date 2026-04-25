"""RLMRouter — orchestrates RLM iterations over a knowledge environment.

Without an LLM provider configured, the router runs in **deterministic mode**:
it does a single retrieval pass and returns a synthesized context block.

With an LLM provider configured (RKE_LLM_PROVIDER + an API key), the router
runs an iterative loop:
    while not done and within budget:
        env.peek/grep/partition/sub_rlm to gather evidence
        ask the LLM to either produce an answer or request another op
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from ..config import Config, load_config
from ..wiki.knowledge_base import KnowledgeBase
from .environment import Environment, Trace

log = logging.getLogger(__name__)


@dataclass
class RLMResult:
    answer: str
    iterations: int = 0
    trace: list[Trace] = field(default_factory=list)
    cost_usd: float = 0.0
    used_llm: bool = False


class RLMRouter:
    def __init__(
        self,
        config: Config | None = None,
        kb: KnowledgeBase | None = None,
        env: Environment | None = None,
    ) -> None:
        self.config = config or load_config()
        if env is not None:
            self.env = env
        else:
            self.env = Environment(kb=kb or KnowledgeBase(self.config), config=self.config)

    # ── Public entry points ──────────────────────────────────────
    def complete(
        self,
        query: str,
        agent: str | None = None,
        _depth: int | None = None,
    ) -> RLMResult:
        provider = os.getenv("RKE_LLM_PROVIDER") or self.config.rlm.get("provider")
        if provider:
            try:
                return self._complete_with_llm(query, agent=agent, provider=str(provider))
            except Exception as exc:
                log.warning("LLM RLM failed (%s); falling back to deterministic mode", exc)
        return self._complete_deterministic(query)

    # ── Deterministic fallback ───────────────────────────────────
    def _complete_deterministic(self, query: str) -> RLMResult:
        hits = self.env.kb.query(query, wiki_limit=3, vector_limit=5)
        if not hits:
            answer = f"No knowledge found for: {query}"
        else:
            blocks = []
            for i, h in enumerate(hits, 1):
                blocks.append(
                    f"[{i}] ({h.source}, score={h.score:.2f}) {h.title}\n{h.snippet}".strip()
                )
            answer = (
                f"Question: {query}\n\n"
                f"Top {len(hits)} knowledge hits:\n\n" + "\n\n---\n\n".join(blocks)
            )
        return RLMResult(answer=answer, iterations=1, trace=list(self.env.trace), used_llm=False)

    # ── LLM-driven path ──────────────────────────────────────────
    def _complete_with_llm(self, query: str, agent: str | None, provider: str) -> RLMResult:
        max_iter = int(self.config.rlm.get("max_iterations", 20))
        budget = float(self.config.rlm.get("max_cost_usd", 1.00))
        client = _build_llm_client(provider, self.config)

        # Seed with a deterministic retrieval so the LLM has grounding.
        seed_hits = self.env.kb.query(query, wiki_limit=3, vector_limit=5)
        seed_block = "\n\n".join(
            f"[{i+1}] {h.source} • {h.title}\n{h.snippet}"
            for i, h in enumerate(seed_hits)
        ) or "(no initial hits)"

        system = (
            "You are an RLM agent. You may answer directly or call ONE of: "
            "PEEK <slug>, GREP <pattern>, PARTITION <query>, SUB_RLM <question>. "
            "When you have enough evidence, reply with 'ANSWER: <final answer>'."
        )
        history = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Question: {query}\n\nInitial evidence:\n{seed_block}"},
        ]

        cost = 0.0
        for i in range(1, max_iter + 1):
            reply, step_cost = client.complete(history)
            cost += step_cost
            if cost > budget:
                return RLMResult(
                    answer="(stopped: cost budget exceeded)",
                    iterations=i, trace=list(self.env.trace),
                    cost_usd=cost, used_llm=True,
                )
            history.append({"role": "assistant", "content": reply})
            stripped = reply.strip()
            if stripped.upper().startswith("ANSWER:"):
                return RLMResult(
                    answer=stripped[len("ANSWER:"):].strip(),
                    iterations=i, trace=list(self.env.trace),
                    cost_usd=cost, used_llm=True,
                )
            tool_result = self._dispatch_tool(stripped)
            history.append({"role": "user", "content": f"OBSERVATION:\n{tool_result}"})
        return RLMResult(
            answer="(stopped: iteration cap reached)",
            iterations=max_iter, trace=list(self.env.trace),
            cost_usd=cost, used_llm=True,
        )

    def _dispatch_tool(self, command: str) -> str:
        head, _, rest = command.partition(" ")
        head = head.strip().upper().rstrip(":")
        rest = rest.strip()
        try:
            if head == "PEEK":
                return self.env.peek(rest)
            if head == "GREP":
                return str(self.env.grep(rest))
            if head == "PARTITION":
                return "\n".join(self.env.partition(rest))
            if head == "SUB_RLM":
                return self.env.sub_rlm(rest)
        except Exception as exc:
            return f"(tool error: {exc})"
        return f"(unknown command: {head})"


# ── Pluggable LLM clients ────────────────────────────────────────

class _LLMClient:
    def complete(self, messages: list[dict]) -> tuple[str, float]:
        raise NotImplementedError


class _AnthropicClient(_LLMClient):
    def __init__(self, model: str) -> None:
        import anthropic  # type: ignore
        self._mod = anthropic
        self._client = anthropic.Anthropic()
        self.model = model

    def complete(self, messages: list[dict]) -> tuple[str, float]:
        sys_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_msgs = [m for m in messages if m["role"] != "system"]
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=sys_msg,
            messages=user_msgs,
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        usage = resp.usage
        # Approximate: Sonnet pricing-ish; user can refine.
        cost = (usage.input_tokens * 3e-6) + (usage.output_tokens * 1.5e-5)
        return text, cost


class _OpenAIClient(_LLMClient):
    def __init__(self, model: str) -> None:
        import openai  # type: ignore
        self._client = openai.OpenAI()
        self.model = model

    def complete(self, messages: list[dict]) -> tuple[str, float]:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=1024,
        )
        text = resp.choices[0].message.content or ""
        u = resp.usage
        cost = (u.prompt_tokens * 5e-6) + (u.completion_tokens * 1.5e-5)
        return text, cost


def _build_llm_client(provider: str, config: Config) -> _LLMClient:
    p = provider.lower().strip()
    model = os.getenv("RKE_LLM_MODEL") or config.rlm.get("model")
    if p == "anthropic":
        return _AnthropicClient(model or "claude-sonnet-4-6")
    if p == "openai":
        return _OpenAIClient(model or "gpt-4o-mini")
    raise ValueError(f"unsupported LLM provider: {provider}")
