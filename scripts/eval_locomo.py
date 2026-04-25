#!/usr/bin/env python3
"""LLM-judge eval harness for RKE on LOCOMO.

Works with OR without an LLM API key:
  --judge substring : SQuAD-style token-overlap F1 (default, no key needed)
  --judge anthropic : ANTHROPIC_API_KEY required, mean of 0/1 verdicts
  --judge openai    : OPENAI_API_KEY required, mean of 0/1 verdicts

Replaces the sandbox script's "top-1 snippet as answer" with mem.answer(question).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import string
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

# ── isolation: scratch env BEFORE importing rke ───────────────────────────────
SCRATCH = Path(f"/tmp/rke_locomo_eval_{uuid.uuid4().hex[:8]}")
SCRATCH.mkdir(parents=True, exist_ok=True)
(SCRATCH / "wiki").mkdir(exist_ok=True)
os.environ["RKE_WIKI_PATH"] = str(SCRATCH / "wiki")
os.environ["RKE_WIKI__PATH"] = str(SCRATCH / "wiki")
os.environ["RKE_QDRANT__HOST"] = "localhost"
os.environ["RKE_QDRANT__PORT"] = "6433"
os.environ["RKE_QDRANT_PORT"] = "6433"
os.environ["RKE_QDRANT__COLLECTION"] = f"locomo_eval_{int(time.time())}"
os.environ["RKE_FALKORDB__HOST"] = "localhost"
os.environ["RKE_FALKORDB__PORT"] = "6390"
os.environ["RKE_FALKORDB_PORT"] = "6390"
os.environ["RKE_FALKORDB__GRAPH"] = f"locomo_eval_{int(time.time())}"
os.environ.setdefault("RKE_EMBEDDING_MODEL",
                      "sentence-transformers/all-MiniLM-L6-v2")
os.environ.setdefault("RKE_EMBEDDING_DIMENSIONS", "384")
os.environ["RKE_EMBEDDING__MODEL"] = os.environ["RKE_EMBEDDING_MODEL"]
os.environ["RKE_EMBEDDING__DIMENSIONS"] = os.environ["RKE_EMBEDDING_DIMENSIONS"]
os.environ["RKE_EMBEDDING__DEVICE"] = "cpu"
os.environ["RKE_LOGGING__LEVEL"] = "WARNING"

RKE_ROOT = Path("/tmp/security-audit/RKEmemory")
sys.path.insert(0, str(RKE_ROOT / ".venv/lib/python3.12/site-packages"))
sys.path.insert(0, str(RKE_ROOT / "src"))

from rke.adapters.chat_memory import ChatMemory  # noqa: E402
from rke.config import load_config  # noqa: E402
from rke.wiki.knowledge_base import KnowledgeBase  # noqa: E402

DATA_PATH = "/tmp/locomo/data/locomo10.json"


# ── judges ────────────────────────────────────────────────────────────────────
def _norm(s: str) -> str:
    s = s.lower()
    s = "".join(c for c in s if c not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1_substring(pred: str, gold: str) -> float:
    p = _norm(pred).split()
    g = _norm(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec = same / len(p)
    rec = same / len(g)
    return 2 * prec * rec / (prec + rec)


_JUDGE_SYSTEM = (
    "You are an LLM judge. Reply 1 if the predicted answer is semantically "
    "correct vs the gold, else 0. Reply with only the digit."
)


def _judge_user(q: str, gold: str, pred: str) -> str:
    return f"Question: {q}\nGold: {gold}\nPredicted: {pred}\nVerdict (0 or 1):"


def judge_anthropic(q: str, gold: str, pred: str) -> float:
    import anthropic  # type: ignore
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model=os.environ.get("ANTHROPIC_JUDGE_MODEL", "claude-3-5-haiku-latest"),
        max_tokens=4,
        system=_JUDGE_SYSTEM,
        messages=[{"role": "user", "content": _judge_user(q, gold, pred)}],
    )
    txt = "".join(b.text for b in resp.content if hasattr(b, "text")).strip()
    return 1.0 if txt.startswith("1") else 0.0


def judge_openai(q: str, gold: str, pred: str) -> float:
    from openai import OpenAI  # type: ignore
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=os.environ.get("OPENAI_JUDGE_MODEL", "gpt-4o-mini"),
        max_tokens=4,
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": _judge_user(q, gold, pred)},
        ],
    )
    txt = (resp.choices[0].message.content or "").strip()
    return 1.0 if txt.startswith("1") else 0.0


def score(judge: str, q: str, gold: str, pred: str) -> float:
    if judge == "substring":
        return f1_substring(pred, gold)
    if judge == "anthropic":
        return judge_anthropic(q, gold, pred)
    if judge == "openai":
        return judge_openai(q, gold, pred)
    raise ValueError(f"unknown judge: {judge}")


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--judge", choices=["anthropic", "openai", "substring"],
                    default="substring")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(DATA_PATH) as f:
        data = json.load(f)

    # Use the smallest sample for tractable ingest within wall budget.
    sample = next(s for s in data if s["sample_id"] == "conv-30")
    print(f"[locomo] sample_id={sample['sample_id']}  judge={args.judge}  "
          f"limit={args.limit}  data={DATA_PATH}", flush=True)

    cfg = load_config()
    kb = KnowledgeBase(cfg)
    mem = ChatMemory(thread_id=f"locomo-{sample['sample_id']}", kb=kb)

    sess_keys = sorted(
        [k for k in sample["conversation"].keys()
         if k.startswith("session_") and not k.endswith("_date_time")],
        key=lambda x: int(x.split("_")[1]),
    )
    speaker_a = sample["conversation"]["speaker_a"]
    # Ingest budget proportional to limit (cap to keep wall time reasonable).
    max_turns = max(40, args.limit * 4)
    ingested = 0
    t_ing = time.time()
    for sk in sess_keys:
        if ingested >= max_turns:
            break
        date = sample["conversation"].get(f"{sk}_date_time", "")
        for turn in sample["conversation"][sk]:
            if ingested >= max_turns:
                break
            text = f"[{date}] {turn['speaker']}: {turn['text']}"
            role = "user" if turn["speaker"] == speaker_a else "assistant"
            try:
                mem.add_message(role, text, dia_id=turn.get("dia_id", ""))
                ingested += 1
            except Exception as e:
                print(f"[ingest err] {e}", flush=True)
                break
    print(f"[locomo] ingested {ingested} turns in {time.time()-t_ing:.1f}s",
          flush=True)

    # Pick factual single-hop QAs
    qas = [qa for qa in sample["qa"]
           if qa.get("category") in (1, 2, 3) and "answer" in qa][:args.limit]

    rows = []
    scores = []
    for i, qa in enumerate(qas):
        q = qa["question"]
        gold = str(qa["answer"])
        try:
            pred = mem.answer(q)
        except Exception as e:
            pred = ""
            print(f"[answer err q{i}] {e}", flush=True)
        try:
            s = score(args.judge, q, gold, pred)
        except Exception as e:
            print(f"[judge err q{i}] {e}", flush=True)
            s = 0.0
        scores.append(s)
        rows.append({"q": q, "gold": gold, "pred": pred[:500], "score": s})
        print(f"  q{i:02d} score={s:.3f}", flush=True)

    mean = sum(scores) / len(scores) if scores else 0.0
    summary = {
        "dataset": "LOCOMO",
        "data_path": DATA_PATH,
        "sample_id": sample["sample_id"],
        "judge": args.judge,
        "n": len(scores),
        "mean_score": mean,
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print("\n=== LOCOMO RESULTS ===")
    print(f"n={len(scores)}  mean_score={mean:.4f}  judge={args.judge}")
    print(f"data={DATA_PATH}")
    print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
