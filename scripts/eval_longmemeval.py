#!/usr/bin/env python3
"""LLM-judge eval harness for RKE on LongMemEval (oracle variant).

--judge substring | anthropic | openai
"""
from __future__ import annotations

import argparse
import json
import os
import re
import string
import sys
import time
import traceback
import uuid
from collections import Counter
from pathlib import Path

SCRATCH = Path(f"/tmp/rke_lme_eval_{uuid.uuid4().hex[:8]}")
SCRATCH.mkdir(parents=True, exist_ok=True)
os.environ["RKE_WIKI_PATH"] = str(SCRATCH / "wiki")
os.environ["RKE_WIKI__PATH"] = str(SCRATCH / "wiki")
os.environ["RKE_QDRANT__HOST"] = "localhost"
os.environ["RKE_QDRANT__PORT"] = "6433"
os.environ["RKE_QDRANT_PORT"] = "6433"
os.environ["RKE_FALKORDB__HOST"] = "localhost"
os.environ["RKE_FALKORDB__PORT"] = "6390"
os.environ["RKE_FALKORDB_PORT"] = "6390"
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

DATA_PATH = "/tmp/longmemeval/data/longmemeval_oracle.json"


def _norm(s: str) -> str:
    s = s.lower()
    s = "".join(c for c in s if c not in string.punctuation)
    s = re.sub(r"\s+", " ", s)
    return " ".join(t for t in s.split() if t not in {"a", "an", "the"})


def f1_substring(pred: str, gold: str) -> float:
    p = _norm(pred).split()
    g = _norm(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    return 2 * (same / len(p)) * (same / len(g)) / (same / len(p) + same / len(g))


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
    raise ValueError(judge)


def session_to_text(turns: list[dict]) -> str:
    return "\n".join(f"{t.get('role', '?')}: {t.get('content', '')}"
                     for t in turns)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--judge", choices=["anthropic", "openai", "substring"],
                    default="substring")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    raw = json.loads(Path(DATA_PATH).read_text())
    items = [q for q in raw if not q["question_id"].endswith("_abs")][:args.limit]
    print(f"[lme] judge={args.judge}  limit={args.limit}  data={DATA_PATH}",
          flush=True)
    print(f"[lme] {len(items)} non-abstention questions", flush=True)

    rows = []
    scores = []
    t0 = time.time()
    for i, q in enumerate(items):
        qid = q["question_id"]
        question = q["question"]
        gold = str(q["answer"])
        sessions = q.get("haystack_sessions", [])

        cfg = load_config()
        cfg.raw.setdefault("qdrant", {})["collection"] = f"lme_q{i:02d}_{int(time.time())}"
        cfg.raw.setdefault("falkordb", {})["graph"] = f"lme_q{i:02d}"
        cfg.raw.setdefault("wiki", {})["path"] = str(SCRATCH / f"wiki_q{i:02d}")

        try:
            kb = KnowledgeBase(config=cfg)
            mem = ChatMemory(thread_id=f"lme-{i}", kb=kb)
            for sid, sess in enumerate(sessions):
                body = session_to_text(sess)
                if body.strip():
                    kb.add_page(
                        title=f"session-{sid}",
                        body=body,
                        category=f"lme/{i}",
                        tags=["lme", qid],
                    )
            pred = mem.answer(question)
        except Exception as exc:
            pred = ""
            print(f"[warn] q{i} {qid}: {exc.__class__.__name__}: {exc}",
                  file=sys.stderr)
            traceback.print_exc(limit=1, file=sys.stderr)

        try:
            s = score(args.judge, question, gold, pred)
        except Exception as e:
            print(f"[judge err q{i}] {e}", flush=True)
            s = 0.0
        scores.append(s)
        rows.append({"qid": qid, "q": question, "gold": gold,
                     "pred": pred[:500], "score": s})
        print(f"[{i+1:02d}/{len(items)}] {time.time()-t0:5.1f}s "
              f"score={s:.3f} qid={qid}", flush=True)

    mean = sum(scores) / len(scores) if scores else 0.0
    summary = {
        "dataset": "LongMemEval (oracle)",
        "data_path": DATA_PATH,
        "judge": args.judge,
        "n": len(scores),
        "mean_score": mean,
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print("\n=== LongMemEval RESULTS ===")
    print(f"n={len(scores)}  mean_score={mean:.4f}  judge={args.judge}")
    print(f"data={DATA_PATH}")
    print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
