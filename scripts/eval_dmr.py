#!/usr/bin/env python3
"""LLM-judge eval harness for RKE on DMR (MSC-Self-Instruct).

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
import uuid
from collections import Counter
from pathlib import Path

SCRATCH = Path(f"/tmp/rke_dmr_eval_{uuid.uuid4().hex[:8]}")
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

DATA_PATH = "/tmp/msc.jsonl"
WORD_RE = re.compile(r"[a-z0-9]+")


def f1_substring(pred: str, gold: str) -> float:
    p = WORD_RE.findall(pred.lower())
    g = WORD_RE.findall(gold.lower())
    if not p or not g:
        return 0.0
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
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
    raise ValueError(judge)


def load_msc(path: str, n: int):
    items = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            si = d.get("self_instruct") or {}
            q = si.get("B")
            a = si.get("A")
            prev = d.get("previous_dialogs") or []
            if not q or not a or not prev:
                continue
            items.append({"q": q, "a": a, "prev": prev})
            if len(items) >= n:
                break
    return items


def ingest_sessions(cm: ChatMemory, prev_sessions):
    for sess_idx, sess in enumerate(prev_sessions):
        if isinstance(sess, dict):
            turns = sess.get("dialog") or sess.get("turns") or []
        elif isinstance(sess, list):
            turns = sess
        else:
            turns = []
        for t in turns:
            if isinstance(t, dict):
                text = t.get("text", "")
                speaker = t.get("id", "Speaker")
            else:
                text = str(t)
                speaker = "Speaker"
            if not text:
                continue
            role = "user" if "1" in str(speaker) else "assistant"
            cm.add_message(role, f"[session {sess_idx}] {text}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--judge", choices=["anthropic", "openai", "substring"],
                    default="substring")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    items = load_msc(DATA_PATH, args.limit)
    print(f"[dmr] judge={args.judge}  limit={args.limit}  data={DATA_PATH}",
          flush=True)
    print(f"[dmr] loaded {len(items)} items", flush=True)

    scores = []
    rows = []
    t0 = time.time()
    for i, item in enumerate(items):
        coll = f"dmr_eval_{uuid.uuid4().hex[:8]}"
        os.environ["RKE_QDRANT_COLLECTION"] = coll
        os.environ["RKE_QDRANT__COLLECTION"] = coll
        cfg = load_config()
        kb = KnowledgeBase(cfg)
        cm = ChatMemory(thread_id=f"dmr-{i}", kb=kb,
                        buffer_size=10, summarize_threshold=10000)
        try:
            ingest_sessions(cm, item["prev"])
            pred = cm.answer(item["q"])
        except Exception as e:
            pred = ""
            print(f"  q{i} ERROR: {e}", flush=True)
        try:
            s = score(args.judge, item["q"], item["a"], pred)
        except Exception as e:
            print(f"  q{i} judge ERROR: {e}", flush=True)
            s = 0.0
        scores.append(s)
        rows.append({"q": item["q"], "gold": item["a"],
                     "pred": pred[:500], "score": s})
        try:
            kb.vectors.client.delete_collection(coll)
        except Exception:
            pass
        print(f"  q{i:2d} score={s:.3f} (elapsed={time.time()-t0:.0f}s)",
              flush=True)

    mean = sum(scores) / len(scores) if scores else 0.0
    summary = {
        "dataset": "DMR (MSC-Self-Instruct)",
        "data_path": DATA_PATH,
        "judge": args.judge,
        "n": len(scores),
        "mean_score": mean,
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print("\n=== DMR RESULTS ===")
    print(f"n={len(scores)}  mean_score={mean:.4f}  judge={args.judge}")
    print(f"data={DATA_PATH}")
    print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
