"""Git repo ingestion — scan a directory of cloned repos (or a single repo)
and ingest matching files into the knowledge base.

Optionally clones remote URLs into a local cache directory.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

from ..config import load_config
from ..wiki.knowledge_base import KnowledgeBase
from .knowledge import iter_files

log = logging.getLogger(__name__)


def _has_git() -> bool:
    return shutil.which("git") is not None


def clone_repo(url: str, dest: Path, branch: str | None = None) -> Path:
    if not _has_git():
        raise RuntimeError("git is not installed")
    if dest.exists() and (dest / ".git").exists():
        log.info("git: %s exists, fetching", dest)
        subprocess.run(["git", "-C", str(dest), "fetch", "--depth", "1"], check=True)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "--depth", "1"]
    if branch:
        cmd += ["--branch", branch]
    cmd += [url, str(dest)]
    log.info("git: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return dest


def list_repos(root: Path) -> list[Path]:
    """Find any directory under `root` that contains a `.git` dir."""
    if not root.exists():
        return []
    if (root / ".git").exists():
        return [root]
    out: list[Path] = []
    for sub in root.iterdir():
        if sub.is_dir() and (sub / ".git").exists():
            out.append(sub)
    return out


def ingest_repo(kb: KnowledgeBase, repo: Path, includes: list[str], excludes: list[str]) -> int:
    """Ingest text-y files from a single repo. Returns # pages indexed."""
    files = iter_files([str(repo)], includes, excludes)
    n = 0
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if not text.strip():
            continue
        rel = f.relative_to(repo)
        title = f"{repo.name}: {rel}"
        body = f"_Repo: {repo.name}_  \n_File: `{rel}`_\n\n```\n{text[:8000]}\n```"
        try:
            kb.add_page(
                title=title,
                body=body,
                category=f"code/{repo.name}",
                tags=["code", repo.name, f.suffix.lstrip(".")],
            )
            n += 1
        except Exception as exc:
            log.warning("ingest %s failed: %s", f, exc)
    return n


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser("rke.ingestion.git_repos")
    parser.add_argument("path", help="Directory containing one or many git repos, OR a single repo path")
    parser.add_argument("--clone", default=None, help="Clone a remote URL into <path> first")
    parser.add_argument("--branch", default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = load_config()
    kb = KnowledgeBase(cfg)
    target = Path(args.path).expanduser()
    if args.clone:
        clone_repo(args.clone, target, branch=args.branch)
    repos = list_repos(target)
    if not repos:
        print(f"no git repos found at {target}", file=sys.stderr)
        return 1
    inc = cfg.ingestion.get("include_globs") or []
    exc = cfg.ingestion.get("exclude_globs") or []
    total = 0
    for repo in repos:
        n = ingest_repo(kb, repo, inc, exc)
        print(f"  {repo.name}: {n} pages indexed")
        total += n
    print(f"total: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
