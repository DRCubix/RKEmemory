"""Parallel knowledge ingestion from local files/dirs.

Reads `config/sources.yaml`, walks each `paths` source, and turns matching
files into wiki pages indexed in the vector store via KnowledgeBase.
"""

from __future__ import annotations

import argparse
import fnmatch
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from ..config import Config, load_config, load_sources
from ..wiki.knowledge_base import KnowledgeBase

log = logging.getLogger(__name__)

_TEXT_SUFFIXES = {
    ".md", ".markdown", ".txt", ".rst",
    ".py", ".js", ".jsx", ".ts", ".tsx",
    ".go", ".rs", ".java", ".kt", ".rb", ".php",
    ".html", ".css", ".yaml", ".yml", ".toml", ".json", ".csv",
}


def _matches(path: Path, includes: list[str], excludes: list[str]) -> bool:
    rel = str(path)
    for pat in excludes:
        if fnmatch.fnmatch(rel, pat):
            return False
    if not includes:
        return True
    return any(fnmatch.fnmatch(rel, pat) for pat in includes)


def _read_text(path: Path, max_bytes: int = 1_000_000) -> str | None:
    try:
        if path.stat().st_size > max_bytes:
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        log.debug("skip %s: %s", path, exc)
        return None


def iter_files(roots: list[str], includes: list[str], excludes: list[str]) -> list[Path]:
    out: list[Path] = []
    for root_str in roots:
        root = Path(root_str).expanduser()
        if root.is_file():
            out.append(root)
            continue
        if not root.exists():
            log.warning("ingestion: path not found: %s", root)
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in _TEXT_SUFFIXES:
                continue
            if _matches(p, includes, excludes):
                out.append(p)
    return out


def ingest_source(
    kb: KnowledgeBase,
    name: str,
    spec: dict[str, Any],
    cfg: Config,
) -> dict[str, int]:
    paths = spec.get("paths") or []
    if not paths:
        log.warning("source %s has no paths", name)
        return {"name": name, "files": 0, "pages": 0}
    includes = cfg.ingestion.get("include_globs") or []
    excludes = cfg.ingestion.get("exclude_globs") or []
    files = iter_files(paths, includes, excludes)
    if not files:
        return {"name": name, "files": 0, "pages": 0}
    category = spec.get("category", f"sources/{name}")
    tags = list(spec.get("tags") or [])
    pages = 0
    for f in files:
        text = _read_text(f)
        if not text:
            continue
        title = f.stem.replace("_", " ").replace("-", " ").title()
        body = f"_From `{f}`_\n\n{text}"
        try:
            kb.add_page(title=title, body=body, category=category, tags=tags + [f.suffix.lstrip(".")])
            pages += 1
        except Exception as exc:
            log.warning("failed to ingest %s: %s", f, exc)
    return {"name": name, "files": len(files), "pages": pages}


def ingest_all(parallel: bool = True, sources_path: str | None = None) -> list[dict[str, int]]:
    cfg = load_config()
    sources = load_sources(sources_path).get("sources", {})
    if not sources:
        log.warning("no sources configured (config/sources.yaml). Nothing to ingest.")
        return []
    kb = KnowledgeBase(cfg)
    path_sources = {
        n: s for n, s in sources.items()
        if (s.get("type") or "paths") == "paths"
    }
    workers = max(1, int(cfg.ingestion.get("parallel_workers", 4)))
    results: list[dict[str, int]] = []
    if parallel and len(path_sources) > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(ingest_source, kb, name, spec, cfg): name
                for name, spec in path_sources.items()
            }
            for fut in as_completed(futures):
                results.append(fut.result())
    else:
        for name, spec in path_sources.items():
            results.append(ingest_source(kb, name, spec, cfg))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser("rke.ingestion.knowledge")
    parser.add_argument("mode", nargs="?", default="all", choices=["all"])
    parser.add_argument("--parallel", action="store_true", default=False)
    parser.add_argument("--sources", default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    results = ingest_all(parallel=args.parallel, sources_path=args.sources)
    for r in results:
        print(f"  {r['name']}: {r['pages']}/{r['files']} pages indexed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
