"""RKE command-line interface."""

from __future__ import annotations

import logging
import sys

import typer
from rich.console import Console
from rich.table import Table

from .agent_integration import format_context_for_agent, gather_context
from .config import load_config
from .graph_store import Entity, GraphStore, Relation
from .ingestion.knowledge import ingest_all
from .rlm.router import RLMRouter
from .vector_store import VectorStore
from .wiki.knowledge_base import KnowledgeBase
from .wiki.manager import WikiManager

app = typer.Typer(
    name="rke",
    help="RKE — Recursive Knowledge Engine",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


def _setup_logging() -> None:
    cfg = load_config()
    logging.basicConfig(
        level=cfg.log_level,
        format="%(levelname)s %(name)s: %(message)s",
    )


# ── status ──────────────────────────────────────────────────────
@app.command()
def status() -> None:
    """Print connection + index status across all subsystems."""
    _setup_logging()
    cfg = load_config()
    table = Table(title="RKE status", show_lines=False)
    table.add_column("Subsystem")
    table.add_column("Detail")
    table.add_column("OK?")

    # Wiki
    try:
        wm = WikiManager(cfg)
        n_pages = len(wm.list_pages())
        table.add_row("Wiki", f"{wm.root} ({n_pages} pages)", "✓")
    except Exception as exc:
        table.add_row("Wiki", str(exc), "✗")

    # Qdrant
    try:
        vs = VectorStore(cfg)
        info = vs.collection_info()
        if "error" in info:
            table.add_row("Qdrant", info["error"], "✗")
        else:
            table.add_row(
                "Qdrant",
                f"{cfg.qdrant.get('host')}:{cfg.qdrant.get('port')} "
                f"• {info['name']} • {info.get('points_count', 0)} points",
                "✓",
            )
    except Exception as exc:
        table.add_row("Qdrant", str(exc), "✗")

    # FalkorDB
    try:
        gs = GraphStore(cfg)
        if gs.ping():
            stats = gs.stats()
            table.add_row(
                "FalkorDB",
                f"{cfg.falkordb.get('host')}:{cfg.falkordb.get('port')} • "
                f"{stats.get('nodes', 0)} nodes / {stats.get('relationships', 0)} rels",
                "✓",
            )
        else:
            table.add_row("FalkorDB", "unreachable", "✗")
    except Exception as exc:
        table.add_row("FalkorDB", str(exc), "✗")

    table.add_row(
        "Embedding model",
        f"{cfg.embedding.get('model')} ({cfg.embedding.get('dimensions')}d) on {cfg.embedding.get('device')}",
        "—",
    )
    console.print(table)


# ── search ──────────────────────────────────────────────────────
@app.command()
def search(query: str, limit: int = typer.Option(5, "-n", "--limit")) -> None:
    """Pure semantic vector search."""
    _setup_logging()
    vs = VectorStore()
    hits = vs.hybrid_search(query, limit=limit)
    if not hits:
        console.print("[dim]No hits.[/dim]")
        return
    for i, h in enumerate(hits, 1):
        console.print(f"[bold]{i}.[/bold] [cyan]{h.score:.3f}[/cyan] {h.metadata.get('title', '(chunk)')}")
        console.print(f"   {h.text[:200]}")
        console.print()


# ── query ──────────────────────────────────────────────────────
@app.command()
def query(question: str, agent: str | None = typer.Option(None, "--agent")) -> None:
    """Multi-layer query (wiki + vectors + graph) → formatted context block."""
    _setup_logging()
    ctx = gather_context(question)
    out = format_context_for_agent(ctx, agent=agent or "generic")
    console.print(out)


# ── graph ──────────────────────────────────────────────────────
@app.command()
def graph(cypher: str) -> None:
    """Run a Cypher query against the FalkorDB knowledge graph."""
    _setup_logging()
    gs = GraphStore()
    try:
        rows = gs.query(cypher)
    except Exception as exc:
        console.print(f"[red]Query failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    for row in rows:
        console.print(row)


@app.command("graph-seed")
def graph_seed() -> None:
    """Seed the knowledge graph with a tiny example so newcomers see something."""
    _setup_logging()
    gs = GraphStore()
    gs.add_entity(Entity("Project", "RKE", {"description": "Recursive Knowledge Engine"}))
    gs.add_entity(Entity("API", "Qdrant", {"kind": "vector_db"}))
    gs.add_entity(Entity("API", "FalkorDB", {"kind": "graph_db"}))
    gs.add_relation(Relation("Project", "RKE", "USES", "API", "Qdrant"))
    gs.add_relation(Relation("Project", "RKE", "USES", "API", "FalkorDB"))
    console.print("[green]Seeded:[/green] Project(RKE) -[USES]-> API(Qdrant), API(FalkorDB)")


# ── ingest ──────────────────────────────────────────────────────
@app.command()
def ingest(
    path: str | None = typer.Argument(None, help="A single file or dir to ingest"),
    parallel: bool = typer.Option(True, "--parallel/--serial"),
) -> None:
    """Ingest a single file/dir, or run all sources from config/sources.yaml."""
    _setup_logging()
    if path:
        from pathlib import Path
        p = Path(path).expanduser()
        if not p.exists():
            console.print(f"[red]Not found:[/red] {p}")
            raise typer.Exit(code=1)
        kb = KnowledgeBase()
        if p.is_file():
            text = p.read_text(encoding="utf-8", errors="replace")
            page = kb.add_page(
                title=p.stem.replace("_", " ").replace("-", " ").title(),
                body=f"_File: `{p}`_\n\n{text}",
                category="ingested",
                tags=[p.suffix.lstrip(".")],
            )
            console.print(f"[green]Indexed:[/green] {page.slug}")
        else:
            from .ingestion.knowledge import iter_files
            cfg = load_config()
            inc = cfg.ingestion.get("include_globs") or []
            exc = cfg.ingestion.get("exclude_globs") or []
            files = iter_files([str(p)], inc, exc)
            n = 0
            for f in files:
                try:
                    text = f.read_text(encoding="utf-8", errors="replace")
                    kb.add_page(
                        title=f"{p.name}: {f.relative_to(p)}",
                        body=f"_File: `{f}`_\n\n{text}",
                        category=f"ingested/{p.name}",
                        tags=["ingested", f.suffix.lstrip(".")],
                    )
                    n += 1
                except Exception as exc2:
                    console.print(f"[yellow]skip[/yellow] {f}: {exc2}")
            console.print(f"[green]Indexed {n}[/green] of {len(files)} files from {p}")
        return
    results = ingest_all(parallel=parallel)
    if not results:
        console.print("[yellow]No sources defined in config/sources.yaml[/yellow]")
        return
    for r in results:
        console.print(f"  {r['name']}: {r['pages']}/{r['files']} pages indexed")


# ── wiki ────────────────────────────────────────────────────────
@app.command("wiki-page")
def wiki_page(
    title: str,
    body: str = typer.Argument(..., help="Markdown body (or '-' to read stdin)"),
    category: str = typer.Option("general", "--category"),
    tags: str = typer.Option("", "--tags", help="Comma-separated tags"),
) -> None:
    """Create or update a wiki page and index it into the vector store."""
    _setup_logging()
    if body == "-":
        body = sys.stdin.read()
    kb = KnowledgeBase()
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    page = kb.add_page(title, body, category=category, tags=tag_list)
    console.print(f"[green]Wrote:[/green] {page.path}")


@app.command("wiki-list")
def wiki_list(category: str | None = None) -> None:
    """List wiki pages."""
    _setup_logging()
    wm = WikiManager()
    pages = wm.list_pages(category=category)
    if not pages:
        console.print("[dim](no pages)[/dim]")
        return
    table = Table(show_header=True)
    table.add_column("Slug")
    table.add_column("Title")
    table.add_column("Category")
    table.add_column("Tags")
    for p in pages:
        table.add_row(p.slug, p.title, p.category, ", ".join(p.tags))
    console.print(table)


@app.command()
def lint() -> None:
    """Wiki health check — orphans, broken links, duplicates, empty pages."""
    _setup_logging()
    wm = WikiManager()
    findings = wm.lint()
    if not findings:
        console.print("[green]Wiki is clean.[/green]")
        return
    by_kind: dict[str, int] = {}
    for f in findings:
        console.print(f"[yellow]{f.kind}[/yellow]  {f.page}: {f.detail}")
        by_kind[f.kind] = by_kind.get(f.kind, 0) + 1
    console.print()
    for k, v in by_kind.items():
        console.print(f"  {k}: {v}")


# ── rlm ─────────────────────────────────────────────────────────
@app.command()
def rlm(query: str, agent: str | None = typer.Option(None, "--agent")) -> None:
    """Run an RLM completion. Falls back to deterministic mode without LLM creds."""
    _setup_logging()
    router = RLMRouter()
    result = router.complete(query=query, agent=agent)
    console.print(result.answer)
    console.print()
    console.print(
        f"[dim]iterations={result.iterations}  used_llm={result.used_llm}  "
        f"cost_usd={result.cost_usd:.4f}[/dim]"
    )


# ── reindex ─────────────────────────────────────────────────────
@app.command()
def reindex() -> None:
    """Rebuild the vector index from the current wiki contents."""
    _setup_logging()
    kb = KnowledgeBase()
    n = kb.reindex_all()
    console.print(f"[green]Reindexed {n} pages.[/green]")


# ── version ─────────────────────────────────────────────────────
@app.command()
def version() -> None:
    from . import __version__
    console.print(f"rke {__version__}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
