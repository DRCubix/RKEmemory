"""FalkorDB knowledge graph wrapper.

Stores entities and relationships as a labeled property graph; supports
Cypher queries and a small GraphRAG helper.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .config import Config, load_config

log = logging.getLogger(__name__)


@dataclass
class Entity:
    label: str
    name: str
    properties: dict[str, Any] = None  # type: ignore[assignment]

    def to_props(self) -> dict[str, Any]:
        out = dict(self.properties or {})
        out["name"] = self.name
        return out


@dataclass
class Relation:
    src_label: str
    src_name: str
    rel_type: str
    dst_label: str
    dst_name: str
    properties: dict[str, Any] = None  # type: ignore[assignment]


class GraphStore:
    """FalkorDB client wrapper. Connection is lazy."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or load_config()
        self._db = None
        self._graph = None
        self._graph_name = self.config.falkordb.get("graph", "rke")

    # ── Connection ───────────────────────────────────────────────
    def connect(self):
        if self._graph is not None:
            return self._graph
        from falkordb import FalkorDB

        fcfg = self.config.falkordb
        self._db = FalkorDB(
            host=fcfg.get("host", "localhost"),
            port=int(fcfg.get("port", 6379)),
            password=fcfg.get("password") or None,
        )
        self._graph = self._db.select_graph(self._graph_name)
        return self._graph

    @property
    def graph(self):
        if self._graph is None:
            self.connect()
        return self._graph

    def ping(self) -> bool:
        try:
            self.connect().query("RETURN 1")
            return True
        except Exception as exc:
            log.warning("falkordb ping failed: %s", exc)
            return False

    # ── Mutations ────────────────────────────────────────────────
    def add_entity(self, entity: Entity) -> None:
        props = entity.to_props()
        cypher = (
            f"MERGE (n:{_safe_label(entity.label)} {{name: $name}}) "
            f"SET n += $props"
        )
        self.graph.query(cypher, {"name": entity.name, "props": props})

    def add_relation(self, relation: Relation) -> None:
        cypher = (
            f"MATCH (a:{_safe_label(relation.src_label)} {{name: $sname}}) "
            f"MATCH (b:{_safe_label(relation.dst_label)} {{name: $dname}}) "
            f"MERGE (a)-[r:{_safe_label(relation.rel_type)}]->(b) "
            f"SET r += $props"
        )
        self.graph.query(cypher, {
            "sname": relation.src_name,
            "dname": relation.dst_name,
            "props": dict(relation.properties or {}),
        })

    # ── Queries ──────────────────────────────────────────────────
    def query(self, cypher: str, params: dict | None = None) -> list[list[Any]]:
        result = self.graph.query(cypher, params or {})
        return list(result.result_set)

    def stats(self) -> dict[str, int]:
        try:
            n = self.query("MATCH (n) RETURN count(n)")
            r = self.query("MATCH ()-[r]->() RETURN count(r)")
            return {
                "nodes": int(n[0][0]) if n else 0,
                "relationships": int(r[0][0]) if r else 0,
            }
        except Exception as exc:
            return {"error": str(exc)}

    def graphrag_query(self, question: str, max_hops: int = 2) -> dict[str, Any]:
        """Naive GraphRAG: find entities whose name appears in the question,
        then expand `max_hops` neighbors and return their relations.

        This is intentionally simple and dependency-free. A real GraphRAG
        layer (LLM-extracted entities + reranked subgraph) is in the
        `rke.rlm` package and is invoked from there.
        """
        terms = [t.strip(",.?!:;\"'()[]") for t in question.split() if len(t) > 3]
        if not terms:
            return {"question": question, "subgraph": []}
        cypher = (
            "MATCH (n) WHERE toLower(n.name) IN $terms "
            f"OPTIONAL MATCH path = (n)-[*1..{max_hops}]-(m) "
            "RETURN n.name AS root, labels(n) AS labels, "
            "       collect(DISTINCT m.name)[0..20] AS neighbors"
        )
        rows = self.query(cypher, {"terms": [t.lower() for t in terms]})
        subgraph = [
            {"root": r[0], "labels": r[1], "neighbors": r[2]}
            for r in rows
        ]
        return {"question": question, "subgraph": subgraph}


_LABEL_OK = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")


def _safe_label(label: str) -> str:
    """Sanitize a label/relation type — Cypher does not parameterize these."""
    cleaned = "".join(c for c in label if c in _LABEL_OK)
    if not cleaned or cleaned[0].isdigit():
        cleaned = "X" + cleaned
    return cleaned
