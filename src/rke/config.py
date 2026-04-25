"""RKE configuration loader.

Layered config:
  1. defaults defined in this module
  2. overrides from config/rke.yaml (if present)
  3. overrides from environment variables (RKE_* prefix, dot-paths use __)

Example: RKE_QDRANT__HOST=remote.example.com overrides qdrant.host.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


DEFAULTS: dict[str, Any] = {
    "wiki": {"path": "data/wiki"},
    "qdrant": {
        "host": "localhost",
        "port": 6333,
        "grpc_port": 6334,
        "collection": "rke_main",
        "api_key": None,
    },
    "falkordb": {
        "host": "localhost",
        "port": 6379,
        "graph": "rke",
        "password": None,
    },
    "embedding": {
        "model": "BAAI/bge-m3",
        "model_path": "models/BAAI/bge-m3",
        "device": "cpu",
        "dimensions": 1024,
        "batch_size": 32,
    },
    "rlm": {
        "max_depth": 5,
        "max_iterations": 20,
        "max_cost_usd": 1.00,
        "timeout_seconds": 300,
        "provider": None,
        "model": None,
    },
    "ingestion": {
        "parallel_workers": 4,
        "chunk_size": 1024,
        "chunk_overlap": 128,
        "include_globs": ["**/*.md", "**/*.txt", "**/*.py"],
        "exclude_globs": ["**/.git/**", "**/__pycache__/**", "**/node_modules/**"],
    },
    "logging": {"level": "INFO"},
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    out = dict(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def _coerce(value: str) -> Any:
    """Best-effort string → typed value coercion for env overrides."""
    low = value.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("null", "none", ""):
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _env_overrides(prefix: str = "RKE_") -> dict:
    """Build a nested dict of overrides from RKE_* env vars.

    RKE_QDRANT__HOST=foo  → {"qdrant": {"host": "foo"}}
    RKE_QDRANT_HOST=foo   → also accepted (single underscore = path separator
                            when the parent key is a known top-level section)
    """
    overrides: dict = {}
    known_sections = set(DEFAULTS.keys())
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        body = env_key[len(prefix):]
        # Prefer "__" separator if present.
        if "__" in body:
            parts = [p.lower() for p in body.split("__")]
        else:
            parts = body.lower().split("_")
            # Re-fuse trailing parts so RKE_QDRANT_GRPC_PORT → ["qdrant", "grpc_port"]
            if len(parts) > 1 and parts[0] in known_sections:
                head = parts[0]
                tail = "_".join(parts[1:])
                parts = [head, tail]
        cursor = overrides
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = _coerce(env_val)
    return overrides


@dataclass
class Config:
    raw: dict[str, Any] = field(default_factory=dict)

    def get(self, *path: str, default: Any = None) -> Any:
        cursor: Any = self.raw
        for p in path:
            if not isinstance(cursor, dict) or p not in cursor:
                return default
            cursor = cursor[p]
        return cursor

    @property
    def wiki_path(self) -> Path:
        return Path(self.get("wiki", "path", default="data/wiki"))

    @property
    def qdrant(self) -> dict:
        return self.get("qdrant", default={})

    @property
    def falkordb(self) -> dict:
        return self.get("falkordb", default={})

    @property
    def embedding(self) -> dict:
        return self.get("embedding", default={})

    @property
    def rlm(self) -> dict:
        return self.get("rlm", default={})

    @property
    def ingestion(self) -> dict:
        return self.get("ingestion", default={})

    @property
    def log_level(self) -> str:
        return str(self.get("logging", "level", default="INFO"))


def load_config(path: str | Path | None = None) -> Config:
    """Load layered config: defaults < yaml file < env vars."""
    yaml_path = Path(path) if path else Path("config/rke.yaml")
    merged = dict(DEFAULTS)
    if yaml_path.exists():
        with yaml_path.open("r", encoding="utf-8") as fh:
            user = yaml.safe_load(fh) or {}
        merged = _deep_merge(merged, user)
    merged = _deep_merge(merged, _env_overrides())
    return Config(raw=merged)


def load_sources(path: str | Path | None = None) -> dict:
    """Load config/sources.yaml. Returns {} if missing."""
    yaml_path = Path(path) if path else Path("config/sources.yaml")
    if not yaml_path.exists():
        return {}
    with yaml_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}
