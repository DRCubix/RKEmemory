"""Tests for the layered config loader."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rke.config import DEFAULTS, _coerce, _deep_merge, _env_overrides, load_config


def test_defaults_have_all_top_keys():
    for key in ("wiki", "qdrant", "falkordb", "embedding", "rlm", "ingestion", "logging"):
        assert key in DEFAULTS


def test_deep_merge_recursive():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    override = {"a": {"y": 99, "z": 100}, "c": 4}
    out = _deep_merge(base, override)
    assert out == {"a": {"x": 1, "y": 99, "z": 100}, "b": 3, "c": 4}
    # base must not be mutated
    assert base == {"a": {"x": 1, "y": 2}, "b": 3}


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("true", True), ("False", False), ("yes", True), ("off", False),
        ("null", None), ("", None),
        ("42", 42), ("3.14", 3.14), ("hello", "hello"),
    ],
)
def test_coerce(raw, expected):
    assert _coerce(raw) == expected


def test_env_overrides_with_double_underscore(monkeypatch):
    monkeypatch.setenv("RKE_QDRANT__HOST", "remote.example.com")
    monkeypatch.setenv("RKE_QDRANT__PORT", "9999")
    out = _env_overrides()
    assert out["qdrant"]["host"] == "remote.example.com"
    assert out["qdrant"]["port"] == 9999


def test_env_overrides_with_single_underscore(monkeypatch):
    monkeypatch.setenv("RKE_FALKORDB_PORT", "6380")
    out = _env_overrides()
    assert out["falkordb"]["port"] == 6380


def test_load_config_uses_defaults_when_no_yaml(tmp_path: Path, monkeypatch):
    # Ensure no spurious env affects this test
    for k in list(os.environ):
        if k.startswith("RKE_"):
            monkeypatch.delenv(k, raising=False)
    cfg = load_config(tmp_path / "missing.yaml")
    assert cfg.qdrant["host"] == "localhost"
    assert cfg.embedding["dimensions"] == 1024


def test_load_config_yaml_overrides(tmp_path: Path, monkeypatch):
    for k in list(os.environ):
        if k.startswith("RKE_"):
            monkeypatch.delenv(k, raising=False)
    yml = tmp_path / "rke.yaml"
    yml.write_text("qdrant:\n  host: db.local\n  port: 7000\n")
    cfg = load_config(yml)
    assert cfg.qdrant["host"] == "db.local"
    assert cfg.qdrant["port"] == 7000
    # untouched defaults survive
    assert cfg.qdrant["grpc_port"] == 6334
