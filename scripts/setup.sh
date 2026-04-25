#!/usr/bin/env bash
# RKE setup — creates a venv, installs the package, and copies config templates.
# Idempotent: safe to re-run.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[rke-setup] working dir: $ROOT"

# 1. Python version check
if ! command -v python3 >/dev/null 2>&1; then
    echo "[rke-setup] ERROR: python3 not found in PATH" >&2
    exit 1
fi
PYV=$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
echo "[rke-setup] python version: $PYV"

# 2. Virtual environment
if [ ! -d ".venv" ]; then
    echo "[rke-setup] creating .venv"
    python3 -m venv .venv
else
    echo "[rke-setup] .venv exists, skipping"
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# 3. Upgrade pip
python -m pip install --upgrade pip

# 4. Install package in editable mode with dev deps
echo "[rke-setup] installing rke (editable, with dev extras)"
pip install -e ".[dev]"

# 5. Project directories
mkdir -p data/wiki
mkdir -p models
mkdir -p logs
mkdir -p storage/qdrant
mkdir -p storage/falkordb
mkdir -p snapshots

# 6. Config templates → real configs (only if missing)
if [ ! -f "config/rke.yaml" ]; then
    cp config/rke.yaml.example config/rke.yaml
    echo "[rke-setup] created config/rke.yaml from template"
fi
if [ ! -f "config/sources.yaml" ]; then
    cp config/sources.yaml.example config/sources.yaml
    echo "[rke-setup] created config/sources.yaml from template"
fi
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "[rke-setup] created .env from template"
fi

echo
echo "[rke-setup] done."
echo "Next steps:"
echo "  1. Start services:    bash scripts/start-services.sh"
echo "  2. Activate venv:     source .venv/bin/activate"
echo "  3. Check status:      rke status"
