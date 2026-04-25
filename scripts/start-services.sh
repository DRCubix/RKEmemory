#!/usr/bin/env bash
# Start Qdrant + FalkorDB. Prefers docker compose; falls back to checking ports.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if command -v docker >/dev/null 2>&1; then
    if docker compose version >/dev/null 2>&1; then
        echo "[rke-services] using: docker compose"
        docker compose up -d
        echo
        echo "[rke-services] services started:"
        docker compose ps
        exit 0
    fi
    if command -v docker-compose >/dev/null 2>&1; then
        echo "[rke-services] using: docker-compose (legacy)"
        docker-compose up -d
        docker-compose ps
        exit 0
    fi
fi

echo "[rke-services] docker not available — checking if services are reachable on default ports"
QHOST=${RKE_QDRANT_HOST:-localhost}
QPORT=${RKE_QDRANT_PORT:-6333}
FHOST=${RKE_FALKORDB_HOST:-localhost}
FPORT=${RKE_FALKORDB_PORT:-6379}

check() {
    local name=$1 host=$2 port=$3
    if (echo > "/dev/tcp/$host/$port") >/dev/null 2>&1; then
        echo "  [ok]   $name reachable at $host:$port"
    else
        echo "  [miss] $name NOT reachable at $host:$port — start it manually"
    fi
}

check Qdrant   "$QHOST" "$QPORT"
check FalkorDB "$FHOST" "$FPORT"

echo
echo "[rke-services] manual install hints:"
echo "  Qdrant:   https://qdrant.tech/documentation/quickstart/"
echo "  FalkorDB: https://docs.falkordb.com/"
