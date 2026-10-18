#!/bin/bash
set -euo pipefail

export TMPDIR="/tmp/ray-worker"
export RAY_AUTH_MODE="disabled"
mkdir -p "$TMPDIR"
ln -sf /root/.local/bin/uv /usr/local/bin/uv 2>/dev/null || true

HEAD_ADDR="${1:-100.64.0.5:6379}"
/usr/local/bin/uv run --no-sync ray stop --force >/dev/null 2>&1 || true
exec /usr/local/bin/uv run --no-sync ray start --address="$HEAD_ADDR" --block
