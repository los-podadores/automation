#!/bin/bash
set -euo pipefail

export TMPDIR="/tmp/ray-head"
export RAY_AUTH_MODE="disabled"
mkdir -p "$TMPDIR"
ln -sf /root/.local/bin/uv /usr/local/bin/uv 2>/dev/null || true

cd /root/rllib 2>/dev/null || true
/usr/local/bin/uv run --no-sync ray stop --force >/dev/null 2>&1 || true
exec /usr/local/bin/uv run --no-sync ray start --head --node-ip-address=100.64.0.21 --port=6379 --disable-usage-stats --block
