#!/bin/bash
set -euo pipefail

export TMPDIR="/tmp/ray-worker"
mkdir -p "$TMPDIR"
ln -sf /root/.local/bin/uv /usr/local/bin/uv 2>/dev/null || true

cd /root/rllib
/usr/local/bin/uv run --no-sync ray stop --force >/dev/null 2>&1 || true
exec /usr/local/bin/uv run --no-sync ray start --address=100.64.0.13:6379 --block
