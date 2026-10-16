#!/bin/bash
export TMPDIR="/tmp/ray-worker"
cd /root/rllib 2>/dev/null || true
/usr/local/bin/uv run --no-sync ray stop --force >/dev/null 2>&1 || true
