#!/bin/bash
set -euo pipefail

export TMPDIR="/tmp/ray-worker"
export RAY_AUTH_MODE="disabled"
export RAY_ENABLE_UV_RUNTIME_ENV="0"
export VIRTUAL_ENV="/root/rllib/.venv"
export PATH="/root/rllib/.venv/bin:/root/.local/bin:/usr/local/bin:$PATH"
export PYTHONPATH="/root/rllib:/root/rllib/src/sim/gzdrl/_lib"
export LD_LIBRARY_PATH="/root/rllib/src/sim/gzdrl/_lib:${LD_LIBRARY_PATH:-}"
export GZ_SIM_RESOURCE_PATH="/root/rllib/src/v2/model"
export GZ_SIM_SYSTEM_PLUGIN_PATH="/root/rllib/src/sim/gzdrl/_lib"
mkdir -p "$TMPDIR"

HEAD_ADDR="${1:-100.64.0.21:6379}"
cd /root/rllib 2>/dev/null || true
/root/rllib/.venv/bin/ray stop --force >/dev/null 2>&1 || true
exec /root/rllib/.venv/bin/ray start --address="$HEAD_ADDR" --num-cpus=1 --block
