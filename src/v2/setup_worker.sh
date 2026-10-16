#!/bin/bash
# Setup script run INSIDE WSL Ubuntu as root on each worker.
# Installs uv + python 3.12, creates /root/rllib venv from pyproject, installs ray.
set -euo pipefail
export PATH="/root/.local/bin:$PATH"
export DEBIAN_FRONTEND=noninteractive

apt-get update -qq
apt-get install -y -qq curl ca-certificates > /dev/null

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="/root/.local/bin:$PATH"
fi

cd /root/rllib
uv sync --frozen
uv run --no-sync python -c "import ray, ray.rllib; print('ray', ray.__version__, 'rllib OK')"
echo "WORKER-SETUP-OK"
