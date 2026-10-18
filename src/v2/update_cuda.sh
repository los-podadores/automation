#!/bin/bash
set -euo pipefail

cp /mnt/c/Windows/Temp/pyproject.toml /root/rllib/pyproject.toml
cd /root/rllib
/usr/local/bin/uv lock --upgrade-package torch
/usr/local/bin/uv sync
/usr/local/bin/uv run python -c "import torch; print('CUDA AVAILABLE:', torch.cuda.is_available(), 'DEVICE:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
