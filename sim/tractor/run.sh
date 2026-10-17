#!/usr/bin/env bash
set -euo pipefail
SIM_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec /usr/bin/python3 "$SIM_DIR/launch.py" "$@"
