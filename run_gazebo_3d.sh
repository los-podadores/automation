#!/usr/bin/env bash
# Runner delegating to scripts/run_gazebo_3d.sh
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$DIR/scripts/run_gazebo_3d.sh" "$@"
