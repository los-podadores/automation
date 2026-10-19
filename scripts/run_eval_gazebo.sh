#!/usr/bin/env bash
# Launch 3D Gazebo Harmonic simulation, ROS 2 Jazzy parameter bridge, and Autonomous Car Evaluation
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

export PATH="/run/current-system/sw/bin:$DIR/scripts:$PATH"
export DBX_CONTAINER_MANAGER="docker"
CONTAINER_NAME="gazebo-harmonic"
export GZ_IP=127.0.0.1
export GZ_PARTITION="los_podadores"
export GZ_SIM_RESOURCE_PATH="$DIR/src/v2/model:${GZ_SIM_RESOURCE_PATH:-}"
export GZ_SIM_SYSTEM_PLUGIN_PATH="$DIR/src/sim/gzdrl/_lib:${GZ_SIM_SYSTEM_PLUGIN_PATH:-}"

CHECKPOINT="${1:-models/car_rllib/final_checkpoint}"
shift 1 || true

echo "=========================================================="
echo "    Los Podadores — Gazebo Harmonic 3D Autonomous Eval"
echo "    Checkpoint: $CHECKPOINT"
echo "=========================================================="

# Terminate any lingering simulation or bridge processes from previous runs
pkill -9 -f "[g]z sim" 2>/dev/null || true
pkill -9 -f "[r]os_gz_bridge" 2>/dev/null || true
pkill -9 -f "[p]arameter_bridge" 2>/dev/null || true
pkill -9 -f "[e]val_car.py" 2>/dev/null || true

# Ensure procedural track SDF is exported
if [ ! -f "src/v2/model/procedural_track.sdf" ]; then
    echo "[1/3] Generating procedural track layout..."
    uv run python -c "
from src.sim.track_generator import ProceduralTrackGenerator
from pathlib import Path
gen = ProceduralTrackGenerator(default_road_width=7.0)
track = gen.generate_track(seed=None, preset='estadio_ho_chi_min')
track.export_gazebo_sdf(Path('src/v2/model/procedural_track.sdf'))
track.export_json(Path('src/v2/model/procedural_track.json'))
"
fi

echo "[1/3] Launching ROS 2 Jazzy Parameter Bridge in background..."
./scripts/ros2 run ros_gz_bridge parameter_bridge --ros-args -p config_file:=src/v2/bridge_config.yaml &
BRIDGE_PID=$!

echo "[2/3] Launching Gazebo Harmonic 3D window..."
"$DIR/scripts/gz" sim -r "$DIR/src/v2/model/procedural_track.sdf" &
GZ_PID=$!

cleanup() {
    echo "  Shutting down Gazebo server and ROS bridge..."
    kill $BRIDGE_PID 2>/dev/null || true
    kill $GZ_PID 2>/dev/null || true
    pkill -9 -f "[g]z sim" 2>/dev/null || true
    pkill -9 -f "[p]arameter_bridge" 2>/dev/null || true
    wait $GZ_PID 2>/dev/null || true
    wait $BRIDGE_PID 2>/dev/null || true
    echo "Evaluation session finished cleanly."
}
trap cleanup EXIT INT TERM

# Wait for Gazebo server to establish topics
sleep 3.0

echo "[3/3] Starting Autonomous Policy Evaluation in Gazebo 3D..."
distrobox enter "$CONTAINER_NAME" -- bash -c '
    for var in $(compgen -v | grep -E "^(NIX_|CMAKE_|LD_|PYTHON)"); do
        unset "$var"
    done
    export PATH=/opt/ros/jazzy/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
    export GZ_IP=127.0.0.1
    export GZ_PARTITION="los_podadores"
    source /opt/ros/jazzy/setup.bash
    export PYTHONPATH="'"$DIR"':${PYTHONPATH:-}"
    export DISPLAY="'"${DISPLAY:-:0}"'"
    export WAYLAND_DISPLAY="'"${WAYLAND_DISPLAY:-}"'"
    export XDG_RUNTIME_DIR="'"${XDG_RUNTIME_DIR:-}"'"
    "'"$DIR"'"/.venv/bin/python src/v2/eval_car.py --checkpoint "'"$CHECKPOINT"'" --sim-mode gazebo "$@"
' -- "$@"
