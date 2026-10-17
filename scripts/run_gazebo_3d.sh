#!/usr/bin/env bash
# Launch 3D Gazebo Harmonic simulation, ROS 2 Jazzy parameter bridge, and Ackermann car teleoperation
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

export PATH="/run/current-system/sw/bin:$DIR/scripts:$PATH"
export DBX_CONTAINER_MANAGER="docker"
export GZ_IP=127.0.0.1
export GZ_SIM_RESOURCE_PATH="$DIR/src/v2/model:${GZ_SIM_RESOURCE_PATH:-}"

echo "=========================================================="
echo "    Los Podadores — 3D Gazebo Simulation & Teleop"
PRESET="${1:-demo_90deg}"
echo "    Preset: $PRESET"

# Terminate any lingering simulation, bridge, or rogue topic publishers from previous runs
kill -9 $(pgrep -f "gz sim" 2>/dev/null) $(pgrep -f "ros_gz_bridge" 2>/dev/null) $(pgrep -f "ros2 topic pub" 2>/dev/null) $(pgrep -f "teleop.py" 2>/dev/null) 2>/dev/null || true

echo "[1/3] Generating procedural track layout ($PRESET)..."
uv run python -c "
import sys
from src.sim.track_generator import ProceduralTrackGenerator
from pathlib import Path
gen = ProceduralTrackGenerator(default_road_width=7.0)
track = gen.generate_track(seed=None, preset='$PRESET')
sdf_p = Path('src/v2/model/procedural_track.sdf')
json_p = Path('src/v2/model/procedural_track.json')
track.export_gazebo_sdf(sdf_p)
track.export_json(json_p)
print(f'  Track ({track.preset_name}) exported: {len(track.wall_segments)} walls, {len(track.obstacles)} obstacles, nominal length: {track.nominal_length:.1f}m, max steps: {track.max_episode_steps}')
"

echo "[2/3] Launching ROS 2 Jazzy Parameter Bridge in background..."
./scripts/ros2 run ros_gz_bridge parameter_bridge --ros-args -p config_file:=src/v2/bridge_config.yaml &
BRIDGE_PID=$!

echo "[3/3] Launching Gazebo Harmonic 3D window and Teleoperation HUD..."
"$DIR/scripts/gz" sim -r "$DIR/src/v2/model/procedural_track.sdf" &
GZ_PID=$!

cleanup() {
    echo "  Shutting down Gazebo server, ROS bridge, and any topic publishers..."
    kill $BRIDGE_PID 2>/dev/null || true
    kill $GZ_PID 2>/dev/null || true
    kill -9 $(pgrep -f "ros2 topic pub" 2>/dev/null) 2>/dev/null || true
    wait $GZ_PID 2>/dev/null || true
    wait $BRIDGE_PID 2>/dev/null || true
    echo "Simulation session finished cleanly."
}
trap cleanup EXIT INT TERM

# Wait for Gazebo server to establish topics
sleep 3.0

echo "  Starting Teleoperation HUD with live ROS 2 bridge to Gazebo Harmonic..."
./scripts/run_teleop --backend gazebo --preset "$PRESET"
