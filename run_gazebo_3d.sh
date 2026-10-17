#!/usr/bin/env bash
# Launch 3D Gazebo Harmonic simulation and real-time Ackermann car teleoperation
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

export GZ_SIM_RESOURCE_PATH="$DIR/src/v2/model:${GZ_SIM_RESOURCE_PATH:-}"

echo "=========================================================="
echo "    Los Podadores — 3D Gazebo Simulation & Teleop"
echo "=========================================================="

echo "[1/2] Generating guaranteed-passable procedural track..."
uv run python -c "
from src.sim.track_generator import ProceduralTrackGenerator
from pathlib import Path
gen = ProceduralTrackGenerator(default_road_width=7.0)
track = gen.generate_track(seed=None, length=95.0, road_width=7.0, num_obstacles=4)
sdf_p = Path('src/v2/model/procedural_track.sdf')
track.export_gazebo_sdf(sdf_p)
print(f'  Track exported to {sdf_p} ({len(track.wall_segments)} walls, {len(track.obstacles)} obstacles)')
"

echo "[2/2] Launching Gazebo Harmonic 3D window and Teleoperation HUD..."
nix develop .#ros-desktop -c bash -c "
  export GZ_IP=127.0.0.1
  export GZ_SIM_RESOURCE_PATH=\"$DIR/src/v2/model:\$GZ_SIM_RESOURCE_PATH\"
  echo '  Starting Gazebo 3D simulation server on 127.0.0.1...'
  gz sim -s -r \"$DIR/src/v2/model/procedural_track.sdf\" &
  GZ_PID=\$!

  # Wait for Gazebo server to establish topics
  sleep 2.0

  echo '  Starting Teleoperation HUD with keyboard control (streaming to /cmd_vel)...'
  env -u PYTHONPATH uv run src/sim/teleop.py --backend gazebo

  echo '  Shutting down Gazebo server...'
  kill \$GZ_PID 2>/dev/null || true
  wait \$GZ_PID 2>/dev/null || true
"

echo "Simulation session finished cleanly."
