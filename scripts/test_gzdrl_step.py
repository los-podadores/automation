#!/usr/bin/env python3
"""Integration smoke test for GzDRL simulation engine with Los Podadores tractor."""

from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np

# Add gzdrl to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
GZDRL_LIB = BASE_DIR / "src" / "sim" / "gzdrl" / "_lib"
sys.path.insert(0, str(GZDRL_LIB))

import gzdrl

# Set Gazebo plugin and resource paths
os.environ["GZ_SIM_SYSTEM_PLUGIN_PATH"] = f"{GZDRL_LIB}:{os.environ.get('GZ_SIM_SYSTEM_PLUGIN_PATH', '')}"
os.environ["GZ_SIM_RESOURCE_PATH"] = f"{BASE_DIR / 'src' / 'v2' / 'model'}:{os.environ.get('GZ_SIM_RESOURCE_PATH', '')}"

print("GzDRL Version:", gzdrl.__version__)
print("Using GZ_SIM_SYSTEM_PLUGIN_PATH:", os.environ["GZ_SIM_SYSTEM_PLUGIN_PATH"])
print("Using GZ_SIM_RESOURCE_PATH:", os.environ["GZ_SIM_RESOURCE_PATH"])

world_path = BASE_DIR / "src" / "v2" / "model" / "procedural_track.sdf"
if not world_path.exists():
    raise FileNotFoundError(f"World SDF not found: {world_path}")

print("Initializing DRLServer with world:", world_path)

server = gzdrl.DRLServer(
    "podadores_test",
    str(world_path),
    ["tractor"],
    True,  # enable_sensors
)

# Step 20 iterations (idle initialization so sensors configure)
server.run_N(20)

# Query sensors after initialization steps
lidars = server.lidar_sensor_names()
print("Discovered LiDAR sensors:", lidars)

# Check initial pose
initial_states = server.state_info("tractor")
print("Tractor links in state_info:", list(initial_states.keys()))
chassis_state = initial_states["chassis"]
print(f"Initial chassis pos: x={chassis_state[0]:.3f}, y={chassis_state[1]:.3f}, z={chassis_state[2]:.3f}")

# Command forward speed 1.5 m/s, steering 0.2 rad
print("Applying Ackermann command: speed=1.5 m/s, steer=0.2 rad")
cmd = np.array([1.5, 0.2], dtype=np.float64)
server.set_ackermann_velocity_cmd("tractor", "chassis", cmd)

# Step 100 iterations (at 0.005s = 0.5s sim time)
server.run_N(100)

new_states = server.state_info("tractor")
new_chassis = new_states["chassis"]
print(f"Moved chassis pos: x={new_chassis[0]:.3f}, y={new_chassis[1]:.3f}, z={new_chassis[2]:.3f}")
dx = new_chassis[0] - chassis_state[0]
dy = new_chassis[1] - chassis_state[1]
dist_moved = np.hypot(dx, dy)
print(f"Distance traveled in 0.5s: {dist_moved:.3f} m")

# Test reading front LiDAR scan
front_lidar_data = server.get_sensor_gpu_lidar("tractor::chassis::lidar_front")
print(f"Front LiDAR buffer shape: {front_lidar_data.shape}, dtype={front_lidar_data.dtype}")
if front_lidar_data.size > 0:
    valid_pts = np.isfinite(front_lidar_data) & (front_lidar_data > 0)
    print(f"Valid LiDAR range points: {np.count_nonzero(valid_pts)} / {front_lidar_data.size}")
    if np.any(valid_pts):
        print(f"Min range: {front_lidar_data[valid_pts].min():.2f}m, Max range: {front_lidar_data[valid_pts].max():.2f}m")

# Request contacts
server.request_contact_data("tractor")
server.run_N(5)
contacts = server.get_contacts("tractor")
print(f"Contact batches detected: {len(contacts)}")

print("Test completed successfully!")
