"""Simulation environment, mock perception, and teleoperation package."""

import sys

# Prevent foreign Python version site-packages (e.g. Nix ROS Python 3.14 leaking into Python 3.12)
_curr_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
sys.path = [p for p in sys.path if "/lib/python" not in p or f"/lib/{_curr_ver}" in p]

from .car_gazebo_env import CarGazeboEnv
from .mock_perception import MockPerceptionEngine, SimulatedEntity
from .spatial_map import EgoKinematics, PathDeviation, SpatialMapBuilder
from .teleop import CarTeleopApp

__all__ = [
    "CarGazeboEnv",
    "CarTeleopApp",
    "EgoKinematics",
    "MockPerceptionEngine",
    "PathDeviation",
    "SimulatedEntity",
    "SpatialMapBuilder",
]
