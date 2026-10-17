"""Simulation environment, mock perception, and teleoperation package."""


import numpy  # noqa: F401

from .car_gazebo_env import CarGazeboEnv  # noqa: E402
from .mock_perception import MockPerceptionEngine, SimulatedEntity  # noqa: E402
from .spatial_map import EgoKinematics, SpatialMapBuilder  # noqa: E402
from .track_generator import ProceduralTrack, ProceduralTrackGenerator  # noqa: E402


def __getattr__(name: str) -> object:
    if name == "CarTeleopApp":
        from .teleop import CarTeleopApp

        return CarTeleopApp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CarGazeboEnv",
    "CarTeleopApp",
    "EgoKinematics",
    "MockPerceptionEngine",
    "ProceduralTrack",
    "ProceduralTrackGenerator",
    "SimulatedEntity",
    "SpatialMapBuilder",
]

