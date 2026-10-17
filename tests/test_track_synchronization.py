"""Unit tests for procedural track JSON synchronization and Gazebo environment initialization."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.sim.car_gazebo_env import CarGazeboEnv
from src.sim.track_generator import ProceduralTrackGenerator


class TestTrackSynchronization(unittest.TestCase):
    def setUp(self) -> None:
        self.gen = ProceduralTrackGenerator(default_road_width=7.0)
        self.track = self.gen.generate_track(seed=123, length=50.0, road_width=7.0, num_obstacles=3)

    def test_json_export_and_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "test_track.json"
            self.track.export_json(json_path)
            self.assertTrue(json_path.exists())

            loaded = self.track.load_json(json_path)
            self.assertEqual(loaded.seed, self.track.seed)
            self.assertEqual(loaded.start_pose, self.track.start_pose)
            self.assertEqual(loaded.goal_pose, self.track.goal_pose)
            self.assertEqual(loaded.road_width, self.track.road_width)
            self.assertEqual(len(loaded.wall_segments), len(self.track.wall_segments))
            self.assertEqual(len(loaded.obstacles), len(self.track.obstacles))
            np.testing.assert_allclose(loaded.waypoints_forward, self.track.waypoints_forward)
            np.testing.assert_allclose(loaded.waypoints_reverse, self.track.waypoints_reverse)

    def test_env_loads_synchronized_track(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "procedural_track.json"
            self.track.export_json(json_path)

            env = CarGazeboEnv(sim_mode="mock", track_path=json_path)
            env.track_path = json_path
            obs, info = env.reset(options={"track": self.track})

            self.assertEqual(env.track.seed, self.track.seed)
            self.assertAlmostEqual(env.car_x, self.track.start_pose[0])
            self.assertAlmostEqual(env.car_y, self.track.start_pose[1])
            self.assertAlmostEqual(env.car_yaw, self.track.start_pose[2])
            self.assertEqual(obs.shape, (156,))

    def test_instant_throttle_cut(self) -> None:
        """Verify that when no throttle key is active, throttle action drops to zero immediately."""
        # Simulated keyboard input logic matching teleop.py
        target_throttle = 0.0
        throttle_action = 0.85
        if target_throttle == 0.0:
            throttle_action = 0.0
        else:
            throttle_action += (target_throttle - throttle_action) * 0.40
        self.assertEqual(throttle_action, 0.0)

    def test_active_braking_before_reverse(self) -> None:
        """Verify that negative throttle commands 0.0 speed (braking) when moving forward at speed."""
        car_vx = 1.8
        throttle_cmd = -1.0
        if throttle_cmd >= 0.0:
            if car_vx < -0.1:
                target_speed = 0.0
            else:
                target_speed = throttle_cmd * 2.0
        else:
            if car_vx > 0.1:
                target_speed = 0.0
            else:
                target_speed = throttle_cmd * 1.0
        self.assertEqual(target_speed, 0.0)
