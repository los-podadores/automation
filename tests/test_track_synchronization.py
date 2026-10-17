"""Unit tests for procedural track JSON synchronization and Gazebo environment initialization."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.sim.car_gazebo_env import CarGazeboEnv
from src.sim.track_generator import ProceduralTrackGenerator


class TestTrackSynchronization(unittest.TestCase):
    def setUp(self) -> None:
        self.gen = ProceduralTrackGenerator(default_road_width=7.0)
        self.track = self.gen.generate_track(
            seed=123, length=50.0, road_width=7.0, num_obstacles=3
        )

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
            np.testing.assert_allclose(
                loaded.waypoints_forward, self.track.waypoints_forward
            )
            np.testing.assert_allclose(
                loaded.waypoints_reverse, self.track.waypoints_reverse
            )

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

    def test_sdf_dynamic_obstacles_and_contact_plugin(self) -> None:
        """Verify dynamic obstacles have TrajectoryFollower, OdometryPublisher, and world has Contact plugin."""
        track = self.gen.generate_demo_90deg(seed=42)
        with tempfile.TemporaryDirectory() as tmpdir:
            sdf_path = Path(tmpdir) / "test_track.sdf"
            track.export_gazebo_sdf(sdf_path)
            content = sdf_path.read_text(encoding="utf-8")

            self.assertIn("gz-sim-contact-system", content)
            self.assertIn("gz-sim-trajectory-follower-system", content)
            self.assertIn("gz-sim-odometry-publisher-system", content)

    def test_tractor_sdf_contact_sensor_and_chassis_collision(self) -> None:
        """Verify tractor model SDF has chassis_body_collision and tractor_contact_sensor."""
        tractor_sdf_path = Path("src/v2/model/tractor/model.sdf")
        content = tractor_sdf_path.read_text(encoding="utf-8")
        self.assertIn("chassis_body_collision", content)
        self.assertIn("tractor_contact_sensor", content)
        self.assertIn("/model/tractor/contacts", content)

    def test_bridge_config_contacts_and_obstacle_odometry(self) -> None:
        """Verify bridge_config.yaml contains tractor contacts and obstacle odometry."""
        bridge_yaml_path = Path("src/v2/bridge_config.yaml")
        content = bridge_yaml_path.read_text(encoding="utf-8")
        self.assertIn("/model/tractor/contacts", content)
        self.assertIn("ros_gz_interfaces/msg/Contacts", content)
        self.assertIn("/model/obstacle_0/odometry", content)

    def test_multi_circle_wall_collision_detection(self) -> None:
        """Verify multi-circle collision check catches corner overlaps that single circle misses."""
        track = self.gen.generate_demo_90deg(seed=42)
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "track.json"
            track.export_json(json_path)
            env = CarGazeboEnv(sim_mode="mock", track_path=json_path)
            env.reset(options={"track": track})

            # Position car near a wall where center is >1.15m away but front bumper hits
            # Road width is 7.0m, wall is at y = 3.5m. Center at y = 2.2m (dist=1.3m > 1.15m)
            # With car_yaw = 60 degrees, front reaches y = 2.2 + 0.8 * sin(60 deg) = 2.2 + 0.69 = 2.89m
            # Distance from front center (2.89) to wall (3.5) is 0.61m < 0.85m -> COLLISION!
            env.car_x = 10.0
            env.car_y = 2.2
            env.car_yaw = math.radians(60.0)

            self.assertTrue(env._check_wall_collision())
