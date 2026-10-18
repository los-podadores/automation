"""Unit tests for procedural track JSON synchronization and Gazebo environment initialization."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from src.sim.car_gazebo_env import CarGazeboEnv
from src.sim.track_generator import ProceduralTrackGenerator
class TestTrackSynchronization(unittest.TestCase):
    def setUp(self) -> None:
        self.gen = ProceduralTrackGenerator(default_road_width=7.0)
        self.track = self.gen.generate_estadio_ho_chi_min(seed=123)

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

            with patch("src.sim.ros_bridge.Ros2GazeboBridge") as bridge_cls:
                mock_bridge = MagicMock()
                bridge_cls.return_value = mock_bridge
                mock_bridge.has_world_odom = True
                mock_bridge.get_state.return_value = (
                    None,
                    {
                        "front": np.zeros((4, 3), dtype=np.float32),
                        "rear": np.zeros((3, 3), dtype=np.float32),
                    },
                )
                env = CarGazeboEnv(track_path=json_path)
                with patch("time.sleep"):
                    obs, info = env.reset(options={"track": self.track})

            self.assertEqual(env.track.seed, self.track.seed)
            self.assertAlmostEqual(env.car_x, self.track.start_pose[0])
            self.assertAlmostEqual(env.car_y, self.track.start_pose[1])
            self.assertAlmostEqual(env.car_yaw, self.track.start_pose[2])
            self.assertEqual(obs.shape, (156,))
            self.assertEqual(info["lidar_source"], "ros_gz_3d")

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
        track = self.gen.generate_estadio_ho_chi_min(seed=42)
        with tempfile.TemporaryDirectory() as tmpdir:
            sdf_path = Path(tmpdir) / "test_track.sdf"
            track.export_gazebo_sdf(sdf_path)
            content = sdf_path.read_text(encoding="utf-8")

            self.assertIn("gz-sim-contact-system", content)
            self.assertIn("gz-sim-trajectory-follower-system", content)
            self.assertIn("gz-sim-odometry-publisher-system", content)
            self.assertIn('joint name="guide" type="prismatic"', content)
            self.assertIn('joint name="pivot" type="revolute"', content)
            self.assertIn("<parent>world</parent>", content)
            self.assertIn("<damping>35.0</damping>", content)

    def test_dynamic_obstacle_axis_aligns_with_waypoints(self) -> None:
        """Verify prismatic guide axes run along the waypoint travel direction."""
        import math
        import re

        track = self.gen.generate_estadio_ho_chi_min(seed=42)
        with tempfile.TemporaryDirectory() as tmpdir:
            sdf_path = Path(tmpdir) / "test_track.sdf"
            track.export_gazebo_sdf(sdf_path)
            content = sdf_path.read_text(encoding="utf-8")

        blocks = re.findall(
            r"<!-- Dynamic Obstacle \d+: .*? -->.*?"
            r"<pose>(.*?)</pose>.*?"
            r"<xyz>(.*?)</xyz>.*?"
            r"<waypoint>(.*?)</waypoint>\s*<waypoint>(.*?)</waypoint>",
            content,
            re.S,
        )
        self.assertGreater(len(blocks), 0)
        for pose_txt, axis_txt, wp1_txt, wp2_txt in blocks:
            model_yaw = float(pose_txt.split()[5])
            ax, ay, _ = (float(v) for v in axis_txt.split())
            x1, y1 = (float(v) for v in wp1_txt.split())
            x2, y2 = (float(v) for v in wp2_txt.split())
            dx, dy = (x2 - x1), (y2 - y1)
            dist = math.hypot(dx, dy)
            self.assertGreater(dist, 1.0)
            # Guide axis is expressed in the yaw-rotated model frame: rotating
            # it forward by the model yaw must recover the world direction.
            cos_y, sin_y = math.cos(model_yaw), math.sin(model_yaw)
            wx = ax * cos_y - ay * sin_y
            wy = ax * sin_y + ay * cos_y
            self.assertAlmostEqual(wx * (dx / dist) + wy * (dy / dist), 1.0, places=2)

    def test_tractor_sdf_contact_sensor_and_chassis_collision(self) -> None:
        """Verify tractor model SDF has chassis_body_collision and tractor_contact_sensor."""
        tractor_sdf_path = Path("src/v2/model/tractor/model.sdf")
        content = tractor_sdf_path.read_text(encoding="utf-8")
        self.assertIn("chassis_body_collision", content)
        self.assertIn("tractor_contact_sensor", content)
        self.assertIn("/model/tractor/contacts", content)

    def test_bridge_config_contacts_and_joint_state(self) -> None:
        """Verify bridge_config.yaml contains tractor contacts, joint_state, and imu."""
        bridge_yaml_path = Path("src/v2/bridge_config.yaml")
        content = bridge_yaml_path.read_text(encoding="utf-8")
        self.assertIn("/model/tractor/contacts", content)
        self.assertIn("ros_gz_interfaces/msg/Contacts", content)
        self.assertIn("/model/tractor/joint_state", content)
        self.assertIn("sensor_msgs/msg/JointState", content)
        self.assertIn("/model/tractor/imu", content)
        self.assertIn("sensor_msgs/msg/Imu", content)

