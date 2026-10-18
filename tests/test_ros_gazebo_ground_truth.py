"""Unit tests for Gazebo and ROS 2 as ground truth for CarGazeboEnv and SpatialMapBuilder."""

from __future__ import annotations

import math
import sys
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from src.sim.spatial_map import EgoKinematics, LaserScanData, SpatialMapBuilder
from src.translate.fusion_types import CLASS_FREE, CLASS_STATIC_HAZARD, GoalState


class TestRosGazeboGroundTruth(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Mock ROS 2 modules if not available in current environment
        for mod in (
            "rclpy",
            "rclpy.executors",
            "geometry_msgs",
            "geometry_msgs.msg",
            "nav_msgs",
            "nav_msgs.msg",
            "sensor_msgs",
            "sensor_msgs.msg",
            "std_msgs",
            "std_msgs.msg",
            "ros_gz_interfaces",
            "ros_gz_interfaces.msg",
        ):
            if mod not in sys.modules:
                sys.modules[mod] = MagicMock()

    def setUp(self) -> None:
        self.map_builder = SpatialMapBuilder(num_rays=72, max_range=15.0)

    def test_pure_lidar_observation_without_mock_perception(self) -> None:
        """Verify that in gazebo mode, observations are generated without mock perception."""
        kinematics = EgoKinematics(
            vx=1.5,
            vy=0.05,
            steering_angle=0.12,
            yaw_rate=0.08,
            ax=0.4,
            ay=0.12,
        )
        goal_state = GoalState(
            dx_ego=8.0,
            dy_ego=1.0,
            distance=8.06,
            heading_error=0.12,
            cross_track_error=0.02,
            path_heading_error=0.01,
            phase=0,
        )
        # Front pod detects obstacle at 3.0m
        ranges = np.full(180, 15.0, dtype=np.float32)
        ranges[90] = 3.0  # Center sample hit
        pod_scans = {
            "front": LaserScanData(
                ranges=ranges,
                angle_min=-1.047198,
                angle_max=1.047198,
                angle_increment=2.094396 / 180,
            )
        }

        # Build observation without passing any obstacles list
        obs = self.map_builder.build_observation(
            kinematics=kinematics,
            goal_state=goal_state,
            pod_scans=pod_scans,
        )

        self.assertEqual(obs.shape, (156,))
        # Kinematics check
        np.testing.assert_allclose(obs[0:6], kinematics.to_array())
        # Goal check
        np.testing.assert_allclose(obs[6:12], goal_state.to_array())

        # Distances and classes
        dist_slice = obs[12:84]
        class_slice = obs[84:156]

        # The boresight ray should have distance < 1.0 and class == CLASS_STATIC_HAZARD
        hit_bins = np.where(class_slice == CLASS_STATIC_HAZARD)[0]
        self.assertGreater(len(hit_bins), 0)
        for b in hit_bins:
            self.assertLess(dist_slice[b], 1.0)

        # Free space rays should have distance == 1.0 and class == CLASS_FREE
        free_bins = np.where(class_slice == CLASS_FREE)[0]
        self.assertGreater(len(free_bins), 0)
        for b in free_bins:
            self.assertEqual(dist_slice[b], 1.0)

    def test_gazebo_collision_is_sole_ground_truth(self) -> None:
        """Verify that in gazebo mode, environment collision relies solely on Gazebo contacts."""
        from src.sim.car_gazebo_env import CarGazeboEnv

        with patch("src.sim.ros_bridge.Ros2GazeboBridge") as mock_bridge_cls:
            mock_bridge = MagicMock()
            mock_bridge_cls.return_value = mock_bridge
            mock_bridge.get_collision.return_value = True
            mock_bridge.has_world_odom = True
            mock_bridge.get_state.return_value = (
                {
                    "x": 0.0,
                    "y": 0.0,
                    "yaw": 0.0,
                    "vx": 0.0,
                    "vy": 0.0,
                    "yaw_rate": 0.0,
                    "steering_angle": 0.0,
                    "ax": 0.0,
                },
                {},
            )
            env = CarGazeboEnv(sim_mode="gazebo")
            with patch("time.sleep"):
                env.reset()
                obs, reward, terminated, truncated, info = env.step(np.array([0.0, 0.5], dtype=np.float32))

            self.assertTrue(info["collision"])
            self.assertTrue(terminated)
            self.assertLess(reward, -90.0)

            # When bridge reports no collision, collision is False even if car is near a nominal wall
            mock_bridge.get_collision.return_value = False
            with patch("time.sleep"):
                obs, reward, terminated, truncated, info = env.step(np.array([0.0, 0.5], dtype=np.float32))
            self.assertFalse(info["collision"])
            self.assertFalse(terminated)

    def test_ros_bridge_joint_state_and_imu_processing(self) -> None:
        """Verify Ros2GazeboBridge processes joint states, IMU data, and odometry."""
        from src.sim.ros_bridge import Ros2GazeboBridge

        with patch("threading.Thread"):
            bridge = Ros2GazeboBridge(node_name="test_bridge")

            # Simulate JointState message from Gazebo
            mock_joint_msg = MagicMock()
            mock_joint_msg.name = ["front_left_steering", "front_right_steering", "rear_left_spin"]
            mock_joint_msg.position = [0.25, 0.23, 10.5]
            bridge._on_joint_state(mock_joint_msg)
            self.assertAlmostEqual(bridge.latest_steering_angle, 0.24, places=3)

            # Simulate IMU message from Gazebo
            mock_imu_msg = MagicMock()
            mock_imu_msg.linear_acceleration.x = 1.2
            mock_imu_msg.linear_acceleration.y = -0.3
            mock_imu_msg.angular_velocity.z = 0.15
            bridge._on_imu(mock_imu_msg)
            self.assertIsNotNone(bridge.latest_imu)
            self.assertAlmostEqual(bridge.latest_imu["ax"], 1.2)
            self.assertAlmostEqual(bridge.latest_imu["ay"], -0.3)

            # Simulate Odometry message
            mock_odom_msg = MagicMock()
            mock_odom_msg.pose.pose.position.x = 5.0
            mock_odom_msg.pose.pose.position.y = 2.0
            mock_odom_msg.pose.pose.orientation.w = 1.0
            mock_odom_msg.pose.pose.orientation.x = 0.0
            mock_odom_msg.pose.pose.orientation.y = 0.0
            mock_odom_msg.pose.pose.orientation.z = 0.0
            mock_odom_msg.twist.twist.linear.x = 1.8
            mock_odom_msg.twist.twist.linear.y = 0.05
            mock_odom_msg.twist.twist.angular.z = 0.12
            bridge._on_odom(mock_odom_msg)

            odom, scans = bridge.get_state()
            self.assertIsNotNone(odom)
            self.assertAlmostEqual(odom["steering_angle"], 0.24, places=3)
            self.assertAlmostEqual(odom["ax"], 1.2)
            self.assertAlmostEqual(odom["ay"], -0.3)
            self.assertAlmostEqual(odom["vx"], 1.8)
            self.assertAlmostEqual(odom["vy"], 0.05)

    def test_track_is_fully_enclosed_at_both_ends(self) -> None:
        """Verify that generated tracks have enclosing perimeter walls at both Point A and Point B."""
        from src.sim.track_generator import ProceduralTrackGenerator

        gen = ProceduralTrackGenerator()
        track = gen.generate_demo_90deg(seed=42)

        # There must be walls enclosing behind Point A (start) and Point B (goal)
        ax, ay, _ = track.start_pose
        bx, by, _ = track.goal_pose

        # Find walls close to Point A
        walls_near_a = [
            seg
            for seg in track.wall_segments
            if math.hypot((seg[0][0] + seg[1][0]) / 2 - ax, (seg[0][1] + seg[1][1]) / 2 - ay) < 10.0
        ]
        self.assertGreaterEqual(len(walls_near_a), 3)

        # Find walls close to Point B
        walls_near_b = [
            seg
            for seg in track.wall_segments
            if math.hypot((seg[0][0] + seg[1][0]) / 2 - bx, (seg[0][1] + seg[1][1]) / 2 - by) < 10.0
        ]
        self.assertGreaterEqual(len(walls_near_b), 3)

    def test_off_track_termination(self) -> None:
        """Verify that driving off the road triggers off_track termination with negative reward."""
        from src.sim.car_gazebo_env import CarGazeboEnv

        env = CarGazeboEnv(sim_mode="mock")
        obs, info = env.reset(seed=42)

        # Teleport car far off-track
        env.car_x += 25.0
        env.car_y += 25.0

        obs, reward, terminated, truncated, info = env.step(np.array([0.0, 0.5], dtype=np.float32))
        self.assertTrue(terminated)
        self.assertTrue(info.get("off_track", False))
        self.assertLessEqual(reward, -50.0)
        env.close()


if __name__ == "__main__":
    unittest.main()
