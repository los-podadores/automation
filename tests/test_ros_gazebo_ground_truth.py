"""Unit tests for Gazebo and ROS 2 as ground truth for CarGazeboEnv and SpatialMapBuilder."""

from __future__ import annotations

import math
import sys
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from src.sim.spatial_map import EgoKinematics, SpatialMapBuilder
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

    def test_pure_lidar_observation_from_point_clouds(self) -> None:
        """Verify that observations come solely from segmented 3D point clouds."""
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
        # Segmented obstacle point straight ahead at 3.0m in ego frame.
        obstacle_points = np.array([[3.0, 0.0, 0.2]], dtype=np.float32)

        # Build observation from geometry only (no semantic input).
        obs = self.map_builder.build_observation(
            kinematics=kinematics,
            goal_state=goal_state,
            obstacle_points=obstacle_points,
        )

        self.assertEqual(obs.shape, (156,))
        # Kinematics check
        np.testing.assert_allclose(obs[0:6], kinematics.to_array())
        # Goal check
        np.testing.assert_allclose(obs[6:12], goal_state.to_array())

        # Distances and classes
        dist_slice = obs[12:84]
        class_slice = obs[84:156]

        # Only FREE and STATIC_HAZARD classes may appear.
        self.assertTrue(set(np.unique(class_slice)) <= {CLASS_FREE, CLASS_STATIC_HAZARD})

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

    def test_ditch_points_produce_hazard_bins(self) -> None:
        """Verify that negative-hazard (ditch) points set STATIC_HAZARD bins."""
        kinematics = EgoKinematics(
            vx=1.0,
            vy=0.0,
            steering_angle=0.0,
            yaw_rate=0.0,
            ax=0.0,
            ay=0.0,
        )
        goal_state = GoalState(
            dx_ego=5.0,
            dy_ego=0.0,
            distance=5.0,
            heading_error=0.0,
            cross_track_error=0.0,
            path_heading_error=0.0,
            phase=0,
        )
        negative_hazards = np.array([[2.0, 1.0, -0.4]], dtype=np.float32)
        obs = self.map_builder.build_observation(
            kinematics=kinematics,
            goal_state=goal_state,
            negative_hazards=negative_hazards,
        )
        class_slice = obs[84:156]
        self.assertIn(CLASS_STATIC_HAZARD, set(class_slice.tolist()))
        hit_bins = np.where(class_slice == CLASS_STATIC_HAZARD)[0]
        self.assertGreater(len(hit_bins), 0)
        self.assertTrue(set(np.unique(class_slice)) <= {CLASS_FREE, CLASS_STATIC_HAZARD})

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
                {"front": np.zeros((4, 3), dtype=np.float32), "rear": np.zeros((3, 3), dtype=np.float32)},
            )
            env = CarGazeboEnv()
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
        track = gen.generate_estadio_ho_chi_min(seed=42)

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

        with patch("src.sim.ros_bridge.Ros2GazeboBridge") as mock_bridge_cls:
            mock_bridge = MagicMock()
            mock_bridge_cls.return_value = mock_bridge
            mock_bridge.has_world_odom = True
            mock_bridge.get_collision.return_value = False
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
                {"front": np.zeros((4, 3), dtype=np.float32), "rear": np.zeros((3, 3), dtype=np.float32)},
            )
            env = CarGazeboEnv()
            with patch("time.sleep"):
                env.reset(seed=42)

                # Drive Gazebo odometry far off-track so the contact gate sees off-track.
                mock_bridge.get_state.return_value = (
                    {
                        "x": -25.0,
                        "y": 25.0,
                        "yaw": 0.0,
                        "vx": 0.0,
                        "vy": 0.0,
                        "yaw_rate": 0.0,
                        "steering_angle": 0.0,
                        "ax": 0.0,
                    },
                    {"front": np.zeros((4, 3), dtype=np.float32), "rear": np.zeros((3, 3), dtype=np.float32)},
                )
                obs, reward, terminated, truncated, info = env.step(np.array([0.0, 0.5], dtype=np.float32))
                self.assertTrue(terminated)
                self.assertTrue(info.get("off_track", False))
                self.assertLessEqual(reward, -50.0)
                env.close()

    def test_reverse_motion_capability(self) -> None:
        """Verify that negative throttle publishes a reverse target speed to Gazebo."""
        from src.sim.car_gazebo_env import CarGazeboEnv

        with patch("src.sim.ros_bridge.Ros2GazeboBridge") as mock_bridge_cls:
            mock_bridge = MagicMock()
            mock_bridge_cls.return_value = mock_bridge
            mock_bridge.has_world_odom = True
            mock_bridge.get_collision.return_value = False
            moving = {
                "x": 0.0,
                "y": 0.0,
                "yaw": 0.0,
                "vx": -1.0,
                "vy": 0.0,
                "yaw_rate": 0.0,
                "steering_angle": 0.0,
                "ax": 0.0,
            }
            mock_bridge.get_state.return_value = (moving, {"front": np.zeros((4, 3), dtype=np.float32), "rear": np.zeros((3, 3), dtype=np.float32)})
            env = CarGazeboEnv()
            with patch("time.sleep"):
                env.reset(seed=123)
                for _ in range(25):
                    env.step(np.array([0.0, -1.0], dtype=np.float32))

            from src.translate.config import MIN_SPEED_MPS

            self.assertLess(env.car_vx, -0.9)
            self.assertGreaterEqual(env.car_vx, MIN_SPEED_MPS - 1e-4)
            env.close()

    def test_random_policy_net_forward_progression(self) -> None:
        """Verify that Gazebo odometry updates drive forward progression."""
        from src.sim.car_gazebo_env import CarGazeboEnv

        with patch("src.sim.ros_bridge.Ros2GazeboBridge") as mock_bridge_cls:
            mock_bridge = MagicMock()
            mock_bridge_cls.return_value = mock_bridge
            mock_bridge.has_world_odom = True
            mock_bridge.get_collision.return_value = False
            moving = {
                "x": 0.0,
                "y": 0.0,
                "yaw": 0.0,
                "vx": 1.0,
                "vy": 0.0,
                "yaw_rate": 0.0,
                "steering_angle": 0.0,
                "ax": 0.0,
            }
            mock_bridge.get_state.return_value = (moving, {"front": np.zeros((4, 3), dtype=np.float32), "rear": np.zeros((3, 3), dtype=np.float32)})
            env = CarGazeboEnv()
            with patch("time.sleep"):
                env.reset(seed=42)
                rng = np.random.default_rng(42)
                initial_x = env.car_x
                for _ in range(100):
                    action = rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)
                    obs, reward, terminated, truncated, info = env.step(action)
                    moving["x"] += 0.05
                    mock_bridge.get_state.return_value = (moving, {"front": np.zeros((4, 3), dtype=np.float32), "rear": np.zeros((3, 3), dtype=np.float32)})
                    if terminated:
                        break

                # Net displacement must be strictly positive (forward)
                self.assertGreater(env.car_x, initial_x)
                env.close()

    def test_ros_bridge_point_cloud_processing(self) -> None:
        """Verify Ros2GazeboBridge processes 3D PointCloud2 messages and transforms them to ego frame."""
        from src.sim.ros_bridge import Ros2GazeboBridge

        with patch("threading.Thread"):
            bridge = Ros2GazeboBridge(node_name="test_cloud_bridge")

            # Create mock PointCloud2 message
            mock_cloud = MagicMock()
            f_x = MagicMock(offset=0)
            f_x.name = "x"
            f_y = MagicMock(offset=4)
            f_y.name = "y"
            f_z = MagicMock(offset=8)
            f_z.name = "z"
            mock_cloud.fields = [f_x, f_y, f_z]
            mock_cloud.point_step = 16

            # Point at (2.0, 0.5, 0.1) in sensor frame
            raw_floats = np.array([2.0, 0.5, 0.1, 0.0], dtype=np.float32)
            mock_cloud.data = raw_floats.tobytes()

            bridge._on_point_cloud("front", mock_cloud)
            clouds = bridge.get_point_clouds()

            self.assertIn("front", clouds)
            front_pts = clouds["front"]
            from src.translate.config import FRONT_LIDAR_X
            # x_ego should be > FRONT_LIDAR_X because sensor is at front bumper pointing forward
            self.assertGreater(front_pts[0, 0], FRONT_LIDAR_X)


if __name__ == "__main__":
    unittest.main()
