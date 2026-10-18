"""Unit tests for SpatialMapBuilder multi-pod LaserScan fusion."""

from __future__ import annotations

import math
import unittest
import numpy as np

from src.sim.spatial_map import EgoKinematics, LaserScanData, SpatialMapBuilder
from src.translate.config import FRONT_LIDAR_X
from src.translate.fusion_types import GoalState


class TestSpatialMapLaserFusion(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = SpatialMapBuilder(num_rays=72, max_range=15.0)

    def test_fuse_pod_scans_empty(self) -> None:
        distances, classes = self.builder.fuse_pod_scans({})
        self.assertEqual(len(distances), 72)
        self.assertEqual(len(classes), 72)
        np.testing.assert_allclose(distances, 1.0)
        np.testing.assert_allclose(classes, 0.0)

    def test_front_pod_obstacle_detection(self) -> None:
        # Create a scan from the front pod (at x=FRONT_LIDAR_X, y=0.0m)
        # Target at 3.0m in front along pod boresight (0 rad)
        # Ego coordinate of hit: x = FRONT_LIDAR_X + 3.0, y = 0.0m, azimuth = 0 rad
        # Boresight angle is in ray bin near index 36 (since -pi is index 0, 0 rad is index 36)
        samples = 180
        ranges = np.full(samples, 15.0, dtype=np.float32)
        center_idx = samples // 2
        ranges[center_idx] = 3.0  # 3m return right ahead

        scan_front = LaserScanData(
            ranges=ranges,
            angle_min=-1.047198,
            angle_max=1.047198,
            angle_increment=2.094396 / samples,
        )

        distances, classes = self.builder.fuse_pod_scans({"front": scan_front})

        # Ray bin for azimuth 0 rad
        zero_bin = int(math.floor((0.0 + math.pi) / (2.0 * math.pi / 72))) % 72
        expected_norm_dist = (FRONT_LIDAR_X + 3.0) / 15.0

        self.assertAlmostEqual(distances[zero_bin], expected_norm_dist, places=2)
        self.assertGreater(classes[zero_bin], 0.0)

    def test_build_observation_with_pod_scans(self) -> None:
        kinematics = EgoKinematics(
            vx=1.2,
            vy=0.0,
            steering_angle=0.1,
            yaw_rate=0.05,
            ax=0.2,
            ay=0.06,
        )
        goal_state = GoalState(
            dx_ego=10.0,
            dy_ego=2.0,
            distance=10.2,
            heading_error=0.15,
            cross_track_error=0.05,
            path_heading_error=0.02,
            phase=0,
        )

        # 4 pods with max range
        pod_scans = {
            "front": LaserScanData(
                ranges=np.full(180, 15.0, dtype=np.float32),
                angle_min=-1.047198,
                angle_max=1.047198,
                angle_increment=2.094396 / 180,
            ),
            "rear": LaserScanData(
                ranges=np.full(180, 15.0, dtype=np.float32),
                angle_min=-1.047198,
                angle_max=1.047198,
                angle_increment=2.094396 / 180,
            ),
            "right": LaserScanData(
                ranges=np.full(150, 15.0, dtype=np.float32),
                angle_min=-0.872665,
                angle_max=0.872665,
                angle_increment=1.74533 / 150,
            ),
            "left": LaserScanData(
                ranges=np.full(150, 15.0, dtype=np.float32),
                angle_min=-0.872665,
                angle_max=0.872665,
                angle_increment=1.74533 / 150,
            ),
        }

        obs = self.builder.build_observation(
            kinematics=kinematics,
            goal_state=goal_state,
            pod_scans=pod_scans,
        )

        self.assertEqual(obs.shape, (156,))
        self.assertEqual(obs.dtype, np.float32)
        # Kinematics check
        np.testing.assert_allclose(obs[0:6], kinematics.to_array())
        # Goal state check
        np.testing.assert_allclose(obs[6:12], goal_state.to_array())

    def test_extract_point_cloud_ego(self) -> None:
        samples = 180
        ranges = np.full(samples, 15.0, dtype=np.float32)
        ranges[samples // 2] = 3.0  # 3m ahead of front pod at x=1.05m -> ego x=4.05m, y=0.0m
        pod_scans = {
            "front": LaserScanData(
                ranges=ranges,
                angle_min=-1.047198,
                angle_max=1.047198,
                angle_increment=2.094396 / samples,
            )
        }
        pts = self.builder.extract_point_cloud_ego(pod_scans)
        self.assertEqual(pts.shape, (samples, 2))
        # Center sample should be at x ~ 4.05, y ~ 0.0
        center_pt = pts[samples // 2]
        self.assertAlmostEqual(center_pt[0], FRONT_LIDAR_X + 3.0, places=2)
        self.assertAlmostEqual(center_pt[1], 0.0, places=2)

    def test_semantic_obstacle_tagging_in_pod_fusion(self) -> None:
        from src.translate.fusion_types import CLASS_PEDESTRIAN, TrackedObstacle

        samples = 180
        ranges = np.full(samples, 15.0, dtype=np.float32)
        ranges[samples // 2] = 3.0  # Hit at ego x = FRONT_LIDAR_X + 3.0m, y=0.0m
        pod_scans = {
            "front": LaserScanData(
                ranges=ranges,
                angle_min=-1.047198,
                angle_max=1.047198,
                angle_increment=2.094396 / samples,
            )
        }
        ped = TrackedObstacle(
            class_id=CLASS_PEDESTRIAN,
            class_name="person",
            rel_x=FRONT_LIDAR_X + 3.0,
            rel_y=0.0,
            rel_vx=0.0,
            rel_vy=0.0,
            radius=0.45,
            confidence=0.99,
        )
        distances, classes = self.builder.fuse_pod_scans(
            pod_scans, obstacles=[ped]
        )
        zero_bin = int(math.floor((0.0 + math.pi) / (2.0 * math.pi / 72))) % 72
        self.assertEqual(classes[zero_bin], float(CLASS_PEDESTRIAN))

    def test_extract_pod_beam_segments_origins_and_endpoints(self) -> None:
        from src.translate.fusion_types import CLASS_PEDESTRIAN, TrackedObstacle

        samples = 180
        ranges = np.full(samples, 15.0, dtype=np.float32)
        ranges[samples // 2] = 2.0  # Hit 2m ahead of front pod
        pod_scans = {
            "front": LaserScanData(
                ranges=ranges,
                angle_min=-1.047198,
                angle_max=1.047198,
                angle_increment=2.094396 / samples,
            )
        }
        ped = TrackedObstacle(
            class_id=CLASS_PEDESTRIAN,
            class_name="person",
            rel_x=FRONT_LIDAR_X + 2.0,
            rel_y=0.0,
            rel_vx=0.0,
            rel_vy=0.0,
            radius=0.45,
            confidence=0.99,
        )
        beam_data = self.builder.extract_pod_beam_segments(pod_scans, obstacles=[ped])
        self.assertEqual(len(beam_data), 1)
        pod = beam_data[0]
        self.assertEqual(pod["pod_name"], "front")
        self.assertEqual(pod["origin_ego"], (FRONT_LIDAR_X, 0.0))
        self.assertEqual(len(pod["endpoints_ego"]), samples)

        center_idx = samples // 2
        self.assertTrue(pod["is_hit"][center_idx])
        self.assertEqual(pod["classes"][center_idx], CLASS_PEDESTRIAN)
        self.assertAlmostEqual(pod["endpoints_ego"][center_idx][0], FRONT_LIDAR_X + 2.0, places=2)
        self.assertAlmostEqual(pod["endpoints_ego"][center_idx][1], 0.0, places=2)

    def test_generate_mock_pod_scans(self) -> None:
        # Place a wall 5m ahead of the car (x=5.0) spanning y in [-10, 10]
        wall_ego = [((5.0, -10.0), (5.0, 10.0))]
        mock_scans = self.builder.generate_mock_pod_scans(wall_segments_ego=wall_ego)
        self.assertIn("front", mock_scans)
        self.assertIn("rear", mock_scans)
        self.assertIn("right", mock_scans)
        self.assertIn("left", mock_scans)

        # Front pod is at x=FRONT_LIDAR_X facing forward -> distance to x=5.0 wall is 5.0 - FRONT_LIDAR_X
        front_scan = mock_scans["front"]
        center_sample = front_scan.ranges[len(front_scan.ranges) // 2]
        self.assertAlmostEqual(center_sample, 5.0 - FRONT_LIDAR_X, places=2)

    def test_fuse_point_cloud_obstacles(self) -> None:
        """Verify that 3D obstacle points project accurately into 72-bin polar range grid."""
        # Single obstacle at x=3.0m, y=0.0m (azimuth 0, bin 36)
        obs_pts = np.array([[3.0, 0.0, 0.2]], dtype=np.float32)
        distances, classes = self.builder.fuse_point_cloud_obstacles(obs_pts)

        self.assertEqual(len(distances), 72)
        self.assertEqual(len(classes), 72)

        zero_bin = int(math.floor((0.0 + math.pi) / (2.0 * math.pi / 72))) % 72
        self.assertAlmostEqual(distances[zero_bin], 3.0 / 15.0, places=3)
        self.assertGreater(classes[zero_bin], 0.0)

    def test_build_observation_with_point_cloud_obstacles(self) -> None:
        """Verify build_observation produces valid 156-dim vector when given point cloud obstacles."""
        kinematics = EgoKinematics(vx=1.0, vy=0.0, steering_angle=0.0, yaw_rate=0.0, ax=0.0, ay=0.0)
        goal_state = GoalState(dx_ego=5.0, dy_ego=0.0, distance=5.0, heading_error=0.0, cross_track_error=0.0, path_heading_error=0.0, phase=0)

        obs_pts = np.array([[2.5, 0.0, 0.15]], dtype=np.float32)
        obs = self.builder.build_observation(
            kinematics=kinematics,
            goal_state=goal_state,
            obstacle_points=obs_pts,
        )

        self.assertEqual(obs.shape, (156,))
        # Check kinematics and goal in slices
        np.testing.assert_allclose(obs[0:6], kinematics.to_array())
        np.testing.assert_allclose(obs[6:12], goal_state.to_array())

        zero_bin = int(math.floor((0.0 + math.pi) / (2.0 * math.pi / 72))) % 72
        dist_slice = obs[12:84]
        self.assertAlmostEqual(dist_slice[zero_bin], 2.5 / 15.0, places=3)


if __name__ == "__main__":
    unittest.main()

