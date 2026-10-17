"""Unit tests for SpatialMapBuilder multi-pod LaserScan fusion."""

from __future__ import annotations

import math
import unittest
import numpy as np

from src.sim.spatial_map import EgoKinematics, LaserScanData, SpatialMapBuilder
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
        # Create a scan from the front pod (at x=1.05m, y=0.0m)
        # Target at 3.0m in front along pod boresight (0 rad)
        # Ego coordinate of hit: x = 1.05 + 3.0 = 4.05m, y = 0.0m, azimuth = 0 rad
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
        expected_norm_dist = 4.05 / 15.0

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
        self.assertAlmostEqual(center_pt[0], 4.05, places=2)
        self.assertAlmostEqual(center_pt[1], 0.0, places=2)

    def test_semantic_obstacle_tagging_in_pod_fusion(self) -> None:
        from src.translate.fusion_types import CLASS_PEDESTRIAN, TrackedObstacle

        samples = 180
        ranges = np.full(samples, 15.0, dtype=np.float32)
        ranges[samples // 2] = 3.0  # Hit at ego x=4.05m, y=0.0m
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
            rel_x=4.05,
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


if __name__ == "__main__":
    unittest.main()

