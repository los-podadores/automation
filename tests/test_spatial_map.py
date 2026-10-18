"""Unit tests for SpatialMapBuilder 3D point-cloud fusion."""

from __future__ import annotations

import math
import unittest
import numpy as np

from src.sim.spatial_map import EgoKinematics, SpatialMapBuilder
from src.translate.fusion_types import GoalState


class TestSpatialMapPointCloudFusion(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = SpatialMapBuilder(num_rays=72, max_range=15.0)

    def test_fuse_point_cloud_obstacles_empty(self) -> None:
        from src.translate.fusion_types import CLASS_FREE

        distances, classes = self.builder.fuse_point_cloud_obstacles(None)
        self.assertEqual(len(distances), 72)
        self.assertEqual(len(classes), 72)
        np.testing.assert_allclose(distances, 1.0)
        np.testing.assert_allclose(classes, float(CLASS_FREE))

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

