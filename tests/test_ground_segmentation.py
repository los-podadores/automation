"""Unit tests for GroundSegmentationEngine: terrain, small animals, curbs, and ditches."""

from __future__ import annotations

import unittest
import numpy as np

from src.sim.ground_segmentation import GroundSegmentationEngine, SegmentationResult


class TestGroundSegmentationEngine(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = GroundSegmentationEngine(
            num_zones=4,
            num_sectors=12,
            max_range=15.0,
            ground_epsilon=0.05,
            small_obstacle_min_h=0.06,
            ditch_threshold=-0.12,
        )

    def test_flat_ground_classification(self) -> None:
        """Verify that a planar ground cloud has all points classified as ground."""
        rng = np.random.default_rng(42)
        # 1000 points on flat ground z=0 with slight terrain noise (+/- 0.02m)
        r = rng.uniform(0.5, 10.0, size=1000)
        theta = rng.uniform(-np.pi, np.pi, size=1000)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = rng.normal(0.0, 0.015, size=1000)
        points = np.column_stack([x, y, z])

        result: SegmentationResult = self.engine.segment(points)

        self.assertGreater(len(result.ground_points), 950)
        self.assertEqual(len(result.obstacle_points), 0)
        self.assertEqual(len(result.negative_hazards), 0)
        self.assertLess(result.roughness, 0.005)

    def test_small_animal_detection(self) -> None:
        """Verify that a small animal (10-15cm tall) placed on the lawn is classified as an obstacle."""
        rng = np.random.default_rng(101)
        # Background lawn: 500 ground points
        r = rng.uniform(1.0, 5.0, size=500)
        theta = rng.uniform(-0.5, 0.5, size=500)
        x_ground = r * np.cos(theta)
        y_ground = r * np.sin(theta)
        z_ground = rng.normal(0.0, 0.01, size=500)
        ground_pts = np.column_stack([x_ground, y_ground, z_ground])

        # Small animal (hedgehog/rabbit) at x=2.5m, y=0.0m with height 0.12m
        animal_pts = np.column_stack([
            rng.uniform(2.4, 2.6, size=50),
            rng.uniform(-0.1, 0.1, size=50),
            rng.uniform(0.07, 0.14, size=50),  # 7cm to 14cm above ground
        ])

        combined = np.vstack([ground_pts, animal_pts])
        result = self.engine.segment(combined)

        # The animal points must be classified as obstacle_points
        self.assertGreater(len(result.obstacle_points), 40)
        # Ground points must remain predominantly in ground
        self.assertGreater(len(result.ground_points), 450)

    def test_ditch_negative_obstacle_detection(self) -> None:
        """Verify that a ditch or step drop-off is classified into negative_hazards."""
        rng = np.random.default_rng(77)
        # Flat approach terrain at z=0.0
        x_ground = rng.uniform(1.0, 3.0, size=300)
        y_ground = rng.uniform(-1.0, 1.0, size=300)
        z_ground = rng.normal(0.0, 0.01, size=300)

        # Ditch floor at x=3.5 to 5.0m, depth -0.25m
        x_ditch = rng.uniform(3.5, 5.0, size=150)
        y_ditch = rng.uniform(-1.0, 1.0, size=150)
        z_ditch = np.full(150, -0.25) + rng.normal(0.0, 0.01, size=150)

        combined = np.vstack([
            np.column_stack([x_ground, y_ground, z_ground]),
            np.column_stack([x_ditch, y_ditch, z_ditch]),
        ])

        result = self.engine.segment(combined)
        self.assertGreater(len(result.negative_hazards), 50)

    def test_empty_and_out_of_range_handling(self) -> None:
        """Verify handling of empty or out-of-range arrays."""
        empty_res = self.engine.segment(np.empty((0, 3), dtype=np.float32))
        self.assertEqual(len(empty_res.ground_points), 0)
        self.assertEqual(len(empty_res.obstacle_points), 0)

        # Points outside max range
        far_pts = np.array([[50.0, 0.0, 0.0], [0.05, 0.0, 0.0]])
        far_res = self.engine.segment(far_pts)
        self.assertEqual(len(far_res.ground_points), 0)


if __name__ == "__main__":
    unittest.main()
