"""Unit tests for modular procedural track generator, turns, and zone types."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.sim.mock_perception import SimulatedEntity
from src.sim.track_generator import (
    MIN_TURN_RADIUS,
    ProceduralTrackGenerator,
)


class TestModularTrackGenerator(unittest.TestCase):
    def setUp(self) -> None:
        self.gen = ProceduralTrackGenerator(default_road_width=7.0)

    def test_demo_90deg_geometry(self) -> None:
        """Verify Demo 1 generates straight + -90 degree (right) turn + courtyard."""
        track = self.gen.generate_demo_90deg(seed=42)

        self.assertEqual(track.preset_name, "demo_90deg")
        self.assertAlmostEqual(track.start_pose[0], 0.0, places=2)
        self.assertAlmostEqual(track.start_pose[1], 0.0, places=2)
        self.assertAlmostEqual(track.start_pose[2], 0.0, places=2)

        # End heading should be -90 degrees (-pi/2)
        expected_goal_heading = -math.pi / 2.0
        self.assertAlmostEqual(track.goal_pose[2], expected_goal_heading, delta=0.08)

        # Waypoints forward shape
        self.assertGreater(len(track.waypoints_forward), 50)
        self.assertEqual(track.waypoints_forward.shape[1], 3)
        self.assertEqual(track.waypoints_reverse.shape, track.waypoints_forward.shape)

        # Fair optimized max episode steps (roundtrip buffer)
        self.assertGreater(track.max_episode_steps, 3000)
        self.assertLess(track.max_episode_steps, 8000)

        # Check turn curvature radius >= MIN_TURN_RADIUS
        pts = track.waypoints_forward[:, :2]
        dx = np.gradient(pts[:, 0])
        dy = np.gradient(pts[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        curvature = np.abs(dx * ddy - dy * ddx) / np.maximum(
            1e-6, (dx**2 + dy**2) ** 1.5
        )
        # Avoid straight line endpoints where curvature is near zero
        curved_mask = curvature > 0.05
        if np.any(curved_mask):
            radii = 1.0 / curvature[curved_mask]
            min_r = float(np.min(radii))
            self.assertGreaterEqual(min_r, MIN_TURN_RADIUS - 0.5)

    def test_multi_zone_plaza_features(self) -> None:
        """Verify OpenPlazaZone creates wide perimeter walls and large box obstacles."""
        track = self.gen.generate_multi_zone_plaza(seed=101)

        self.assertEqual(track.preset_name, "multi_zone_plaza")
        self.assertGreater(track.nominal_length, 100.0)

        # Check for large box obstacles (Shipping container or delivery truck)
        box_obstacles = [obs for obs in track.obstacles if obs.shape == "box"]
        self.assertGreaterEqual(len(box_obstacles), 2)
        for box_obs in box_obstacles:
            self.assertGreater(box_obs.length, 3.0)
            self.assertGreater(box_obs.width, 1.5)

        # Check for dynamic obstacle
        dynamic_obs = [obs for obs in track.obstacles if obs.is_dynamic]
        self.assertGreaterEqual(len(dynamic_obs), 1)
        self.assertGreater(dynamic_obs[0].patrol_distance, 5.0)

        # Check wide perimeter walls (distance between left and right walls in plaza >= 25m)
        plaza_left_walls = [
            p for p in track.wall_segments if abs(p[0][1]) > 10.0 or abs(p[1][1]) > 10.0
        ]
        self.assertGreater(len(plaza_left_walls), 0)

    def test_random_track_generation(self) -> None:
        """Verify random track scales inside range with fair max_episode_steps."""
        track = self.gen.generate_random_track(
            seed=555, min_length=120.0, max_length=150.0
        )

        self.assertGreaterEqual(track.nominal_length, 100.0)
        self.assertLessEqual(track.nominal_length, 180.0)
        self.assertGreater(len(track.obstacles), 4)

        # Truncation scaling matches track length
        nominal_roundtrip = 2.0 * track.nominal_length + 15.0
        expected_min_steps = 60 + int((nominal_roundtrip / 1.5) * (2.0 / 0.05))
        self.assertGreaterEqual(track.max_episode_steps, expected_min_steps - 200)

    def test_json_roundtrip_with_new_fields(self) -> None:
        """Verify JSON export/load retains preset_name, box shapes, and dynamic fields."""
        track = self.gen.generate_demo_90deg(seed=42)
        with tempfile.TemporaryDirectory() as tmpdir:
            json_p = Path(tmpdir) / "track.json"
            track.export_json(json_p)
            self.assertTrue(json_p.exists())

            loaded = track.load_json(json_p)
            self.assertEqual(loaded.preset_name, track.preset_name)
            self.assertEqual(loaded.nominal_length, track.nominal_length)
            self.assertEqual(loaded.max_episode_steps, track.max_episode_steps)
            self.assertEqual(len(loaded.obstacles), len(track.obstacles))

            orig_dyn = [o for o in track.obstacles if o.is_dynamic]
            loaded_dyn = [o for o in loaded.obstacles if o.is_dynamic]
            self.assertEqual(len(orig_dyn), len(loaded_dyn))
            if orig_dyn:
                self.assertEqual(
                    orig_dyn[0].patrol_distance, loaded_dyn[0].patrol_distance
                )

    def test_dynamic_entity_patrol_step(self) -> None:
        """Verify dynamic obstacle steps along velocity and bounces at patrol boundary."""
        entity = SimulatedEntity(
            name="test_ped",
            class_id=2,
            class_name="Person",
            world_x=0.0,
            world_y=0.0,
            world_vx=1.0,
            world_vy=0.0,
            radius=0.4,
            is_dynamic=True,
            patrol_distance=3.0,
            origin_x=0.0,
            origin_y=0.0,
        )

        # Step 2 seconds at dt=0.1s -> x = 2.0m (under 3.0m)
        for _ in range(20):
            entity.step(0.1)
        self.assertAlmostEqual(entity.world_x, 2.0, places=2)
        self.assertAlmostEqual(entity.world_vx, 1.0, places=2)

        # Step 15 more steps (1.5s) -> exceeds 3.0m, velocity should reverse!
        for _ in range(15):
            entity.step(0.1)
        self.assertLess(entity.world_vx, 0.0)


if __name__ == "__main__":
    unittest.main()
