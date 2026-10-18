"""Unit tests for modular procedural track generator, turns, and zone types."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.sim.track_generator import (
    ESTADIO_SITE,
    MIN_TURN_RADIUS,
    ProceduralTrackGenerator,
    SimulatedEntity,
)


class TestModularTrackGenerator(unittest.TestCase):
    def setUp(self) -> None:
        self.gen = ProceduralTrackGenerator(default_road_width=7.0)

    def test_estadio_ho_chi_min_geometry(self) -> None:
        """Verify site scenario generates 100m road + left turn + 60x40m yard."""
        track = self.gen.generate_estadio_ho_chi_min(seed=42)

        self.assertEqual(track.preset_name, "estadio_ho_chi_min")
        self.assertAlmostEqual(track.start_pose[0], 0.0, places=2)
        self.assertAlmostEqual(track.start_pose[1], 0.0, places=2)
        self.assertAlmostEqual(track.start_pose[2], 0.0, places=2)

        # End heading should be +90 degrees (+pi/2) after left turn
        expected_goal_heading = math.pi / 2.0
        self.assertAlmostEqual(track.goal_pose[2], expected_goal_heading, delta=0.08)

        # Waypoints forward shape
        self.assertGreater(len(track.waypoints_forward), 200)
        self.assertEqual(track.waypoints_forward.shape[1], 3)
        self.assertEqual(track.waypoints_reverse.shape, track.waypoints_forward.shape)

        # 1:1 site scale: ~164.6m nominal (100 + 12.57 + 52)
        self.assertGreater(track.nominal_length, 150.0)
        self.assertLess(track.nominal_length, 180.0)

        # Fair optimized max episode steps (roundtrip buffer)
        self.assertGreater(track.max_episode_steps, 8000)

        # Per-waypoint corridor widths: road third narrow, yard third wide
        k = len(track.waypoints_forward)
        self.assertEqual(track.corridor_half_widths.shape, (k,))
        road_hw = float(np.median(track.corridor_half_widths[: k // 4]))
        yard_hw = float(np.median(track.corridor_half_widths[int(k * 0.7) :]))
        self.assertAlmostEqual(road_hw, ESTADIO_SITE["road_width"] / 2.0, delta=0.2)
        self.assertAlmostEqual(yard_hw, ESTADIO_SITE["yard_width"] / 2.0, delta=1.0)

        # Yard is a full rectangle: goal sits goal_margin inside the far wall,
        # centered laterally, with no 7m throat near the old plaza/courtyard joint.
        gx, gy, _ = (float(v) for v in track.goal_pose)
        dists_to_walls = [
            min(math.hypot(gx - x1, gy - y1), math.hypot(gx - x2, gy - y2))
            for (x1, y1), (x2, y2) in track.wall_segments
        ]
        self.assertAlmostEqual(
            min(dists_to_walls), ESTADIO_SITE["goal_margin"], delta=1.0
        )

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
        track = self.gen.generate_estadio_ho_chi_min(seed=42)
        with tempfile.TemporaryDirectory() as tmpdir:
            json_p = Path(tmpdir) / "track.json"
            track.export_json(json_p)
            self.assertTrue(json_p.exists())

            loaded = track.load_json(json_p)
            self.assertEqual(loaded.preset_name, track.preset_name)
            self.assertEqual(loaded.nominal_length, track.nominal_length)
            self.assertEqual(loaded.max_episode_steps, track.max_episode_steps)
            self.assertEqual(len(loaded.obstacles), len(track.obstacles))
            np.testing.assert_allclose(
                loaded.corridor_half_widths, track.corridor_half_widths
            )

            orig_dyn = [o for o in track.obstacles if o.is_dynamic]
            loaded_dyn = [o for o in loaded.obstacles if o.is_dynamic]
            self.assertEqual(len(orig_dyn), len(loaded_dyn))
            if orig_dyn:
                self.assertEqual(
                    orig_dyn[0].patrol_distance, loaded_dyn[0].patrol_distance
                )

    def test_rectangular_yard_rejects_bad_entry(self) -> None:
        """Verify yard entry/goal validation fails fast on bad dimensions."""
        from src.sim.track_generator import RectangularYardZone

        with self.assertRaises(ValueError):
            RectangularYardZone(length=60.0, width=40.0, entry_width=39.0)
        with self.assertRaises(ValueError):
            RectangularYardZone(length=60.0, width=40.0, entry_offset=30.0)
        with self.assertRaises(ValueError):
            RectangularYardZone(length=60.0, width=40.0, goal_margin=2.0)

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

    def test_dynamic_ped_swept_path_clear_of_statics(self) -> None:
        """Verify every dynamic ped's full patrol corridor clears all statics."""
        from src.sim.track_generator import PED_PATROL_CLEARANCE_MARGIN

        for seed in [42, *range(100, 120)]:
            track = self.gen.generate_estadio_ho_chi_min(seed=seed)
            for ped in track.obstacles:
                if not ped.is_dynamic:
                    continue
                speed = math.hypot(ped.world_vx, ped.world_vy)
                self.assertGreater(speed, 1e-4, f"{seed}/{ped.name} has no velocity")
                dir_x, dir_y = ped.world_vx / speed, ped.world_vy / speed
                for static in track.obstacles:
                    if static.is_dynamic:
                        continue
                    seg_x, seg_y = dir_x * ped.patrol_distance, dir_y * ped.patrol_distance
                    seg_len2 = seg_x * seg_x + seg_y * seg_y
                    if seg_len2 < 1e-8:
                        clearance = math.hypot(
                            ped.origin_x - static.world_x, ped.origin_y - static.world_y
                        )
                    else:
                        t = max(
                            0.0,
                            min(
                                1.0,
                                (
                                    (static.world_x - ped.origin_x) * seg_x
                                    + (static.world_y - ped.origin_y) * seg_y
                                )
                                / seg_len2,
                            ),
                        )
                        clearance = math.hypot(
                            ped.origin_x + t * seg_x - static.world_x,
                            ped.origin_y + t * seg_y - static.world_y,
                        )
                    self.assertGreaterEqual(
                        clearance - (ped.radius + static.radius),
                        PED_PATROL_CLEARANCE_MARGIN - 1e-6,
                        f"{seed}/{ped.name} patrol clips {static.name}",
                    )


if __name__ == "__main__":
    unittest.main()
