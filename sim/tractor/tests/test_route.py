"""Geometry and closed-loop route tests, without native Gazebo dependencies."""

import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_model import ROOT, create_route_world
from control import approach
from geometry import extract_geometry
from obstacles import ObstacleGuard, VehicleState
from route import RoutePath, angle_error, load_route_config


class RouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geometry = extract_geometry(ROOT / "cad/TRACTOR.stp")
        cls.config = load_route_config(cls.geometry)
        cls.route = RoutePath(cls.geometry, cls.config)

    def test_straight_curve_and_return_geometry(self):
        route = self.route
        straight = [p for p in route.points if p.phase == "IDA_RECTA"]
        self.assertAlmostEqual(straight[-1].s, 110)
        self.assertAlmostEqual(straight[-1].x - straight[0].x, 110)
        entry, exit_point = route.at(route.turn_start_s), route.at(route.turn_end_s)
        self.assertLess(
            math.hypot(entry.x - exit_point.x, entry.y - exit_point.y), 1e-8
        )
        self.assertAlmostEqual(abs(angle_error(exit_point.yaw - entry.yaw)), math.pi)
        final = route.points[-1]
        offset = self.geometry["rear"]["x_m"]
        self.assertAlmostEqual(final.x - offset * math.cos(final.yaw), 0)
        self.assertAlmostEqual(final.y - offset * math.sin(final.yaw), 0)
        self.assertAlmostEqual(angle_error(final.yaw - math.pi), 0)
        self.assertTrue(all(b.s > a.s for a, b in zip(route.points, route.points[1:])))
        self.assertLess(max(abs(math.atan(p.curvature * 2)) for p in route.points), 0.4)

    def test_gate_has_clearance_on_centreline_both_directions(self):
        for heading in (math.pi / 2, -math.pi / 2):
            guard = ObstacleGuard(self.geometry)
            vehicle = VehicleState(
                self.route.gate_x, self.route.gate_y, heading, 0.45, 0
            )
            self.assertTrue(
                guard.decide(
                    vehicle, self.route.gate_observations(), 0.45, 0, 0, True
                ).allowed
            )
            offset = self.config["layout"]["gate_width_m"] / 2
            vehicle = VehicleState(
                self.route.gate_x + offset, self.route.gate_y - 1, heading, 0.45, 0
            )
            self.assertFalse(
                guard.decide(
                    vehicle, self.route.gate_observations(), 0.45, 0, 0, True
                ).allowed
            )

    def test_obstacles_trigger_once_in_each_direction(self):
        crossers = self.route.crossers()
        self.assertEqual(len(crossers), 4)
        for name, spec in crossers.items():
            self.assertGreater(spec["trigger_s"], 0)
            if name.endswith("return"):
                self.assertGreater(spec["trigger_s"], self.route.turn_end_s)
            else:
                self.assertLess(spec["trigger_s"], 110)

    def test_reject_unusable_gate_or_turn_radius(self):
        source = (ROOT / "route.toml").read_text(encoding="utf-8")
        for before, after in (
            ("gate_width_m = 3.5", "gate_width_m = 2.0"),
            ("curve_radius_m = 6.0", "curve_radius_m = 2.0"),
        ):
            with patch(
                "route.Path.read_text", return_value=source.replace(before, after)
            ):
                with self.assertRaises(ValueError):
                    load_route_config(self.geometry)

    def test_world_contains_full_ground_collidable_gate_and_four_actors(self):
        world = create_route_world(self.geometry, self.config).find("world")
        self.assertEqual(world.attrib["name"], "tractor_route")
        size = world.findtext("model[@name='ground']/link/collision/geometry/box/size")
        self.assertGreater(float(size.split()[0]), 150)
        for name in ("gate_post_-1", "gate_post_1", *self.route.crossers()):
            self.assertIsNotNone(world.find(f"model[@name='{name}']/link/collision"))
        self.assertIsNone(world.find("model[@name='east_boundary']"))

    def test_closed_loop_ackermann_completes_without_jumping_route_loops(self):
        vehicle = VehicleState(0, 0, 0, 0, 0)
        progress, steering, speed = 0.0, 0.0, 0.0
        max_error = 0
        dt = 0.05
        for step in range(16000):
            new_progress, error = self.route.project(vehicle, progress)
            self.assertLess(new_progress - progress, 0.15)
            progress = new_progress
            max_error = max(max_error, error)
            target_speed, target_steering = self.route.command(vehicle, progress)
            if (
                self.route.length - progress < 0.4
                and math.hypot(vehicle.x, vehicle.y) < 0.2
            ):
                break
            speed = approach(speed, target_speed, 0.6 * dt)
            steering = approach(steering, target_steering, 0.6 * dt)
            yaw_rate = speed * math.tan(steering) / 2
            rx, ry = self.route.rear_position(vehicle)
            yaw = vehicle.yaw + yaw_rate * dt
            rx += speed * math.cos(vehicle.yaw + yaw_rate * dt / 2) * dt
            ry += speed * math.sin(vehicle.yaw + yaw_rate * dt / 2) * dt
            vehicle = VehicleState(
                rx + math.cos(yaw), ry + math.sin(yaw), yaw, speed, step * dt
            )
        self.assertGreater(progress, self.route.length - 0.4)
        self.assertLess(math.hypot(vehicle.x, vehicle.y), 0.2)
        self.assertLess(abs(angle_error(vehicle.yaw - math.pi)), 0.1)
        self.assertLess(max_error, 0.4)


if __name__ == "__main__":
    unittest.main()
