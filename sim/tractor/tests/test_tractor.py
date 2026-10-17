"""Offline checks; runtime checks in Gazebo are a separate validation step."""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import subprocess
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_model import (
    ROOT,
    create_model,
    create_obstacle_world,
    create_world,
    load_parameters,
)
from control import KeyboardDrive
from geometry import extract_geometry
from obstacles import Observation, ObstacleGuard, VehicleState, straight_path_steering
from teleop import GazeboCommands


class TractorModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geometry = extract_geometry(ROOT / "cad/TRACTOR.stp")
        cls.parameters = load_parameters()
        cls.model = create_model(cls.geometry, cls.parameters).find("model")

    def test_dimensions_and_source_units(self):
        g = self.geometry
        self.assertEqual(g["source_units"], "mm")
        self.assertAlmostEqual(g["wheelbase_m"], 2.0)
        self.assertAlmostEqual(g["rear"]["wheel_radius_m"], 0.5)
        self.assertAlmostEqual(g["front"]["wheel_radius_m"], 0.37)
        self.assertAlmostEqual(g["rear"]["wheel_width_m"], 0.27)
        self.assertAlmostEqual(g["front"]["wheel_width_m"], 0.16)
        self.assertAlmostEqual(g["rear"]["track_m"], 1.4)
        self.assertAlmostEqual(g["front"]["track_m"], 1.29)
        self.assertAlmostEqual(g["beam_width_m"], 0.15)
        self.assertAlmostEqual(g["beam_height_m"], 0.15)
        digest = hashlib.sha256((ROOT / "cad/TRACTOR.stp").read_bytes()).hexdigest()
        self.assertEqual(g["source_sha256"], digest)

    def test_articulated_tree_is_connected(self):
        links = {link.attrib["name"] for link in self.model.findall("link")}
        joints = self.model.findall("joint")
        self.assertEqual(len(links), 7)
        self.assertEqual(len(joints), 6)
        children = set()
        reached = {"chassis"}
        for joint in joints:
            self.assertIn(joint.findtext("parent"), links)
            child = joint.findtext("child")
            self.assertIn(child, links)
            self.assertNotIn(child, children)
            children.add(child)
        for _ in links:
            for joint in joints:
                if joint.findtext("parent") in reached:
                    reached.add(joint.findtext("child"))
        self.assertEqual(reached, links)

    def test_joint_axes_and_inner_steering_limit(self):
        g = self.geometry
        delta = self.parameters["control"]["max_steering_rad"]
        inside = math.atan(
            g["wheelbase_m"]
            / (g["wheelbase_m"] / math.tan(delta) - g["front"]["axle_length_m"] / 2)
        )
        for joint in self.model.findall("joint"):
            if "steering" in joint.attrib["name"]:
                self.assertEqual(joint.findtext("axis/xyz"), "0 0 1")
                self.assertLess(inside, float(joint.findtext("axis/limit/upper")))
            else:
                self.assertEqual(joint.findtext("axis/xyz"), "0 1 0")

    def test_positive_physically_valid_inertias(self):
        for link in self.model.findall("link"):
            self.assertGreater(float(link.findtext("inertial/mass")), 0)
            diagonal = [
                float(link.findtext(f"inertial/inertia/{key}"))
                for key in ("ixx", "iyy", "izz")
            ]
            self.assertTrue(all(value > 0 for value in diagonal))
            for value in diagonal:
                self.assertLessEqual(value, sum(diagonal) - value + 1e-9)

    def test_drive_actuates_only_rear_wheels(self):
        plugins = {
            p.attrib["name"].split("::")[-1]: p for p in self.model.findall("plugin")
        }
        self.assertEqual(plugins["AckermannSteering"].findtext("steering_only"), "true")
        self.assertEqual(plugins["DiffDrive"].findtext("left_joint"), "rear_left_spin")
        self.assertEqual(
            plugins["DiffDrive"].findtext("right_joint"), "rear_right_spin"
        )
        self.assertAlmostEqual(
            float(plugins["DiffDrive"].findtext("wheel_radius")), 0.5
        )

    def test_ground_clearance_and_local_assets(self):
        for axle in (self.geometry["rear"], self.geometry["front"]):
            clearance = axle["z_m"] - axle["wheel_radius_m"]
            self.assertGreaterEqual(clearance, -1e-6)
            self.assertLess(clearance, 0.002)
        world = create_world()
        self.assertEqual(
            world.findtext("world/include/uri"), "../models/tractor/model.sdf"
        )
        self.assertTrue((ROOT / "models/tractor/model.sdf").is_file())
        self.assertEqual(world.findtext("world/physics/max_step_size"), "0.001")

    def test_generated_model_matches_sources(self):
        saved = ET.parse(ROOT / "models/tractor/model.sdf").getroot()
        rebuilt = create_model(self.geometry, self.parameters)
        ET.indent(saved)
        ET.indent(rebuilt)
        self.assertEqual(ET.tostring(saved), ET.tostring(rebuilt))

    def test_unique_names_within_every_link(self):
        for document in (self.model, create_world(), create_obstacle_world()):
            for link in document.iter("link"):
                with self.subTest(link=link.attrib["name"]):
                    names = [
                        child.attrib["name"] for child in link if "name" in child.attrib
                    ]
                    self.assertEqual(
                        len(names),
                        len(set(names)),
                        f"Duplicate named children in {link.attrib['name']}: {names}",
                    )


@unittest.skipUnless(shutil.which("gz"), "Gazebo is not installed on this platform")
class GazeboValidationTests(unittest.TestCase):
    def validate(self, relative_path):
        env = os.environ.copy()
        env["GZ_SIM_RESOURCE_PATH"] = (
            str(ROOT / "models") + os.pathsep + env.get("GZ_SIM_RESOURCE_PATH", "")
        )
        result = subprocess.run(
            ["gz", "sdf", "-k", str(ROOT / relative_path)],
            env=env,
            cwd=ROOT / "worlds",
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, (result.stdout + result.stderr)[-2000:])

    def test_model_is_accepted_by_gazebo(self):
        self.validate("models/tractor/model.sdf")

    def test_world_is_accepted_by_gazebo(self):
        self.validate("worlds/tractor_field.sdf")

    def test_obstacle_world_is_accepted_by_gazebo(self):
        self.validate("worlds/tractor_obstacles.sdf")


class ObstacleDecisionTests(unittest.TestCase):
    def setUp(self):
        self.guard = ObstacleGuard(extract_geometry(ROOT / "cad/TRACTOR.stp"))
        self.vehicle = VehicleState(0, 0, 0, 0.6, 0)
        self.person = Observation("person", "persona", 3, 0, 0.36)

    def decide(self, observations, now=0, speed=0.6, steering=0, fresh=True):
        return self.guard.decide(
            self.vehicle, observations, speed, steering, now, fresh
        )

    def test_person_in_path_brakes_then_waits(self):
        decision = self.decide([self.person])
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.state, "FRENAR")
        self.vehicle = VehicleState(0, 0, 0, 0, 1)
        self.assertEqual(self.decide([self.person]).state, "ESPERAR")

    def test_side_obstacle_does_not_block_straight_path(self):
        dog = Observation("dog", "perro", 3, 3, 0.65)
        self.assertTrue(self.decide([dog]).allowed)

    def test_unrecognised_class_in_path_also_blocks(self):
        self.assertFalse(self.decide([Observation("box", "objeto", 3, 0, 0.2)]).allowed)

    def test_reverse_checks_behind_vehicle(self):
        rear = Observation("rear", "persona", -3, 0, 0.36)
        self.assertTrue(self.decide([rear]).allowed)
        self.assertFalse(self.decide([rear], speed=-0.35).allowed)

    def test_turn_checks_swept_body_not_only_current_position(self):
        obstacle = Observation("corner", "persona", 3, 2, 0.1)
        self.assertTrue(self.decide([obstacle]).allowed)
        self.assertFalse(self.decide([obstacle], steering=0.4).allowed)

    def test_lost_perception_blocks_and_requires_clear_hold_after_recovery(self):
        self.assertEqual(self.decide([], fresh=False).state, "SIN DATOS")
        self.assertFalse(self.decide([], now=1).allowed)
        self.assertTrue(self.decide([], now=1.81).allowed)
        self.assertFalse(self.guard.decide(None, [], 0.6, 0, 2, True).allowed)

    def test_reappearing_obstacle_resets_clear_hold(self):
        self.decide([self.person])
        self.assertFalse(self.decide([], now=1).allowed)
        self.assertFalse(self.decide([], now=1.7).allowed)
        self.decide([self.person], now=1.75)
        self.assertFalse(self.decide([], now=2).allowed)
        self.assertFalse(self.decide([], now=2.7).allowed)
        self.assertTrue(self.decide([], now=2.81).allowed)

    def test_invalid_observation_or_state_blocks(self):
        for obs in (
            Observation("bad", "perro", float("nan"), 0, 0.5),
            Observation("bad", "perro", 0, 0, -1),
        ):
            with self.subTest(observation=obs):
                self.assertEqual(self.decide([obs]).state, "SIN DATOS")
        self.vehicle = VehicleState(float("inf"), 0, 0, 0, 0)
        self.assertEqual(self.decide([]).state, "SIN DATOS")

    def test_path_controller_corrects_toward_lane(self):
        self.assertLess(straight_path_steering(VehicleState(0, 1, 0, 0, 0), 2), 0)
        self.assertGreater(straight_path_steering(VehicleState(0, -1, 0, 0, 0), 2), 0)
        self.assertEqual(straight_path_steering(VehicleState(0, 0, 0, 0, 0), 2), 0)


class KeyboardTests(unittest.TestCase):
    def setUp(self):
        self.drive = KeyboardDrive(2, 0.6, 0.35, 0.4, 0.6, 0.6)

    def advance(self, keys, steps=100):
        self.drive.keys = set(keys)
        result = None
        for _ in range(steps):
            result = self.drive.step(0.05)
        return result

    def test_no_rotation_in_place(self):
        speed, steering, yaw = self.advance(["a"])
        self.assertEqual(speed, 0)
        self.assertAlmostEqual(steering, 0.4)
        self.assertEqual(yaw, 0)

    def test_forward_left_curve_and_rear_speed_ratio(self):
        v, delta, yaw = self.advance(["w", "a"])
        self.assertAlmostEqual(v, 0.6)
        self.assertAlmostEqual(yaw, v * math.tan(delta) / 2)
        left_rate = (v - yaw * 1.4 / 2) / 0.5
        right_rate = (v + yaw * 1.4 / 2) / 0.5
        self.assertGreater(left_rate, 0)
        self.assertGreater(right_rate, left_rate)

    def test_reverse_changes_yaw_sign(self):
        v, delta, yaw = self.advance(["s", "a"])
        self.assertAlmostEqual(v, -0.35)
        self.assertGreater(delta, 0)
        self.assertLess(yaw, 0)

    def test_release_stops_and_centers(self):
        self.advance(["w", "d"])
        speed, steering, yaw = self.advance([])
        self.assertEqual((speed, steering, yaw), (0, 0, 0))

    def test_conflicting_commands_cancel(self):
        self.assertEqual(self.advance(["w", "s", "a", "d"]), (0, 0, 0))

    def test_acceleration_and_stop(self):
        self.drive.keys.add("w")
        speed, _delta, _yaw = self.drive.step(0.05)
        self.assertAlmostEqual(speed, 0.03)
        self.drive.stop()
        self.assertEqual(self.drive.step(0.05), (0, 0, 0))
        self.assertFalse(self.drive.keys)

    def test_bad_or_stalled_time_step(self):
        for dt in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.drive.step(dt)
        self.drive.keys.add("w")
        self.assertLessEqual(self.drive.step(100)[0], 0.06)


class PublisherTests(unittest.TestCase):
    def setUp(self):
        # Exercise the adapter without requiring native Gazebo bindings.
        self.commands = GazeboCommands.__new__(GazeboCommands)
        self.commands.double_type = SimpleNamespace
        self.commands.twist_type = lambda: SimpleNamespace(
            linear=SimpleNamespace(x=0), angular=SimpleNamespace(z=0)
        )
        self.commands.steering_pub = Mock()
        self.commands.drive_pub = Mock()

    def test_stop_still_reaches_rear_axle_if_steering_publish_fails(self):
        self.commands.steering_pub.publish.return_value = False
        self.commands.drive_pub.publish.return_value = True
        with self.assertRaises(RuntimeError):
            self.commands.stop()
        message = self.commands.drive_pub.publish.call_args.args[0]
        self.assertEqual(message.linear.x, 0)
        self.assertEqual(message.angular.z, 0)

    def test_simulation_messages_keep_angle_and_yaw_separate(self):
        self.commands.publish(0.6, 0.4, 0.6 * math.tan(0.4) / 2)
        angle = self.commands.steering_pub.publish.call_args.args[0]
        velocity = self.commands.drive_pub.publish.call_args.args[0]
        self.assertEqual(angle.data, 0.4)
        self.assertEqual(velocity.linear.x, 0.6)
        self.assertAlmostEqual(velocity.angular.z, 0.6 * math.tan(0.4) / 2)


if __name__ == "__main__":
    unittest.main()
