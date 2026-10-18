"""Unit tests for the CAD-derived articulated tractor SDF model and LiDAR sensors."""

from __future__ import annotations

import json
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "src" / "v2" / "model"


class TestTractorModelSDF(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sdf_path = ROOT / "tractor.sdf"
        cls.json_path = ROOT / "cad_dimensions.json"
        assert cls.sdf_path.exists(), f"SDF not found at {cls.sdf_path}"
        assert cls.json_path.exists(), f"CAD dimensions not found at {cls.json_path}"

        tree = ET.parse(cls.sdf_path)
        cls.model = tree.find("model")
        assert cls.model is not None, "Model tag not found in SDF"
        cls.dimensions = json.loads(cls.json_path.read_text(encoding="utf-8"))

    def test_cad_dimensions(self) -> None:
        d = self.dimensions
        self.assertAlmostEqual(d["wheelbase_m"], 2.00, places=3)
        self.assertAlmostEqual(d["rear"]["wheel_radius_m"], 0.50, places=3)
        self.assertAlmostEqual(d["front"]["wheel_radius_m"], 0.37, places=3)
        self.assertAlmostEqual(d["rear"]["track_m"], 1.40, places=3)
        self.assertAlmostEqual(d["front"]["track_m"], 1.29, places=3)

    def test_kinematic_tree(self) -> None:
        links = {link.attrib["name"] for link in self.model.findall("link")}
        expected_links = {
            "chassis",
            "front_left_knuckle",
            "front_right_knuckle",
            "front_left_wheel",
            "front_right_wheel",
            "rear_left_wheel",
            "rear_right_wheel",
        }
        self.assertEqual(links, expected_links)

        joints = {joint.attrib["name"] for joint in self.model.findall("joint")}
        expected_joints = {
            "front_left_steering",
            "front_right_steering",
            "front_left_spin",
            "front_right_spin",
            "rear_left_spin",
            "rear_right_spin",
        }
        self.assertEqual(joints, expected_joints)

    def test_lidar_sensors(self) -> None:
        chassis = self.model.find("link[@name='chassis']")
        sensors = chassis.findall("sensor[@type='gpu_lidar']")
        sensor_topics = {s.find("topic").text for s in sensors}

        if len(sensors) == 2:
            expected_topics = {
                "/tractor/lidar/front/points",
                "/tractor/lidar/rear/points",
            }
            self.assertEqual(sensor_topics, expected_topics)
            for s in sensors:
                vert = s.find("ray/scan/vertical")
                self.assertIsNotNone(vert, "Vertical scan missing on 3D directional LiDAR")
                self.assertGreaterEqual(int(vert.find("samples").text), 16)
        elif len(sensors) == 4:
            expected_topics = {
                "/tractor/lidar/front",
                "/tractor/lidar/rear",
                "/tractor/lidar/right",
                "/tractor/lidar/left",
            }
            self.assertEqual(sensor_topics, expected_topics)
        else:
            self.fail(f"Unexpected number of LiDAR sensors: {len(sensors)}")

    def test_split_actuation_plugins(self) -> None:
        plugins = self.model.findall("plugin")
        plugin_names = {p.attrib["name"] for p in plugins}
        self.assertIn("gz::sim::systems::AckermannSteering", plugin_names)
        self.assertIn("gz::sim::systems::DiffDrive", plugin_names)
        self.assertIn("gz::sim::systems::JointStatePublisher", plugin_names)

    def test_chassis_collision_dimensions(self) -> None:
        chassis = self.model.find("link[@name='chassis']")
        coll = chassis.find("collision[@name='chassis_body_collision']")
        self.assertIsNotNone(coll)
        size_str = coll.find("geometry/box/size").text
        sizes = [float(v) for v in size_str.split()]
        self.assertAlmostEqual(sizes[0], 2.90, places=2)
        self.assertAlmostEqual(sizes[1], 1.72, places=2)
        self.assertAlmostEqual(sizes[2], 0.40, places=2)

        contact_sensor = chassis.find("sensor[@name='tractor_contact_sensor']")
        self.assertIsNotNone(contact_sensor)
        monitored_collisions = {c.text for c in contact_sensor.findall("contact/collision")}
        self.assertIn("chassis_body_collision", monitored_collisions)

    def test_imu_sensor(self) -> None:
        chassis = self.model.find("link[@name='chassis']")
        imu = chassis.find("sensor[@name='chassis_imu']")
        self.assertIsNotNone(imu)
        self.assertEqual(imu.attrib["type"], "imu")
        self.assertEqual(imu.find("topic").text, "/model/tractor/imu")


if __name__ == "__main__":
    unittest.main()
