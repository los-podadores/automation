"""Photo fixtures preserve route geometry and a conservative collision envelope."""

import hashlib
import json
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_model import ROOT, create_route_world
from geometry import extract_geometry
from photo_posters import ASSETS, HIDDEN_Z
from route import RoutePath, load_route_config


class PhotoPosterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geometry = extract_geometry(ROOT / "cad/TRACTOR.stp")
        cls.config = load_route_config(cls.geometry)
        cls.route = RoutePath(cls.geometry, cls.config)
        cls.world = create_route_world(
            cls.geometry, cls.config, photo_posters=True
        ).find("world")

    def test_textures_match_documented_sources(self):
        manifest = json.loads((ASSETS / "sources.json").read_text(encoding="utf-8"))
        for name, metadata in manifest["textures"].items():
            self.assertEqual(
                hashlib.sha256((ASSETS / name).read_bytes()).hexdigest(),
                metadata["sha256"],
            )

    def test_photos_start_hidden_and_have_real_mesh_textures(self):
        for name in self.route.crossers(True):
            model = self.world.find(f"model[@name='{name}']")
            self.assertEqual(float(model.findtext("pose").split()[2]), HIDDEN_Z)
            visual = model.find("link/visual")
            self.assertEqual(visual.attrib["name"], "photograph")
            self.assertTrue(Path(visual.findtext("geometry/mesh/uri")).is_file())
            self.assertTrue(
                Path(visual.findtext("material/pbr/metal/albedo_map")).is_file()
            )
            self.assertEqual(len(model.findall("link/visual")), 1)

    def test_guard_radius_covers_the_whole_panel_footprint(self):
        for name, spec in self.route.crossers(True).items():
            size = self.world.findtext(
                f"model[@name='{name}']/link/collision/geometry/box/size"
            )
            width, depth, _ = map(float, size.split())
            self.assertGreaterEqual(spec["radius"], math.hypot(width / 2, depth / 2))

    def test_crossings_activate_ahead_and_exit_both_road_and_camera(self):
        for name, spec in self.route.crossers(True).items():
            original = self.route.crossers()[name]
            self.assertEqual(spec["trigger_s"], original["trigger_s"] - 4)
            self.assertGreater(spec["trigger_s"], 0)
            self.assertLess(spec["start_y"] * spec["end_y"], 0)
            self.assertGreaterEqual(abs(spec["end_y"]), 10)
            self.assertTrue(spec["photo_poster"])

    def test_standard_route_retains_geometric_actors(self):
        for spec in self.route.crossers().values():
            self.assertNotIn("photo_poster", spec)
        normal = create_route_world(self.geometry, self.config).find("world")
        self.assertIsNone(
            normal.find("model[@name='person_out']/link/visual/geometry/mesh")
        )


if __name__ == "__main__":
    unittest.main()
