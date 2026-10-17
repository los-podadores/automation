"""Image transport correctness, frame pairing and missing-depth behaviour."""

import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camera_bridge import CameraFrames
from camera_io import decode_image, detection_depth
from build_model import ROOT, create_model, load_parameters
from geometry import extract_geometry


def message(data, pixel_format, width=2, height=1, step=None, stamp=1):
    return SimpleNamespace(
        width=width,
        height=height,
        step=step or len(data) // height,
        data=data,
        pixel_format_type=pixel_format,
        header=SimpleNamespace(stamp=SimpleNamespace(sec=stamp, nsec=0)),
    )


class CameraTests(unittest.TestCase):
    def test_decode_rgb_with_row_padding_and_bgr(self):
        rgb = decode_image(message(bytes([255, 0, 0, 0, 255, 0, 99, 99]), 3, step=8))
        self.assertEqual(rgb.tolist(), [[[255, 0, 0], [0, 255, 0]]])
        bgr = decode_image(message(bytes([0, 0, 255, 0, 255, 0]), 8))
        np.testing.assert_array_equal(rgb, bgr)

    def test_decode_float_depth_preserves_invalid_values(self):
        data = np.array([2.5, float("inf")], dtype="<f4").tobytes()
        result = decode_image(message(data, 13), depth=True)
        self.assertEqual(result[0, 0], 2.5)
        self.assertTrue(np.isinf(result[0, 1]))

    def test_reject_incorrect_format_and_stride(self):
        for msg in (
            message(b"123456", 1),
            message(b"12345", 3),
            message(b"123456", 3, width=0),
        ):
            with self.subTest(msg=msg), self.assertRaises(ValueError):
                decode_image(msg)

    def test_depth_uses_inner_roi_and_never_invents_missing_distance(self):
        depth = np.full((20, 20), float("inf"), dtype=np.float32)
        self.assertIsNone(detection_depth(depth, (0, 0, 20, 20)))
        depth[5:15, 5:15] = 3.0
        depth[6, 6] = float("nan")
        self.assertEqual(detection_depth(depth, (0, 0, 20, 20)), 3.0)
        self.assertIsNone(detection_depth(depth, (30, 30, 40, 40)))

    def test_pair_requires_matching_timestamp_and_recent_reception(self):
        frames = CameraFrames.__new__(CameraFrames)
        frames.lock = threading.Lock()
        frames.rgb = frames.depth = None
        frames.error = ""
        frames.received = 0
        self.assertIsNone(frames.snapshot())
        frames.callback(False)(message(bytes(6), 3))
        self.assertIsNone(frames.snapshot())
        frames.callback(True)(
            message(np.array([1, 2], dtype="<f4").tobytes(), 13, stamp=2)
        )
        self.assertIsNone(frames.snapshot())
        frames.callback(True)(message(np.array([1, 2], dtype="<f4").tobytes(), 13))
        self.assertIsNotNone(frames.snapshot())
        frames.received = time.monotonic() - 2
        self.assertIsNone(frames.snapshot())

    def test_camera_is_on_chassis_without_changing_articulated_tree(self):
        model = create_model(
            extract_geometry(ROOT / "cad/TRACTOR.stp"), load_parameters()
        ).find("model")
        sensor = model.find("link[@name='chassis']/sensor")
        self.assertEqual(sensor.attrib["type"], "rgbd_camera")
        self.assertEqual(sensor.findtext("camera/image/width"), "640")
        self.assertEqual(sensor.findtext("topic"), "/tractor/front_camera")
        self.assertEqual(len(model.findall("link")), 7)


if __name__ == "__main__":
    unittest.main()
