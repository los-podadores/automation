"""Unit tests for GzDRL sensor adapter and Gymnasium environment."""

from __future__ import annotations

import math
import unittest
import numpy as np

from src.sim.gzdrl_sensor_adapter import (
    GzDrlSensorAdapter,
    HORIZ_MAX_RAD,
    HORIZ_MIN_RAD,
    VERT_MAX_RAD,
    VERT_MIN_RAD,
)


class TestGzDrlSensorAdapter(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = GzDrlSensorAdapter(
            front_sensor_name="tractor::chassis::lidar_front",
            rear_sensor_name="tractor::chassis::lidar_rear",
        )

    def test_grid_precomputation(self) -> None:
        v_samples, h_samples = 16, 180
        self.adapter._ensure_grid(v_samples, h_samples)
        self.assertEqual(self.adapter._cached_shape, (16, 180))
        self.assertEqual(self.adapter._front_dir_x.shape, (16, 180))
        self.assertEqual(self.adapter._front_dir_y.shape, (16, 180))
        self.assertEqual(self.adapter._front_dir_z.shape, (16, 180))
        self.assertEqual(self.adapter._rear_dir_x.shape, (16, 180))
        self.assertEqual(self.adapter._rear_dir_y.shape, (16, 180))
        self.assertEqual(self.adapter._rear_dir_z.shape, (16, 180))

    def test_fast_grid_precomputation(self) -> None:
        v_samples, h_samples = 8, 90
        self.adapter._ensure_grid(v_samples, h_samples)
        self.assertEqual(self.adapter._cached_shape, (8, 90))
        self.assertEqual(self.adapter._front_dir_x.shape, (8, 90))

    def test_forward_ray_direction(self) -> None:
        # Check center beam of front sensor (elevation ~ 0, azimuth ~ 0)
        v_samples, h_samples = 17, 181
        self.adapter._ensure_grid(v_samples, h_samples)
        v_mid = v_samples // 2
        h_mid = h_samples // 2

        # In chassis frame, center front beam should point forward (+x) and downward (-z) due to pitch_down
        dir_x = self.adapter._front_dir_x[v_mid, h_mid]
        dir_y = self.adapter._front_dir_y[v_mid, h_mid]
        dir_z = self.adapter._front_dir_z[v_mid, h_mid]

        self.assertGreater(dir_x, 0.8)
        self.assertAlmostEqual(dir_y, 0.0, places=2)
        # Center elevation is mid of [-0.436, 0.262] which is negative, plus pitch_down, so z is negative
        self.assertLess(dir_z, 0.0)

    def test_mock_server_point_cloud_extraction(self) -> None:
        class MockServer:
            def get_sensor_gpu_lidar(self, name: str) -> np.ndarray:
                # Return 16x180x3 buffer with constant 2.0m range
                buf = np.zeros((16, 180, 3), dtype=np.float32)
                buf[:, :, 0] = 2.0
                return buf

        server = MockServer()
        pts = self.adapter.extract_point_cloud(server)
        self.assertEqual(pts.ndim, 2)
        self.assertEqual(pts.shape[1], 3)
        self.assertEqual(pts.shape[0], 16 * 180 * 2)  # front + rear


if __name__ == "__main__":
    unittest.main()
