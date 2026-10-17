"""RGB-D decoding and an atomic latest-frame exchange; no inference or control."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def decode_image(message, depth: bool = False):
    width, height, step = message.width, message.height, message.step
    if not 0 < width <= 4096 or not 0 < height <= 4096:
        raise ValueError("Dimensiones de imagen inválidas")
    # gz.msgs.PixelFormatType: RGB_INT8=3, BGR_INT8=8, R_FLOAT32=13.
    channels, bytes_per_pixel = (1, 4) if depth else (3, 3)
    if message.pixel_format_type not in ((13,) if depth else (3, 8)):
        raise ValueError(f"Formato de imagen no soportado: {message.pixel_format_type}")
    if step < width * bytes_per_pixel or len(message.data) != height * step:
        raise ValueError("Tamaño o stride de imagen inválido")
    rows = np.frombuffer(message.data, dtype=np.uint8).reshape(height, step)
    packed = rows[:, : width * bytes_per_pixel].copy()
    if depth:
        return packed.view("<f4").reshape(height, width)
    rgb = packed.reshape(height, width, channels)
    return rgb[:, :, ::-1].copy() if message.pixel_format_type == 8 else rgb


def write_frame(folder: Path, rgb, depth, stamp: float, sequence: int):
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / "frame.pending.npz"
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream, rgb=rgb, depth=depth, stamp=stamp, sequence=sequence
        )
    os.replace(temporary, folder / "frame.npz")


def read_frame(folder: Path):
    with np.load(folder / "frame.npz", allow_pickle=False) as data:
        rgb, depth = data["rgb"], data["depth"]
        if (
            rgb.dtype != np.uint8
            or rgb.ndim != 3
            or rgb.shape[2] != 3
            or depth.shape != rgb.shape[:2]
        ):
            raise ValueError("Par RGB-D inválido")
        stamp, sequence = float(data["stamp"]), int(data["sequence"])
        if not np.isfinite(stamp):
            raise ValueError("Marca de tiempo inválida")
        return rgb, depth, stamp, sequence


def detection_depth(depth, bbox):
    """Median forward depth in the central bbox region; no monocular guess."""
    x1, y1, x2, y2 = bbox
    height, width = depth.shape
    x1, x2 = max(0, int(x1)), min(width, int(x2))
    y1, y2 = max(0, int(y1)), min(height, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    dx, dy = (x2 - x1) // 4, (y2 - y1) // 4
    region = depth[y1 + dy : y2 - dy, x1 + dx : x2 - dx]
    valid = region[np.isfinite(region) & (region >= 0.15) & (region < 25)]
    if valid.size < 5:
        return None
    return float(np.median(valid))
