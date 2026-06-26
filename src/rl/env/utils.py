"""Low-level utility functions used by the environment sub-modules."""

from __future__ import annotations

import numpy as np


def total_variation(
    img: np.ndarray,
    img2: np.ndarray | None = None,
    mode: str = "sym-iso",
) -> float:
    """Compute the total variation (smoothness metric) of a coverage map.

    Args:
        img: 2-D array representing the coverage map.
        img2: Optional second array; when provided the per-pixel minimum
            difference is used (useful for obstacle-aware TV).
        mode: One of ``"sym-iso"``, ``"non-sym-iso"``, or ``"non-iso"``.

    Returns:
        Scalar total variation value.

    Raises:
        ValueError: If *mode* is not one of the accepted values.
    """
    if mode not in ("sym-iso", "non-sym-iso", "non-iso"):
        raise ValueError(f"Unknown mode {mode!r}")
    arr = img.astype(float)
    diff1 = np.abs(arr[1:, :] - arr[:-1, :])
    diff2 = np.abs(arr[:, 1:] - arr[:, :-1])
    if img2 is not None:
        if arr.shape != img2.shape:
            raise ValueError(f"Shape mismatch: {arr.shape} vs {img2.shape}")
        img2_arr = np.maximum(arr, img2)
        diff1 = np.minimum(diff1, np.abs(img2_arr[1:, :] - img2_arr[:-1, :]))
        diff2 = np.minimum(diff2, np.abs(img2_arr[:, 1:] - img2_arr[:, :-1]))
    if mode == "sym-iso":
        tv = (
            np.sum(np.sqrt(diff1[:, 1:] ** 2 + diff2[1:, :] ** 2))
            + np.sum(np.sqrt(diff1[:, 1:] ** 2 + diff2[:-1, :] ** 2))
            + np.sum(np.sqrt(diff1[:, :-1] ** 2 + diff2[:-1, :] ** 2))
            + np.sum(np.sqrt(diff1[:, :-1] ** 2 + diff2[1:, :] ** 2))
        )
        return float(tv / 4)
    elif mode == "non-sym-iso":
        return float(np.sum(np.sqrt(diff1[:, :-1] ** 2 + diff2[:-1, :] ** 2)))
    else:
        return float(np.sum(diff1) + np.sum(diff2))
