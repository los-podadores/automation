"""Top-level utilities — re-exports ``total_variation`` from env for backward compat."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from env.utils import total_variation


def seed_everything(seed: int, env=None) -> None:
    """Seed all RNGs for reproducibility."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if env is not None:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        env.action_space.np_random.seed(seed)
        env.observation_space.np_random.seed(seed)


__all__ = ["seed_everything", "total_variation"]
