import numpy as np
import os
import random
import torch


def seed_everything(seed, env=None):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if env is not None:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        env.action_space.np_random.seed(seed)
        env.observation_space.np_random.seed(seed)


def total_variation(img, img2=None, mode="sym-iso"):
    """
    Computes the isotropic discrete total variation of a 2D image.

    If img2 is provided, pixel variation is computed in both img and
    max(img, img2), keeping the minimum of the two at each pixel.
    This allows masking with an obstacle map.

    Modes:
      sym-iso: Symmetric and isotropic (slowest, most accurate)
      non-sym-iso: Isotropic but not symmetric (faster)
      non-iso: Non-isotropic and symmetric (fastest)
    """
    assert mode in ["sym-iso", "non-sym-iso", "non-iso"]
    img = img.astype(float)
    diff1 = np.abs(img[1:, :] - img[:-1, :])
    diff2 = np.abs(img[:, 1:] - img[:, :-1])
    if img2 is not None:
        assert img.shape == img2.shape
        img2 = np.maximum(img, img2)
        diff1 = np.minimum(diff1, np.abs(img2[1:, :] - img2[:-1, :]))
        diff2 = np.minimum(diff2, np.abs(img2[:, 1:] - img2[:, :-1]))
    if mode == "sym-iso":
        tv = (
            np.sum(np.sqrt(diff1[:, 1:] ** 2 + diff2[1:, :] ** 2))
            + np.sum(np.sqrt(diff1[:, 1:] ** 2 + diff2[:-1, :] ** 2))
            + np.sum(np.sqrt(diff1[:, :-1] ** 2 + diff2[:-1, :] ** 2))
            + np.sum(np.sqrt(diff1[:, :-1] ** 2 + diff2[1:, :] ** 2))
        )
        return tv / 4
    elif mode == "non-sym-iso":
        return np.sum(np.sqrt(diff1[:, :-1] ** 2 + diff2[:-1, :] ** 2))
    else:
        return np.sum(diff1) + np.sum(diff2)
