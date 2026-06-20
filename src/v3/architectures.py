import gymnasium as gym
import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class StackedMapFeaturesExtractor(BaseFeaturesExtractor):
    """
    Scale-Grouped CNN (SGCNN) feature extractor for multi-scale map observations.

    Processes multi-scale coverage, obstacle, and frontier maps with grouped
    convolutions — each scale is convolved independently since spatial positions
    across scales do not correspond. Fuses map features with sensor features.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int,
        map_size: int,
        num_maps: int,
        sensor_dim: int = 11,
        num_map_types: int = 3,
    ):
        super().__init__(observation_space, features_dim=features_dim)

        in_channels = num_map_types * num_maps
        out_channels = 2 * in_channels

        out_spatial = map_size // 2 - 6
        out_size = out_spatial * out_spatial * out_channels

        self.map_extractor = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2,
                      padding=0, groups=num_maps),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                      padding=0, groups=num_maps),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                      padding=0, groups=num_maps),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                      padding=0, groups=num_maps),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(out_size, features_dim),
            nn.ReLU(),
        )

        self.sensor_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(sensor_dim, sensor_dim),
            nn.ReLU(),
        )

        self.fused_extractor = nn.Sequential(
            nn.Linear(features_dim + sensor_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict) -> th.Tensor:
        coverage = observations["coverage"]
        obstacles = observations["obstacles"]
        frontier = observations["frontier"]
        sensors = observations["sensors"]

        # cat gives (B, num_map_types*num_maps, W, H) with channels ordered as
        # [cov_s0, cov_s1, cov_s2, obs_s0, obs_s1, obs_s2, fro_s0, fro_s1, fro_s2]
        maps = th.cat([coverage, obstacles, frontier], dim=1)

        # Reorder so channels at the same scale are consecutive (grouped convs
        # require each group to see all map types at one spatial resolution):
        # [cov_s0, obs_s0, fro_s0, cov_s1, obs_s1, fro_s1, cov_s2, obs_s2, fro_s2]
        b, _, w, h = maps.shape
        num_maps = maps.shape[1] // 3
        maps = maps.reshape(b, 3, num_maps, w, h)
        maps = maps.permute(0, 2, 1, 3, 4)
        maps = maps.reshape(b, num_maps * 3, w, h)

        map_features = self.map_extractor(maps)
        sensor_features = self.sensor_extractor(sensors)

        fused = th.cat([map_features, sensor_features], dim=1)
        return self.fused_extractor(fused)
