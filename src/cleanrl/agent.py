"""Agent architecture for RPO on the RobotCoverageEnv Dict observation."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    if hasattr(layer, "weight") and layer.weight is not None:
        nn.init.orthogonal_(layer.weight, std)
    if hasattr(layer, "bias") and layer.bias is not None:
        nn.init.constant_(layer.bias, bias_const)
    return layer


class NatureCNN(nn.Module):
    """Standard NatureCNN adapted for 3x64x64 input -> 64x4x4 feature map."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.feature_dim = 64 * 4 * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x / 255.0)


class SensorMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(in_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, out_dim)),
            nn.Tanh(),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Agent(nn.Module):
    def __init__(self, sensor_dim: int, action_dim: int, rpo_alpha: float = 0.5):
        super().__init__()
        self.rpo_alpha = rpo_alpha

        self.cnn = NatureCNN(in_channels=3)
        self.sensor = SensorMLP(in_dim=sensor_dim, hidden_dim=64, out_dim=64)

        fused_dim = self.cnn.feature_dim + self.sensor.out_dim

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(fused_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        self.critic = nn.Sequential(
            layer_init(nn.Linear(fused_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def _fused(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        visual = obs_dict["visual"].float()
        sensors = obs_dict["sensors"].float()
        return torch.cat([self.cnn(visual), self.sensor(sensors)], dim=-1)

    def get_value(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.critic(self._fused(obs_dict))

    def get_action_and_value(
        self,
        obs_dict: dict[str, torch.Tensor],
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fused = self._fused(obs_dict)
        action_mean = self.actor_mean(fused)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        else:
            z = torch.empty_like(action_mean).uniform_(-self.rpo_alpha, self.rpo_alpha)
            action_mean = action_mean + z
            probs = Normal(action_mean, action_std)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(fused)
