"""Simulation-only keyboard state and Ackermann-to-rear-axle conversion."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def approach(value: float, target: float, change: float) -> float:
    return value + max(-change, min(change, target - value))


@dataclass
class KeyboardDrive:
    wheelbase: float
    forward_speed: float
    reverse_speed: float
    max_steering: float
    steering_rate: float
    acceleration: float
    keys: set[str] = field(default_factory=set)
    speed: float = 0.0
    steering: float = 0.0

    def stop(self) -> None:
        self.keys.clear()
        self.speed = 0.0
        self.steering = 0.0

    def step(self, dt: float) -> tuple[float, float, float]:
        """Return speed [m/s], virtual steering [rad], rear-axle yaw rate [rad/s]."""
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        dt = min(dt, 0.1)  # No command jump after a paused/blocked UI.
        forward = bool(self.keys & {"w", "up"})
        reverse = bool(self.keys & {"s", "down"})
        left = bool(self.keys & {"a", "left"})
        right = bool(self.keys & {"d", "right"})
        target_speed = 0.0
        if forward != reverse:
            target_speed = self.forward_speed if forward else -self.reverse_speed
        target_steering = (int(left) - int(right)) * self.max_steering
        self.speed = approach(self.speed, target_speed, self.acceleration * dt)
        self.steering = approach(
            self.steering, target_steering, self.steering_rate * dt
        )
        # Rear wheels emulate a mechanical differential. No rotation in place.
        yaw_rate = self.speed * math.tan(self.steering) / self.wheelbase
        return self.speed, self.steering, yaw_rate
