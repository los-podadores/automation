"""Conservative, simulation-only stop/wait/resume decisions.

The input is labelled geometric observations, not camera detections. Gazebo
ground truth supplies these observations in the first demonstration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# Synthetic crossing scenario, independent of the unmeasured real gate.
CROSSERS = {
    "person_crossing": {
        "label": "persona",
        "x": 4.0,
        "start_y": 2.8,
        "end_y": -2.8,
        "speed": 0.7,
        "trigger_x": 0.5,
        "radius": 0.36,
    },
    "dog_crossing": {
        "label": "perro",
        "x": 8.0,
        "start_y": -2.8,
        "end_y": 2.8,
        "speed": 1.0,
        "trigger_x": 4.5,
        "radius": 0.65,
    },
}


@dataclass(frozen=True)
class VehicleState:
    x: float
    y: float
    yaw: float
    speed: float
    sim_time: float


@dataclass(frozen=True)
class Observation:
    name: str
    label: str
    x: float
    y: float
    radius: float
    vx: float = 0.0
    vy: float = 0.0


@dataclass(frozen=True)
class Decision:
    allowed: bool
    state: str
    reason: str
    obstacle: str | None = None


def circle_rectangle_gap(
    x: float,
    y: float,
    radius: float,
    vehicle: VehicleState,
    front: float,
    rear: float,
    half_width: float,
) -> float:
    """Conservative clearance from a circular obstacle to the full tractor box."""
    dx, dy = x - vehicle.x, y - vehicle.y
    along = dx * math.cos(vehicle.yaw) + dy * math.sin(vehicle.yaw)
    across = -dx * math.sin(vehicle.yaw) + dy * math.cos(vehicle.yaw)
    outside_x = max(-rear - along, 0.0, along - front)
    outside_y = max(abs(across) - half_width, 0.0)
    return math.hypot(outside_x, outside_y) - radius


class ObstacleGuard:
    def __init__(
        self, geometry: dict, clearance: float = 0.65, clear_hold: float = 0.8
    ) -> None:
        self.wheelbase = geometry["wheelbase_m"]
        self.rear_x = geometry["rear"]["x_m"]
        self.front = geometry["front"]["x_m"] + geometry["front"]["wheel_radius_m"]
        self.rear = -geometry["rear"]["x_m"] + geometry["rear"]["wheel_radius_m"]
        self.half_width = max(
            (geometry[end]["track_m"] + geometry[end]["wheel_width_m"]) / 2
            for end in ("front", "rear")
        )
        self.clearance = clearance
        self.clear_hold = clear_hold
        self.clear_since: float | None = None
        self.waiting = False

    def decide(
        self,
        vehicle: VehicleState | None,
        observations: list[Observation],
        speed: float,
        steering: float,
        now: float,
        fresh: bool,
    ) -> Decision:
        if vehicle is None or not fresh:
            self.waiting = True
            self.clear_since = None
            return Decision(
                False, "SIN DATOS", "Esperando percepción simulada actualizada"
            )
        values = (
            vehicle.x,
            vehicle.y,
            vehicle.yaw,
            vehicle.speed,
            speed,
            steering,
            now,
        )
        if not all(math.isfinite(v) for v in values):
            self.waiting = True
            self.clear_since = None
            return Decision(False, "SIN DATOS", "Estado de movimiento inválido")
        direction = -1 if speed < 0 or (speed == 0 and vehicle.speed < -0.01) else 1
        # Extra distance accounts for reaction and provisional braking at low speed.
        v = max(abs(speed), abs(vehicle.speed))
        horizon = 1.6 + 0.8 * v + v * v / (2 * 0.6)
        curvature = math.tan(steering) / self.wheelbase
        rear_x = vehicle.x + self.rear_x * math.cos(vehicle.yaw)
        rear_y = vehicle.y + self.rear_x * math.sin(vehicle.yaw)
        for obs in observations:
            if (
                not all(
                    math.isfinite(value)
                    for value in (obs.x, obs.y, obs.radius, obs.vx, obs.vy)
                )
                or obs.radius < 0
            ):
                self.waiting = True
                self.clear_since = None
                return Decision(False, "SIN DATOS", "Observación inválida")
            for sample in range(17):
                distance = direction * horizon * sample / 16
                yaw = vehicle.yaw + curvature * distance
                if abs(curvature) < 1e-8:
                    rx = rear_x + distance * math.cos(vehicle.yaw)
                    ry = rear_y + distance * math.sin(vehicle.yaw)
                else:
                    rx = rear_x + (math.sin(yaw) - math.sin(vehicle.yaw)) / curvature
                    ry = rear_y - (math.cos(yaw) - math.cos(vehicle.yaw)) / curvature
                projected = VehicleState(
                    rx - self.rear_x * math.cos(yaw),
                    ry - self.rear_x * math.sin(yaw),
                    yaw,
                    speed,
                    vehicle.sim_time,
                )
                # Check both current occupancy and constant-velocity crossing.
                # Requested cruising speed is retained while stopped, so a
                # crossing is not declared clear merely because v became zero.
                predicted_time = abs(distance) / max(v, 0.1)
                if (
                    min(
                        circle_rectangle_gap(
                            obs.x + obs.vx * time_ahead,
                            obs.y + obs.vy * time_ahead,
                            obs.radius,
                            projected,
                            self.front,
                            self.rear,
                            self.half_width,
                        )
                        for time_ahead in (0.0, predicted_time)
                    )
                    < self.clearance
                ):
                    self.waiting = True
                    self.clear_since = None
                    state = "FRENAR" if abs(vehicle.speed) > 0.03 else "ESPERAR"
                    return Decision(
                        False,
                        state,
                        f"{obs.label.capitalize()} en la trayectoria",
                        obs.name,
                    )
        if self.waiting:
            if self.clear_since is None:
                self.clear_since = now
            if now - self.clear_since < self.clear_hold:
                return Decision(False, "ESPERAR", "Comprobando que el paso siga libre")
        self.waiting = False
        return Decision(True, "CONTINUAR", "Trayectoria libre")


def straight_path_steering(vehicle: VehicleState, wheelbase: float) -> float:
    """Pure-pursuit reference for the short test lane y=0 (not the 110 m route)."""
    rear_y = vehicle.y - wheelbase / 2 * math.sin(vehicle.yaw)
    along = 3.0
    across = -rear_y
    local_y = -math.sin(vehicle.yaw) * along + math.cos(vehicle.yaw) * across
    delta = math.atan2(2 * wheelbase * local_y, along**2 + across**2)
    return max(-0.4, min(0.4, delta))
