"""Measured-length, provisional-layout round trip; all points use the rear axle.

The forward-only turnaround is a three-arc RLR manoeuvre (60/300/60 degrees),
returning to its entry position with opposite heading. It needs a large clear
area; it is not a claim that the real site has that space.
"""

from __future__ import annotations

import bisect
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

from obstacles import Observation, VehicleState

ROOT = Path(__file__).resolve().parent


def angle_error(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def load_route_config(geometry: dict, path: Path | None = None) -> dict:
    config = tomllib.loads((path or ROOT / "route.toml").read_text(encoding="utf-8"))
    for section in config.values():
        for name, value in section.items():
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"route.toml: {name} debe ser finito y positivo")
    layout, motion = config["layout"], config["motion"]
    minimum_radius = geometry["wheelbase_m"] / math.tan(0.4)
    if min(layout["curve_radius_m"], layout["turn_radius_m"]) < minimum_radius + 0.5:
        raise ValueError("Radio insuficiente para la dirección delantera del tractor")
    width = max(
        geometry[end]["track_m"] + geometry[end]["wheel_width_m"]
        for end in ("rear", "front")
    )
    if layout["gate_width_m"] < width + 2 * motion["obstacle_clearance_m"] + 0.4:
        raise ValueError(
            "La puerta no permite el margen configurado y la tolerancia lateral"
        )
    if layout["gate_after_curve_m"] >= layout["after_curve_m"]:
        raise ValueError("La puerta debe estar dentro del tramo posterior a la curva")
    if layout["turn_staging_m"] < layout["turn_radius_m"] + 4:
        raise ValueError("Falta espacio entre la puerta y la maniobra de retorno")
    if max(motion[key] for key in ("cruise_mps", "curve_mps", "gate_mps")) > 2:
        raise ValueError("La simulación admite velocidades de hasta 2 m/s")
    return config


@dataclass(frozen=True)
class PathPoint:
    s: float
    x: float
    y: float
    yaw: float
    curvature: float
    phase: str


class RoutePath:
    def __init__(self, geometry: dict, config: dict, spacing: float = 0.15):
        self.geometry, self.config = geometry, config
        self.spacing = spacing
        self.points = [PathPoint(0, geometry["rear"]["x_m"], 0, 0, 0, "IDA_RECTA")]
        layout = config["layout"]
        self.add(layout["straight_m"], 0, "IDA_RECTA")
        radius = layout["curve_radius_m"]
        self.add(math.pi * radius / 2, 1 / radius, "IDA_CURVA")
        self.gate_x = self.points[-1].x
        self.gate_y = self.points[-1].y + layout["gate_after_curve_m"]
        self.gate_out_s = self.points[-1].s + layout["gate_after_curve_m"]
        self.add(layout["after_curve_m"] + layout["turn_staging_m"], 0, "IDA_PUERTA")
        self.turn_start_s = self.points[-1].s
        radius = layout["turn_radius_m"]
        for angle, sign in ((math.pi / 3, -1), (5 * math.pi / 3, 1), (math.pi / 3, -1)):
            self.add(angle * radius, sign / radius, "GIRO_RETORNO")
        self.turn_end_s = self.points[-1].s
        self.gate_return_s = (
            self.turn_end_s
            + layout["after_curve_m"]
            + layout["turn_staging_m"]
            - layout["gate_after_curve_m"]
        )
        self.add(
            layout["after_curve_m"] + layout["turn_staging_m"], 0, "REGRESO_PUERTA"
        )
        radius = layout["curve_radius_m"]
        self.add(math.pi * radius / 2, -1 / radius, "REGRESO_CURVA")
        # Opposite heading: rear axle must end at +1 m so chassis centre returns to 0.
        self.add(layout["straight_m"] + 2 * geometry["rear"]["x_m"], 0, "REGRESO_RECTA")
        self.distances = [point.s for point in self.points]
        self.length = self.points[-1].s

    def add(self, length: float, curvature: float, phase: str):
        start = self.points[-1]
        count = math.ceil(length / self.spacing)
        for index in range(1, count + 1):
            distance = length * index / count
            yaw = start.yaw + curvature * distance
            if abs(curvature) < 1e-9:
                x = start.x + distance * math.cos(start.yaw)
                y = start.y + distance * math.sin(start.yaw)
            else:
                x = start.x + (math.sin(yaw) - math.sin(start.yaw)) / curvature
                y = start.y - (math.cos(yaw) - math.cos(start.yaw)) / curvature
            self.points.append(
                PathPoint(start.s + distance, x, y, yaw, curvature, phase)
            )

    def at(self, distance: float) -> PathPoint:
        return self.points[
            min(bisect.bisect_left(self.distances, distance), len(self.points) - 1)
        ]

    def rear_position(self, vehicle: VehicleState) -> tuple[float, float]:
        offset = self.geometry["rear"]["x_m"]
        return vehicle.x + offset * math.cos(
            vehicle.yaw
        ), vehicle.y + offset * math.sin(vehicle.yaw)

    def project(self, vehicle: VehicleState, progress: float) -> tuple[float, float]:
        """Local, monotonic projection avoids skipping across the turnaround loops."""
        x, y = self.rear_position(vehicle)
        first = max(0, bisect.bisect_left(self.distances, progress - 0.5) - 1)
        last = min(
            len(self.points) - 1, bisect.bisect_left(self.distances, progress + 5)
        )
        best = (float("inf"), progress)
        for index in range(first, last):
            a, b = self.points[index : index + 2]
            dx, dy = b.x - a.x, b.y - a.y
            fraction = max(
                0.0, min(1.0, ((x - a.x) * dx + (y - a.y) * dy) / (dx * dx + dy * dy))
            )
            error = math.hypot(x - a.x - fraction * dx, y - a.y - fraction * dy)
            if error < best[0]:
                best = error, a.s + fraction * (b.s - a.s)
        return max(progress, best[1]), best[0]

    def command(self, vehicle: VehicleState, progress: float) -> tuple[float, float]:
        motion = self.config["motion"]
        x, y = self.rear_position(vehicle)
        target = self.at(progress + motion["lookahead_m"])
        dx, dy = target.x - x, target.y - y
        local_y = -math.sin(vehicle.yaw) * dx + math.cos(vehicle.yaw) * dy
        steering = math.atan2(
            2 * self.geometry["wheelbase_m"] * local_y, max(0.01, dx * dx + dy * dy)
        )
        upcoming = (self.at(progress + offset) for offset in (0, 1, 2, 3, 4))
        speed = (
            motion["curve_mps"]
            if any(abs(point.curvature) > 0.01 for point in upcoming)
            else motion["cruise_mps"]
        )
        if min(abs(progress - self.gate_out_s), abs(progress - self.gate_return_s)) < 5:
            speed = min(speed, motion["gate_mps"])
        remaining = self.length - progress
        speed = min(speed, max(0.05, remaining * 0.7))
        return speed, max(-0.4, min(0.4, steering))

    def crossers(self) -> dict:
        specs = {}
        for name, label, x, start, end, speed, phase, radius in (
            ("person_out", "persona", 30, 3.8, -3.8, 0.7, "IDA_RECTA", 0.36),
            ("dog_out", "perro", 85, -3.8, 3.8, 1.0, "IDA_RECTA", 0.65),
            ("person_return", "persona", 70, -3.8, 3.8, 0.7, "REGRESO_RECTA", 0.36),
            ("dog_return", "perro", 20, 3.8, -3.8, 1.0, "REGRESO_RECTA", 0.65),
        ):
            location = min(
                (p for p in self.points if p.phase == phase), key=lambda p: abs(p.x - x)
            )
            specs[name] = dict(
                label=label,
                x=x,
                start_y=start,
                end_y=end,
                speed=speed,
                trigger_s=location.s - 6,
                radius=radius,
            )
        return specs

    def gate_observations(self) -> list[Observation]:
        """Circles conservatively cover the solid side panels and posts."""
        half = self.config["layout"]["gate_width_m"] / 2
        return [
            Observation(
                f"gate_{side}_{index}",
                "puerta",
                self.gate_x + side * (half + 0.18 + index * 0.3),
                self.gate_y,
                0.19,
            )
            for side in (-1, 1)
            for index in range(21)
        ]
