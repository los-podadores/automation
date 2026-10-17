"""Extract the analytic dimensions of the supplied, single-solid STEP model.

This is intentionally a reader for this CAD's cylinders, not a general STEP
mesher. The articulated SDF reconstructs the wheels/axles as cylinders and the
connecting beam as a box. It does not claim an exact BREP-to-mesh conversion.
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path


def _refs(entity: str) -> list[int]:
    return [int(value) for value in re.findall(r"#(\d+)", entity)]


def _point(entity: str) -> tuple[float, float, float]:
    match = re.search(r"\(([-+\d.Ee]+),([-+\d.Ee]+),([-+\d.Ee]+)\)\)$", entity)
    if not match:
        raise ValueError(f"Unsupported STEP coordinate: {entity}")
    return tuple(float(value) for value in match.groups())


def extract_geometry(path: Path) -> dict:
    raw = path.read_bytes()
    entities = {
        int(key): re.sub(r"\s+", "", value)
        for key, value in re.findall(r"#(\d+)\s*=\s*(.*?);", raw.decode(), re.S)
    }
    if "SI_UNIT(.MILLI.,.METRE.)" not in entities.values():
        if not any("SI_UNIT(.MILLI.,.METRE.)" in e for e in entities.values()):
            raise ValueError("The supplied CAD must use millimetres.")

    stations: dict[float, dict] = {}
    for entity in entities.values():
        if not entity.startswith("CYLINDRICAL_SURFACE("):
            continue
        placement = entities[_refs(entity)[0]]
        point_id, axis_id, _ = _refs(placement)
        x, y, _z = _point(entities[point_id])
        axis = _point(entities[axis_id])
        if abs(axis[0]) > 1e-6 or abs(axis[1]) > 1e-6:
            raise ValueError("Expected the CAD axle direction to be STEP Z.")
        radius = float(entity.rsplit(",", 1)[1].rstrip(")"))
        station = stations.setdefault(x, {"height": y, "radii": set()})
        station["radii"].add(radius)
    if len(stations) != 2:
        raise ValueError("Expected exactly two axle stations in TRACTOR.stp.")

    axles = []
    for x, station in sorted(stations.items()):
        if len(station["radii"]) != 2:
            raise ValueError("Each axle must have an axle and wheel radius.")
        axle_radius, wheel_radius = sorted(station["radii"])
        caps: dict[float, set[float]] = {axle_radius: set(), wheel_radius: set()}
        for entity in entities.values():
            if not entity.startswith("CIRCLE("):
                continue
            radius = float(entity.rsplit(",", 1)[1].rstrip(")"))
            if radius not in caps:
                continue
            placement = entities[_refs(entity)[0]]
            cx, cy, cz = _point(entities[_refs(placement)[0]])
            if math.isclose(cx, x) and math.isclose(
                cy, station["height"], abs_tol=1e-6
            ):
                caps[radius].add(round(cz, 6))
        wheel_caps = sorted(caps[wheel_radius])
        if len(wheel_caps) != 4:
            raise ValueError("Expected four wheel cap coordinates per axle.")
        neg_width = wheel_caps[1] - wheel_caps[0]
        pos_width = wheel_caps[3] - wheel_caps[2]
        if not math.isclose(neg_width, pos_width, abs_tol=1e-4):
            raise ValueError("The two wheels on an axle must have equal width.")
        center = (wheel_caps[2] + wheel_caps[3]) / 2
        axles.append(
            {
                "cad_x_mm": x,
                "cad_height_mm": station["height"],
                "wheel_radius_m": wheel_radius / 1000,
                "wheel_width_m": pos_width / 1000,
                "track_m": 2 * center / 1000,
                "axle_radius_m": axle_radius / 1000,
                "axle_length_m": (max(caps[axle_radius]) - min(caps[axle_radius]))
                / 1000,
            }
        )
    rear, front = axles
    if rear["wheel_radius_m"] <= front["wheel_radius_m"]:
        raise ValueError("Expected the larger rear wheels at negative STEP X.")
    midpoint = (rear["cad_x_mm"] + front["cad_x_mm"]) / 2
    ground_mm = rear["cad_height_mm"] - 1000 * rear["wheel_radius_m"]
    for axle in axles:
        axle["x_m"] = (axle["cad_x_mm"] - midpoint) / 1000
        axle["z_m"] = (axle["cad_height_mm"] - ground_mm) / 1000

    small_caps = []
    for entity in entities.values():
        if entity.startswith("CIRCLE("):
            radius = float(entity.rsplit(",", 1)[1].rstrip(")")) / 1000
            if math.isclose(radius, rear["axle_radius_m"]):
                placement = entities[_refs(entity)[0]]
                z = abs(_point(entities[_refs(placement)[0]])[2]) / 1000
                if z > 1e-9:
                    small_caps.append(z)
    beam_width = 2 * min(small_caps)
    beam_heights = []
    for entity in entities.values():
        if entity.startswith("VERTEX_POINT("):
            _x, y, z = _point(entities[_refs(entity)[0]])
            if math.isclose(abs(z) / 1000, beam_width / 2, abs_tol=1e-6):
                beam_heights.append(y / 1000)
    if not beam_heights:
        raise ValueError("Cannot identify the connecting beam in this STEP.")
    return {
        "source": str(path),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_units": "mm",
        "axis_conversion": "Gazebo (x,y,z) = STEP (x,-z,y)/1000, centred between axles and raised to ground",
        "representation": "Articulated analytic reconstruction; beam simplified to a box; not exact CAD tessellation",
        "wheelbase_m": front["x_m"] - rear["x_m"],
        "beam_width_m": beam_width,
        "beam_height_m": max(beam_heights) - min(beam_heights),
        "beam_z_m": ((max(beam_heights) + min(beam_heights)) / 2 - ground_mm / 1000),
        "rear": rear,
        "front": front,
    }
