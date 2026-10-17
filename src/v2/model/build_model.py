"""Build the articulated Gazebo model with 4-pod LiDAR array from project STEP CAD.

Extracts analytic dimensions from TRACTOR.stp, constructs the 7-link 6-joint kinematic tree,
equips 4 distributed LiDAR pods (360-degree surround coverage), and generates SDF 1.10.
"""

from __future__ import annotations

import json
import math
import tomllib
import xml.dom.minidom
import xml.etree.ElementTree as ET
from pathlib import Path

from geometry import extract_geometry

ROOT = Path(__file__).resolve().parent


def element(parent: ET.Element, tag: str, text=None, **attrs) -> ET.Element:
    child = ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})
    if text is not None:
        if isinstance(text, (tuple, list)):
            text = " ".join(f"{value:.10g}" for value in text)
        child.text = str(text)
    return child


def material(parent: ET.Element, color: tuple) -> None:
    mat = element(parent, "material")
    element(mat, "ambient", color)
    element(mat, "diffuse", color)


def box(parent: ET.Element, size: tuple) -> None:
    element(element(parent, "geometry"), "box")
    element(parent.find("geometry/box"), "size", size)


def cylinder(parent: ET.Element, radius: float, length: float) -> None:
    shape = element(element(parent, "geometry"), "cylinder")
    element(shape, "radius", radius)
    element(shape, "length", length)


def sphere(parent: ET.Element, radius: float) -> None:
    element(element(element(parent, "geometry"), "sphere"), "radius", radius)


def inertia(link: ET.Element, mass: float, diagonal: tuple, pose=None) -> None:
    inertial = element(link, "inertial")
    if pose is not None:
        element(inertial, "pose", pose)
    element(inertial, "mass", mass)
    tensor = element(inertial, "inertia")
    for tag, value in zip(("ixx", "iyy", "izz"), diagonal, strict=True):
        element(tensor, tag, value)
    for tag in ("ixy", "ixz", "iyz"):
        element(tensor, tag, 0)


def shape_pair(
    link: ET.Element,
    name: str,
    shape: str,
    dimensions: tuple,
    pose: tuple,
    color: tuple,
    friction: float = 2.0,
    friction_lateral: float | None = None,
) -> None:
    for tag in ("visual", "collision"):
        part = element(link, tag, name=f"{name}_{tag}")
        element(part, "pose", pose)
        if shape == "box":
            box(part, dimensions)
        elif shape == "sphere":
            sphere(part, *dimensions)
        else:
            cylinder(part, *dimensions)
        if tag == "visual":
            material(part, color)
        else:
            surface = element(part, "surface")
            ode = element(element(surface, "friction"), "ode")
            element(ode, "mu", friction)
            element(ode, "mu2", friction_lateral if friction_lateral is not None else friction)
            if "tire" in name:
                element(ode, "fdir1", (1, 0, 0))
                element(ode, "slip1", 0.0)
                element(ode, "slip2", 0.0)
                contact = element(element(surface, "contact"), "ode")
                element(contact, "kp", 1e7)
                element(contact, "kd", 1.0)
                element(contact, "min_depth", 0.005)


def visual_only(
    link: ET.Element,
    name: str,
    shape: str,
    dimensions: tuple,
    pose: tuple,
    color: tuple,
) -> None:
    part = element(link, "visual", name=name)
    element(part, "pose", pose)
    if shape == "box":
        box(part, dimensions)
    elif shape == "sphere":
        sphere(part, *dimensions)
    else:
        cylinder(part, *dimensions)
    material(part, color)


def revolute(
    model: ET.Element,
    name: str,
    parent: str,
    child: str,
    axis: tuple,
    steering: bool = False,
) -> None:
    joint = element(model, "joint", name=name, type="revolute")
    element(joint, "parent", parent)
    element(joint, "child", child)
    axis_el = element(joint, "axis")
    element(axis_el, "xyz", axis)
    limit = element(axis_el, "limit")
    if steering:
        element(limit, "lower", -0.4)
        element(limit, "upper", 0.4)
        element(limit, "effort", 800)
        element(limit, "velocity", 1.5)
    else:
        element(limit, "lower", -1e16)
        element(limit, "upper", 1e16)
        element(limit, "effort", 1000)
        element(limit, "velocity", 30)
    dynamics = element(axis_el, "dynamics")
    element(dynamics, "damping", 0.1)


def plugin(parent: ET.Element, system: str, filename: str, values: dict) -> None:
    entry = element(
        parent,
        "plugin",
        name=f"gz::sim::systems::{system}",
        filename=f"gz-sim-{filename}-system",
    )
    for key, value in values.items():
        element(entry, key, value)


def add_gpu_lidar(
    parent: ET.Element,
    name: str,
    pose: tuple,
    topic: str,
    samples: int,
    min_angle: float,
    max_angle: float,
    min_range: float = 0.15,
    max_range: float = 25.0,
    update_rate: int = 20,
) -> None:
    sensor = element(parent, "sensor", name=name, type="gpu_lidar")
    element(sensor, "pose", pose)
    element(sensor, "topic", topic)
    element(sensor, "update_rate", update_rate)
    element(sensor, "always_on", "true")
    element(sensor, "visualize", "false")
    ray = element(sensor, "ray")
    scan = element(ray, "scan")
    horizontal = element(scan, "horizontal")
    element(horizontal, "samples", samples)
    element(horizontal, "resolution", 1)
    element(horizontal, "min_angle", min_angle)
    element(horizontal, "max_angle", max_angle)
    range_el = element(ray, "range")
    element(range_el, "min", min_range)
    element(range_el, "max", max_range)
    element(range_el, "resolution", 0.02)


def create_model(g: dict, p: dict) -> ET.Element:
    sdf = ET.Element("sdf", version="1.10")
    model = element(sdf, "model", name="tractor", canonical_link="chassis")
    element(model, "self_collide", "false")
    physics = p["physics"]

    yellow = (0.85, 0.57, 0.09, 1)
    metal = (0.36, 0.39, 0.43, 1)
    rubber = (0.065, 0.07, 0.08, 1)
    hub = (0.72, 0.74, 0.76, 1)

    chassis = element(model, "link", name="chassis")

    # ================= 4 Distributed LiDAR Pods =================
    # Pod 0: Front-Center (0 deg, 120 deg FOV, 180 samples)
    add_gpu_lidar(
        chassis,
        name="lidar_front",
        pose=(g["front"]["x_m"] + 0.05, 0.0, g["beam_z_m"] + 0.15, 0, 0, 0),
        topic="/tractor/lidar/front",
        samples=180,
        min_angle=-1.047198,
        max_angle=1.047198,
    )

    # Pod 1: Rear-Center (180 deg, 120 deg FOV, 180 samples)
    add_gpu_lidar(
        chassis,
        name="lidar_rear",
        pose=(g["rear"]["x_m"] - 0.05, 0.0, g["beam_z_m"] + 0.15, 0, 0, math.pi),
        topic="/tractor/lidar/rear",
        samples=180,
        min_angle=-1.047198,
        max_angle=1.047198,
    )

    # Pod 2: Front-Right (-90 deg, 100 deg FOV, 150 samples)
    add_gpu_lidar(
        chassis,
        name="lidar_right",
        pose=(0.50, -g["rear"]["track_m"] / 2.0, g["beam_z_m"] + 0.20, 0, 0, -math.pi / 2.0),
        topic="/tractor/lidar/right",
        samples=150,
        min_angle=-0.872665,
        max_angle=0.872665,
    )

    # Pod 3: Front-Left (+90 deg, 100 deg FOV, 150 samples)
    add_gpu_lidar(
        chassis,
        name="lidar_left",
        pose=(0.50, g["rear"]["track_m"] / 2.0, g["beam_z_m"] + 0.20, 0, 0, math.pi / 2.0),
        topic="/tractor/lidar/left",
        samples=150,
        min_angle=-0.872665,
        max_angle=0.872665,
    )

    # Chassis Inertia and Structural Geometry
    length = g["wheelbase_m"] + 0.22
    width = g["rear"]["axle_length_m"]
    height = 0.25
    mass = physics["chassis_mass_kg"]
    inertia(
        chassis,
        mass,
        (
            mass * (width**2 + height**2) / 12,
            mass * (length**2 + height**2) / 12,
            mass * (length**2 + width**2) / 12,
        ),
        (0, 0, g["beam_z_m"], 0, 0, 0),
    )
    shape_pair(
        chassis,
        "connecting_beam",
        "box",
        (g["wheelbase_m"], g["beam_width_m"], g["beam_height_m"]),
        (0, 0, g["beam_z_m"], 0, 0, 0),
        yellow,
    )

    # Axles, Steering Knuckles, and Wheels
    for end in ("rear", "front"):
        a = g[end]
        visual_only(
            chassis,
            f"{end}_axle_visual",
            "cylinder",
            (a["axle_radius_m"], a["axle_length_m"]),
            (a["x_m"], 0, a["z_m"], -math.pi / 2, 0, 0),
            metal,
        )
        for side, sign in (("left", 1), ("right", -1)):
            name = f"{end}_{side}_wheel"
            parent = "chassis"
            if end == "front":
                parent = f"front_{side}_knuckle"
                knuckle = element(model, "link", name=parent)
                element(
                    knuckle,
                    "pose",
                    (a["x_m"], sign * a["axle_length_m"] / 2, a["z_m"], 0, 0, 0),
                )
                m = physics["knuckle_mass_kg"]
                inertia(knuckle, m, (m * 0.04**2 * 0.4,) * 3)
                revolute(
                    model,
                    f"front_{side}_steering",
                    "chassis",
                    parent,
                    (0, 0, 1),
                    steering=True,
                )
            wheel = element(model, "link", name=name)
            element(
                wheel, "pose", (a["x_m"], sign * a["track_m"] / 2, a["z_m"], 0, 0, 0)
            )
            m = physics[f"{end}_wheel_mass_kg"]
            r, w = a["wheel_radius_m"], a["wheel_width_m"]
            radial = m * (3 * r**2 + w**2) / 12
            inertia(wheel, m, (radial, m * r**2 / 2, radial))
            shape_pair(
                wheel,
                "tire",
                "cylinder",
                (r, w),
                (0, 0, 0, -math.pi / 2, 0, 0),
                rubber,
                physics["friction"],
                physics.get("friction_lateral", physics["friction"]),
            )
            # Hub and rotation marker for visual inspection
            vis_hub = element(wheel, "visual", name="hub")
            element(vis_hub, "pose", (0, 0, 0, -math.pi / 2, 0, 0))
            cylinder(vis_hub, r * 0.44, w + 0.006)
            material(vis_hub, hub)
            spoke = element(wheel, "visual", name="rotation_marker")
            box(spoke, (0.035, w + 0.01, r * 1.6))
            material(spoke, yellow)
            revolute(model, f"{end}_{side}_spin", parent, name, (0, 1, 0))

    rear_frame = element(model, "frame", name="rear_axle", attached_to="chassis")
    element(
        rear_frame,
        "pose",
        (g["rear"]["x_m"], 0, g["rear"]["z_m"], 0, 0, 0),
        relative_to="chassis",
    )

    # Actuation: Ackermann steering knuckle control + rear differential drive
    plugin(
        model,
        "AckermannSteering",
        "ackermann-steering",
        {
            "steering_only": "true",
            "left_steering_joint": "front_left_steering",
            "right_steering_joint": "front_right_steering",
            "wheel_base": g["wheelbase_m"],
            "wheel_separation": g["front"]["axle_length_m"],
            "steering_limit": p["control"]["max_steering_rad"],
            "steer_p_gain": 4.0,
            "topic": "/model/tractor/steering_angle",
        },
    )
    plugin(
        model,
        "DiffDrive",
        "diff-drive",
        {
            "left_joint": "rear_left_spin",
            "right_joint": "rear_right_spin",
            "wheel_separation": g["rear"]["track_m"],
            "wheel_radius": g["rear"]["wheel_radius_m"],
            "topic": "/model/tractor/rear_axle/cmd_vel",
            "odom_topic": "/model/tractor/wheel_odometry",
            "tf_topic": "/model/tractor/wheel_odometry_tf",
            "frame_id": "tractor/odom",
            "child_frame_id": "tractor/rear_axle",
            "odom_publish_frequency": 30,
            "max_linear_acceleration": 4.0,
            "min_linear_acceleration": -4.0,
            "max_angular_acceleration": 6.0,
            "min_angular_acceleration": -6.0,
        },
    )
    plugin(
        model,
        "OdometryPublisher",
        "odometry-publisher",
        {
            "odom_frame": "world",
            "robot_base_frame": "tractor/chassis",
            "odom_topic": "/model/tractor/odometry",
            "odom_publish_frequency": 30,
            "dimensions": 3,
        },
    )
    plugin(
        model,
        "JointStatePublisher",
        "joint-state-publisher",
        {
            "topic": "/model/tractor/joint_state",
        },
    )
    return sdf


def write_xml(path: Path, root: ET.Element) -> None:
    raw = ET.tostring(root, encoding="utf-8")
    pretty = xml.dom.minidom.parseString(raw).toprettyxml(indent="  ")
    # Clean up empty lines generated by toprettyxml
    cleaned = "\n".join(line for line in pretty.splitlines() if line.strip()) + "\n"
    path.write_text(cleaned, encoding="utf-8")


def load_parameters() -> dict:
    p = tomllib.loads((ROOT / "parameters.toml").read_text(encoding="utf-8"))
    for section in p.values():
        for key, value in section.items():
            if (
                not isinstance(value, (float, int))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{key} must be finite and positive")
    return p


def main() -> None:
    cad_path = ROOT / "TRACTOR.stp"
    if not cad_path.exists():
        raise FileNotFoundError(f"CAD file not found at {cad_path}")

    g = extract_geometry(cad_path)
    p = load_parameters()

    # Generate SDF and dimensions
    sdf_tree = create_model(g, p)
    target_sdf = ROOT / "tractor.sdf"
    write_xml(target_sdf, sdf_tree)

    # Also write to tractor/ directory for model://tractor resolution
    model_dir = ROOT / "tractor"
    model_dir.mkdir(parents=True, exist_ok=True)
    write_xml(model_dir / "model.sdf", sdf_tree)

    config_tree = ET.Element("model")
    element(config_tree, "name", "tractor")
    element(config_tree, "version", "1.0")
    element(config_tree, "sdf", "model.sdf", version="1.10")
    element(
        config_tree,
        "description",
        "CAD-derived articulated tractor with 4-pod LiDAR array.",
    )
    write_xml(model_dir / "model.config", config_tree)

    target_json = ROOT / "cad_dimensions.json"
    target_json.write_text(
        json.dumps(g, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        f"Model generated at {target_sdf} and {model_dir / 'model.sdf'}:\n"
        f"  Wheelbase: {g['wheelbase_m']:.3f} m\n"
        f"  Rear wheel radius: {g['rear']['wheel_radius_m']:.3f} m (track: {g['rear']['track_m']:.3f} m)\n"
        f"  Front wheel radius: {g['front']['wheel_radius_m']:.3f} m (track: {g['front']['track_m']:.3f} m)\n"
        f"  LiDAR sensors: 4 pods (front, rear, right, left) on /tractor/lidar/*"
    )


if __name__ == "__main__":
    main()
