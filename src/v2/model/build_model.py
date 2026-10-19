"""Build the articulated Gazebo model with dual 3D LiDARs from project STEP CAD.

Extracts analytic dimensions from TRACTOR.stp, constructs the 7-link 6-joint kinematic tree,
equips 2 directional 3D LiDARs (front, rear), and generates SDF 1.10.
"""

from __future__ import annotations

import json
import math
import tomllib
import xml.dom.minidom
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    from .geometry import extract_geometry
except ImportError:
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
    fn = filename if (filename.endswith(".so") or "plugin" in filename) else f"gz-sim-{filename}-system"
    entry = element(
        parent,
        "plugin",
        name=f"gz::sim::systems::{system}",
        filename=fn,
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
    vertical_samples: int = 1,
    vertical_min_angle: float = 0.0,
    vertical_max_angle: float = 0.0,
) -> None:
    sensor = element(parent, "sensor", name=name, type="gpu_lidar")
    element(sensor, "pose", pose)
    element(sensor, "topic", topic)
    element(sensor, "update_rate", update_rate)
    element(sensor, "always_on", "true")
    element(sensor, "visualize", "true")
    ray = element(sensor, "ray")
    scan = element(ray, "scan")
    horizontal = element(scan, "horizontal")
    element(horizontal, "samples", samples)
    element(horizontal, "resolution", 1)
    element(horizontal, "min_angle", min_angle)
    element(horizontal, "max_angle", max_angle)
    if vertical_samples > 1:
        vertical = element(scan, "vertical")
        element(vertical, "samples", vertical_samples)
        element(vertical, "resolution", 1)
        element(vertical, "min_angle", vertical_min_angle)
        element(vertical, "max_angle", vertical_max_angle)
    range_el = element(ray, "range")
    element(range_el, "min", min_range)
    element(range_el, "max", max_range)
    element(range_el, "resolution", 0.02)


def _scaled_geometry(g: dict, size_scale: float) -> dict:
    """Apply a uniform rigid-body scale factor over CAD ground-truth dimensions."""
    if size_scale == 1.0:
        return g
    scaled = dict(g)
    for key in ("wheelbase_m", "beam_width_m", "beam_height_m", "beam_z_m"):
        scaled[key] = float(g[key]) * size_scale
    for end in ("rear", "front"):
        axle = dict(g[end])
        for key in ("wheel_radius_m", "wheel_width_m", "track_m", "axle_radius_m", "axle_length_m", "x_m", "z_m"):
            axle[key] = float(axle[key]) * size_scale
        scaled[end] = axle
    return scaled


def create_model(
    g: dict,
    p: dict,
    model_name: str = "tractor",
    fast_lidar: bool = False,
) -> ET.Element:
    sdf = ET.Element("sdf", version="1.10")
    model = element(sdf, "model", name=model_name, canonical_link="chassis")
    element(model, "self_collide", "false")
    physics = p["physics"]
    size_scale = float(p.get("vehicle", {}).get("size_scale", 1.0))
    if not math.isfinite(size_scale) or size_scale <= 0:
        raise ValueError(f"vehicle.size_scale must be finite and positive, got {size_scale}")
    g = _scaled_geometry(g, size_scale)

    yellow = (0.85, 0.57, 0.09, 1)
    metal = (0.36, 0.39, 0.43, 1)
    rubber = (0.065, 0.07, 0.08, 1)
    hub = (0.72, 0.74, 0.76, 1)

    chassis = element(model, "link", name="chassis")

    # ================= LiDAR Sensor Configuration =================
    lidar_cfg = p.get("lidar", {})
    layout = lidar_cfg.get("layout", "dual_directional_3d")

    h_samples = 90 if fast_lidar else 180
    default_v_samples = 8 if fast_lidar else 16

    if layout == "dual_directional_3d":
        pitch_down = float(lidar_cfg.get("pitch_down_rad", 0.349066))
        v_samples = default_v_samples if fast_lidar else int(lidar_cfg.get("vertical_samples", 16))
        v_min = float(lidar_cfg.get("vertical_min_rad", -0.436332))
        v_max = float(lidar_cfg.get("vertical_max_rad", 0.261799))
        front_x = float(lidar_cfg.get("front_x_m", 1.45))
        front_y = float(lidar_cfg.get("front_y_m", 0.00))
        front_z = float(lidar_cfg.get("front_z_m", 0.60))
        rear_x = float(lidar_cfg.get("rear_x_m", -1.45))
        rear_y = float(lidar_cfg.get("rear_y_m", 0.00))
        rear_z = float(lidar_cfg.get("rear_z_m", 0.60))

        # Pod 0: Front Directional 3D LiDAR (mounted on front bumper, tilted down towards ground)
        add_gpu_lidar(
            chassis,
            name="lidar_front",
            pose=(front_x, front_y, front_z, 0, pitch_down, 0),
            topic="/tractor/lidar/front",
            samples=h_samples,
            min_angle=-1.047198,
            max_angle=1.047198,
            vertical_samples=v_samples,
            vertical_min_angle=v_min,
            vertical_max_angle=v_max,
        )

        # Pod 1: Rear Directional 3D LiDAR (mounted on rear bumper/hitch, tilted down towards ground)
        add_gpu_lidar(
            chassis,
            name="lidar_rear",
            pose=(rear_x, rear_y, rear_z, 0, pitch_down, math.pi),
            topic="/tractor/lidar/rear",
            samples=h_samples,
            min_angle=-1.047198,
            max_angle=1.047198,
            vertical_samples=v_samples,
            vertical_min_angle=v_min,
            vertical_max_angle=v_max,
        )
    else:
        # Fallback: 4 Distributed LiDAR Pods (Horizontal)
        add_gpu_lidar(
            chassis,
            name="lidar_front",
            pose=(g["front"]["x_m"] + 0.05, 0.0, g["beam_z_m"] + 0.15, 0, 0, 0),
            topic="/tractor/lidar/front",
            samples=180,
            min_angle=-1.047198,
            max_angle=1.047198,
        )
        add_gpu_lidar(
            chassis,
            name="lidar_rear",
            pose=(g["rear"]["x_m"] - 0.05, 0.0, g["beam_z_m"] + 0.15, 0, 0, math.pi),
            topic="/tractor/lidar/rear",
            samples=180,
            min_angle=-1.047198,
            max_angle=1.047198,
        )
        add_gpu_lidar(
            chassis,
            name="lidar_right",
            pose=(0.50, -g["rear"]["track_m"] / 2.0, g["beam_z_m"] + 0.20, 0, 0, -math.pi / 2.0),
            topic="/tractor/lidar/right",
            samples=150,
            min_angle=-0.872665,
            max_angle=0.872665,
        )
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
    length = g["wheelbase_m"] + 0.22 * size_scale
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

    # Chassis Body Collision Boundary (uniform vehicle scale over CAD footprint)
    chassis_coll = element(chassis, "collision", name="chassis_body_collision")
    element(chassis_coll, "pose", (0.0, 0, 0.55 * size_scale, 0, 0, 0))
    box(chassis_coll, (2.90 * size_scale, 1.72 * size_scale, 0.40 * size_scale))
    surf = element(chassis_coll, "surface")
    ode_fr = element(element(surf, "friction"), "ode")
    element(ode_fr, "mu", 1.0)
    element(ode_fr, "mu2", 1.0)

    # Tractor Physics Contact Sensor for Ground-Truth Collision Detection
    contact_sensor = element(chassis, "sensor", name="tractor_contact_sensor", type="contact")
    element(contact_sensor, "always_on", "true")
    element(contact_sensor, "update_rate", 30)
    element(contact_sensor, "topic", "/model/tractor/contacts")
    contact_el = element(contact_sensor, "contact")
    element(contact_el, "collision", "connecting_beam_collision")
    element(contact_el, "collision", "chassis_body_collision")

    # Chassis IMU Sensor for native dynamic linear acceleration and angular velocity
    imu_sensor = element(chassis, "sensor", name="chassis_imu", type="imu")
    element(imu_sensor, "always_on", "true")
    element(imu_sensor, "update_rate", 50)
    element(imu_sensor, "topic", "/model/tractor/imu")

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

    # Actuation: Unified Ackermann steering and traction plugin (GzDRL in-process control)
    plugin(
        model,
        "AckermannSteering",
        "ackermann_plugin",
        {
            "steering_only": "false",
            "left_steering_joint": "front_left_steering",
            "right_steering_joint": "front_right_steering",
            "left_joint": "rear_left_spin",
            "right_joint": "rear_right_spin",
            "wheel_base": g["wheelbase_m"],
            "wheel_separation": g["rear"]["track_m"],
            "kingpin_width": g["front"]["axle_length_m"],
            "wheel_radius": g["rear"]["wheel_radius_m"],
            "steering_limit": p["control"]["max_steering_rad"],
            "steer_p_gain": 10.0,
            "min_velocity": -5.0,
            "max_velocity": 5.0,
            "min_acceleration": -10.0,
            "max_acceleration": 10.0,
            "topic": f"/model/{model_name}/steering_angle",
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
    for sec_name, section in p.items():
        if sec_name in ("physics", "control"):
            for key, value in section.items():
                if (
                    not isinstance(value, (float, int))
                    or not math.isfinite(value)
                    or value <= 0
                ):
                    raise ValueError(f"{key} must be finite and positive")
        elif sec_name == "lidar":
            layout = section.get("layout", "dual_directional_3d")
            if layout not in ("dual_directional_3d", "surround_4pod_2d"):
                raise ValueError(f"Unknown lidar layout: {layout}")
            for key, value in section.items():
                if key != "layout" and (
                    not isinstance(value, (float, int)) or not math.isfinite(value)
                ):
                    raise ValueError(f"{key} must be finite numeric")
    return p


def main() -> None:
    cad_path = ROOT / "TRACTOR.stp"
    if not cad_path.exists():
        raise FileNotFoundError(f"CAD file not found at {cad_path}")

    g = extract_geometry(cad_path)
    p = load_parameters()

    # 1. Standard model (180x16 LiDAR)
    sdf_tree = create_model(g, p, model_name="tractor", fast_lidar=False)
    target_sdf = ROOT / "tractor.sdf"
    write_xml(target_sdf, sdf_tree)

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
        "CAD-derived articulated tractor with dual directional 3D LiDAR (180x16).",
    )
    write_xml(model_dir / "model.config", config_tree)

    # 2. Fast model (90x8 LiDAR for rapid RL rollouts)
    sdf_fast_tree = create_model(g, p, model_name="tractor", fast_lidar=True)
    target_fast_sdf = ROOT / "tractor_fast.sdf"
    write_xml(target_fast_sdf, sdf_fast_tree)

    model_fast_dir = ROOT / "tractor_fast"
    model_fast_dir.mkdir(parents=True, exist_ok=True)
    write_xml(model_fast_dir / "model.sdf", sdf_fast_tree)

    config_fast_tree = ET.Element("model")
    element(config_fast_tree, "name", "tractor_fast")
    element(config_fast_tree, "version", "1.0")
    element(config_fast_tree, "sdf", "model.sdf", version="1.10")
    element(
        config_fast_tree,
        "description",
        "CAD-derived articulated tractor with fast dual directional 3D LiDAR (90x8).",
    )
    write_xml(model_fast_dir / "model.config", config_fast_tree)

    target_json = ROOT / "cad_dimensions.json"
    target_json.write_text(
        json.dumps(g, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        f"Models generated:\n"
        f"  Standard (180x16): {target_sdf} and {model_dir / 'model.sdf'}\n"
        f"  Fast (90x8): {target_fast_sdf} and {model_fast_dir / 'model.sdf'}\n"
        f"  Wheelbase: {g['wheelbase_m']:.3f} m\n"
        f"  Rear wheel radius: {g['rear']['wheel_radius_m']:.3f} m (track: {g['rear']['track_m']:.3f} m)\n"
        f"  Front wheel radius: {g['front']['wheel_radius_m']:.3f} m (track: {g['front']['track_m']:.3f} m)\n"
    )


if __name__ == "__main__":
    main()
