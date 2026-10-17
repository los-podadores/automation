"""Build the articulated Gazebo model from this project's analytic STEP CAD.

Run with Python 3.12+; no CAD kernel, ROS, training dependencies or GPU required.
"""

from __future__ import annotations

import json
import math
import tomllib
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
    friction: float = 0.9,
) -> None:
    for tag in ("visual", "collision"):
        # Gazebo's SDF validator requires unique names across child types.
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
            ode = element(element(element(part, "surface"), "friction"), "ode")
            element(ode, "mu", friction)
            element(ode, "mu2", friction)


def revolute(
    model: ET.Element, name: str, parent: str, child: str, axis: tuple, steering=False
) -> None:
    joint = element(model, "joint", name=name, type="revolute")
    element(joint, "parent", parent)
    element(joint, "child", child)
    # Default SDF joint frame is the child link frame.
    ax = element(joint, "axis")
    element(ax, "xyz", axis)
    limits = element(ax, "limit")
    element(limits, "lower", -0.6 if steering else -1e16)
    element(limits, "upper", 0.6 if steering else 1e16)
    element(limits, "effort", 1000)
    element(limits, "velocity", 0.8 if steering else 20)
    dynamics = element(ax, "dynamics")
    element(dynamics, "damping", 0.1)


def plugin(model: ET.Element, system: str, filename: str, values: dict) -> None:
    entry = element(
        model,
        "plugin",
        name=f"gz::sim::systems::{system}",
        filename=f"gz-sim-{filename}-system",
    )
    for key, value in values.items():
        element(entry, key, value)


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
    # Gazebo-frame camera: +X forward, +Y left, +Z up. Optional Sensors
    # system is enabled by the launcher; normal physics runs do not render it.
    sensor = element(chassis, "sensor", name="front_rgbd", type="rgbd_camera")
    element(sensor, "pose", (1.15, 0, 1.25, 0, 0.06, 0))
    element(sensor, "topic", "/tractor/front_camera")
    element(sensor, "update_rate", 10)
    element(sensor, "always_on", "true")
    camera = element(sensor, "camera")
    element(camera, "horizontal_fov", 1.4)
    image = element(camera, "image")
    element(image, "width", 640)
    element(image, "height", 360)
    element(image, "format", "R8G8B8")
    clip = element(camera, "clip")
    element(clip, "near", 0.15)
    element(clip, "far", 25)
    # Approximate lumped chassis inertia; actual mass distribution is unknown.
    length, width, height = g["wheelbase_m"] + 0.22, g["rear"]["axle_length_m"], 0.25
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
    for end in ("rear", "front"):
        a = g[end]
        shape_pair(
            chassis,
            f"{end}_axle",
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
            )
            # Raised hub and a spoke make wheel rotation visible.
            visual = element(wheel, "visual", name="hub")
            element(visual, "pose", (0, 0, 0, -math.pi / 2, 0, 0))
            cylinder(visual, r * 0.44, w + 0.006)
            material(visual, hub)
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

    # Steering and traction are independent so the front wheels can be steered
    # while stationary. Rear wheel speed difference emulates a differential;
    # teleop computes yaw = v*tan(delta)/L, never an independent pivot command.
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


def create_world() -> ET.Element:
    sdf = ET.Element("sdf", version="1.10")
    world = element(sdf, "world", name="tractor_field")
    physics = element(world, "physics", name="1ms", type="ignored")
    element(physics, "max_step_size", 0.001)
    element(physics, "real_time_factor", 1)
    element(world, "gravity", (0, 0, -9.81))
    for system, filename in (
        ("Physics", "physics"),
        ("UserCommands", "user-commands"),
        ("SceneBroadcaster", "scene-broadcaster"),
    ):
        plugin(world, system, filename, {})
    scene = element(world, "scene")
    element(scene, "ambient", (0.7, 0.7, 0.7, 1))
    element(scene, "background", (0.68, 0.8, 0.9, 1))
    sun = element(world, "light", name="sun", type="directional")
    element(sun, "pose", (0, 0, 10, 0, 0, 0))
    element(sun, "cast_shadows", "true")
    element(sun, "diffuse", (0.9, 0.9, 0.9, 1))
    element(sun, "specular", (0.2, 0.2, 0.2, 1))
    element(sun, "direction", (-0.5, 0.1, -0.9))
    floor = element(world, "model", name="ground")
    element(floor, "static", "true")
    ground = element(floor, "link", name="ground")
    shape_pair(
        ground,
        "field",
        "box",
        (40, 30, 0.2),
        (0, 0, -0.1, 0, 0, 0),
        (0.20, 0.36, 0.18, 1),
    )
    include = element(world, "include")
    # The launcher runs both validation and simulation from the worlds directory.
    # A local file works without Gazebo Sim's model:// resource callback.
    element(include, "uri", "../models/tractor/model.sdf")
    element(include, "pose", (0, 0, 0.03, 0, 0, 0))
    for name, pose, size in (
        ("obstacle_box", (6, 3, 0.5, 0, 0, 0), (1, 1, 1)),
        ("north_boundary", (0, 10, 0.25, 0, 0, 0), (30, 0.15, 0.5)),
        ("south_boundary", (0, -10, 0.25, 0, 0, 0), (30, 0.15, 0.5)),
        ("east_boundary", (15, 0, 0.25, 0, 0, 0), (0.15, 20, 0.5)),
        ("west_boundary", (-15, 0, 0.25, 0, 0, 0), (0.15, 20, 0.5)),
    ):
        obstacle = element(world, "model", name=name)
        element(obstacle, "static", "true")
        link = element(obstacle, "link", name="body")
        shape_pair(link, "body", "box", size, pose, (0.65, 0.37, 0.16, 1))
    for index, (x, y) in enumerate(((4, -3), (7, -3), (10, -3))):
        obstacle = element(world, "model", name=f"post_{index}")
        element(obstacle, "static", "true")
        link = element(obstacle, "link", name="body")
        shape_pair(
            link,
            "post",
            "cylinder",
            (0.18, 0.8),
            (x, y, 0.4, 0, 0, 0),
            (0.9, 0.3, 0.05, 1),
        )
    return sdf


def write_xml(path: Path, root: ET.Element) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    # Consistent newlines on Windows and Linux.
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write("\n")


def create_obstacle_world(crossers: dict | None = None) -> ET.Element:
    """Short crossing test with collidable, kinematic person/dog proxies."""
    from obstacles import CROSSERS

    sdf = create_world()
    world = sdf.find("world")
    world.set("name", "tractor_obstacles")
    skin = (0.65, 0.43, 0.29, 1)
    blue = (0.15, 0.32, 0.7, 1)
    dark = (0.15, 0.17, 0.2, 1)
    brown = (0.5, 0.28, 0.12, 1)
    for name, spec in (CROSSERS if crossers is None else crossers).items():
        actor = element(world, "model", name=name)
        yaw = math.copysign(math.pi / 2, spec["end_y"] - spec["start_y"])
        element(actor, "pose", (spec["x"], spec["start_y"], 0, 0, 0, yaw))
        link = element(actor, "link", name="body")
        element(link, "kinematic", "true")
        element(link, "gravity", "false")
        inertia(link, 70 if name.startswith("person") else 12, (7, 7, 1))
        if name.startswith("person"):
            parts = [
                ("torso", "box", (0.3, 0.38, 0.58), (0, 0, 1.05, 0, 0, 0), blue),
                ("head", "sphere", (0.14,), (0, 0, 1.52, 0, 0, 0), skin),
            ]
            for side in (-1, 1):
                parts.extend(
                    [
                        (
                            f"leg_{side}",
                            "cylinder",
                            (0.085, 0.74),
                            (0, side * 0.11, 0.37, 0, 0, 0),
                            dark,
                        ),
                        (
                            f"arm_{side}",
                            "cylinder",
                            (0.065, 0.55),
                            (0, side * 0.27, 1.03, 0, 0, 0),
                            blue,
                        ),
                    ]
                )
        else:
            parts = [
                ("torso", "box", (0.65, 0.28, 0.26), (0, 0, 0.39, 0, 0, 0), brown),
                ("head", "sphere", (0.13,), (0.36, 0, 0.53, 0, 0, 0), brown),
                ("snout", "box", (0.16, 0.15, 0.11), (0.49, 0, 0.5, 0, 0, 0), dark),
                ("tail", "cylinder", (0.035, 0.32), (-0.4, 0, 0.5, 0, -0.8, 0), brown),
            ]
            for x in (-0.22, 0.22):
                for y in (-0.11, 0.11):
                    parts.append(
                        (
                            f"leg_{x}_{y}",
                            "cylinder",
                            (0.045, 0.28),
                            (x, y, 0.14, 0, 0, 0),
                            brown,
                        )
                    )
        for part_name, shape, dims, pose, color in parts:
            shape_pair(link, part_name, shape, dims, pose, color)
        plugin(
            actor,
            "VelocityControl",
            "velocity-control",
            {
                "topic": f"/model/{name}/cmd_vel",
            },
        )
    return sdf


def create_route_world(
    geometry: dict, config: dict, time_scale: float = 1
) -> ET.Element:
    from route import RoutePath

    route = RoutePath(geometry, config)
    sdf = create_obstacle_world(route.crossers())
    world = sdf.find("world")
    world.set("name", "tractor_route")
    world.find("physics/real_time_factor").text = str(time_scale)
    gui = element(world, "gui", fullscreen="false")
    for filename, title in (
        ("MinimalScene", "3D View"),
        ("GzSceneManager", "Scene Manager"),
        ("InteractiveViewControl", "Interactive view control"),
        ("CameraTracking", "Camera Tracking"),
        ("EntityContextMenuPlugin", "Entity context menu"),
        ("SelectEntities", "Select Entities"),
        ("EntityTree", "Entity tree"),
        ("WorldControl", "World control"),
    ):
        entry = element(gui, "plugin", filename=filename, name=title)
        metadata = element(entry, "gz-gui")
        element(metadata, "title", title)
        element(metadata, "property", "false", type="bool", key="showTitleBar")
        element(
            metadata,
            "property",
            "docked" if filename in ("MinimalScene", "EntityTree") else "floating",
            type="string",
            key="state",
        )
        if filename == "MinimalScene":
            for tag, value in {
                "engine": "ogre2",
                "scene": "scene",
                "ambient_light": "0.7 0.7 0.7",
                "background_color": "0.68 0.8 0.9",
                "camera_pose": "-8 -6 5 0 0.45 0.64",
            }.items():
                element(entry, tag, value)
        elif filename == "CameraTracking":
            element(entry, "follow_target", "tractor")
            element(entry, "follow_offset", (-8, -6, 5))
            element(entry, "follow_pgain", 0.04)
        elif filename == "WorldControl":
            for tag, value in {
                "play_pause": "true",
                "step": "true",
                "start_paused": "false",
                "use_event": "true",
            }.items():
                element(entry, tag, value)
            anchors = element(metadata, "anchors", target="3D View")
            element(anchors, "line", own="left", target="left")
            element(anchors, "line", own="bottom", target="bottom")
            element(metadata, "property", 72, type="double", key="height")
        if filename not in ("MinimalScene", "EntityTree", "WorldControl"):
            element(metadata, "property", 5, type="double", key="width")
            element(metadata, "property", 5, type="double", key="height")
    for model in list(world.findall("model")):
        if model.attrib["name"] not in ("ground", *route.crossers()):
            world.remove(model)
    ground = world.find("model[@name='ground']/link")
    for geometry_box in ground.findall("*/geometry/box/size"):
        geometry_box.text = "180 100 0.2"
    for pose in ground.findall("*/pose"):
        pose.text = "60 15 -0.1 0 0 0"

    scenery = element(world, "model", name="road_markings")
    element(scenery, "static", "true")
    link = element(scenery, "link", name="road")
    # Visual-only pavement strips; the original floor supplies flat collisions.
    for index, point in enumerate(route.points[::7]):
        if point.phase.startswith("REGRESO"):
            continue
        visual = element(link, "visual", name=f"pavement_{index}")
        element(visual, "pose", (point.x, point.y, 0.004, 0, 0, point.yaw))
        box(visual, (1.15, config["layout"]["road_width_m"], 0.008))
        material(visual, (0.33, 0.34, 0.35, 1))
        marker = element(link, "visual", name=f"centre_{index}")
        element(marker, "pose", (point.x, point.y, 0.01, 0, 0, point.yaw))
        box(marker, (0.5, 0.055, 0.004))
        material(marker, (0.94, 0.82, 0.3, 1))

    def static_box(name, size, pose, color):
        model = element(world, "model", name=name)
        element(model, "static", "true")
        body = element(model, "link", name="body")
        shape_pair(body, name, "box", size, pose, color)
        return body

    width = config["layout"]["gate_width_m"]
    for side in (-1, 1):
        x = route.gate_x + side * (width / 2 + 0.15)
        static_box(
            f"gate_post_{side}",
            (0.3, 0.25, 2.5),
            (x, route.gate_y, 1.25, 0, 0, 0),
            (0.8, 0.63, 0.1, 1),
        )
        static_box(
            f"gate_fence_{side}",
            (6, 0.12, 1.5),
            (route.gate_x + side * (width / 2 + 3.3), route.gate_y, 0.75, 0, 0, 0),
            (0.34, 0.38, 0.4, 1),
        )
    static_box(
        "gate_lintel",
        (width + 0.6, 0.25, 0.2),
        (route.gate_x, route.gate_y, 2.6, 0, 0, 0),
        (0.8, 0.63, 0.1, 1),
    )

    # Stadium-side fence with gaps at crossing points, trees on the opposite verge.
    crossing_x = [spec["x"] for spec in route.crossers().values()]
    for index, x in enumerate(range(3, 105, 3)):
        if min(abs(x - value) for value in crossing_x) < 3:
            continue
        static_box(
            f"stadium_fence_{index}",
            (2.9, 0.1, 1.2),
            (x, 3.2, 0.6, 0, 0, 0),
            (0.38, 0.43, 0.4, 1),
        )
    for index, x in enumerate(range(8, 105, 12)):
        if min(abs(x - value) for value in crossing_x) < 2:
            continue
        tree = element(world, "model", name=f"tree_{index}")
        element(tree, "static", "true")
        body = element(tree, "link", name="body")
        shape_pair(
            body,
            "trunk",
            "cylinder",
            (0.18, 2.5),
            (x, -5.0, 1.25, 0, 0, 0),
            (0.32, 0.2, 0.1, 1),
        )
        shape_pair(
            body,
            "crown",
            "sphere",
            (1.25,),
            (x, -5.0, 3, 0, 0, 0),
            (0.14, 0.32, 0.1, 1),
        )
    for name, x, y, color in (
        ("start", 0, 0, (0.1, 0.7, 0.9, 1)),
        ("arrival", route.gate_x, route.gate_y + 5, (0.9, 0.5, 0.1, 1)),
    ):
        visual = element(link, "visual", name=name)
        element(visual, "pose", (x, y, 0.015, 0, 0, 0))
        box(visual, (0.15, 3.0, 0.005))
        material(visual, color)
    return sdf


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
    if p["control"]["max_steering_rad"] > 0.4:
        raise ValueError(
            "This model is validated structurally for max_steering_rad <= 0.4"
        )
    if p["control"]["forward_speed_mps"] > 2 or p["control"]["reverse_speed_mps"] > 2:
        raise ValueError("This initial low-speed model supports speeds <= 2 m/s")
    if p["control"]["steering_rate_radps"] > 0.8:
        raise ValueError(
            "Steering command rate must not exceed the joint's 0.8 rad/s limit"
        )
    return p


def main() -> None:
    g = extract_geometry(ROOT / "cad" / "TRACTOR.stp")
    p = load_parameters()
    write_xml(ROOT / "models" / "tractor" / "model.sdf", create_model(g, p))
    config = ET.Element("model")
    element(config, "name", "Los Podadores - Tractor")
    element(config, "version", "0.1.0")
    element(config, "sdf", "model.sdf", version="1.10")
    element(
        config,
        "description",
        "CAD-derived articulated tractor; provisional physical parameters.",
    )
    write_xml(ROOT / "models" / "tractor" / "model.config", config)
    write_xml(ROOT / "worlds" / "tractor_field.sdf", create_world())
    write_xml(ROOT / "worlds" / "tractor_obstacles.sdf", create_obstacle_world())
    from route import load_route_config

    write_xml(
        ROOT / "worlds" / "tractor_route.sdf",
        create_route_world(g, load_route_config(g)),
    )
    (ROOT / "cad_dimensions.json").write_text(
        json.dumps(g, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"Model built: wheelbase={g['wheelbase_m']:.3f} m; "
        f"rear radius={g['rear']['wheel_radius_m']:.3f} m; "
        f"front radius={g['front']['wheel_radius_m']:.3f} m"
    )


if __name__ == "__main__":
    main()
