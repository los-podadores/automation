"""Double-sided photographic panels; physical objects rendered by Gazebo."""

from __future__ import annotations

import math
from pathlib import Path

from build_model import box, element, inertia, plugin

ASSETS = Path(__file__).resolve().parent / "assets/posters"
HIDDEN_Z = -5.0
PANEL_SIZES = {"person": (0.95, 1.8), "dog": (1.09375, 2.0)}

# Bottom-anchored, unit X/Z quad. Each side has its own UV mapping and normal,
# so both approach directions see the same upright, unmirrored photograph.
MESH = """v -0.5 0 0
v 0.5 0 0
v 0.5 0 1
v -0.5 0 1
vt 0 0
vt 1 0
vt 1 1
vt 0 1
vn 0 -1 0
vn 0 1 0
f 1/1/1 2/2/1 3/3/1
f 1/1/1 3/3/1 4/4/1
f 2/1/2 1/2/2 4/3/2
f 2/1/2 4/3/2 3/4/2
"""


def add_posters(world, crossers):
    for filename in ("person.jpg", "dog.jpg"):
        if not (ASSETS / filename).is_file():
            raise ValueError(
                "Faltan fotos: ejecuta .venv/Scripts/python.exe sim/tractor/prepare_posters.py en Windows"
            )
    mesh_path = ASSETS / "panel.obj"
    if not mesh_path.exists() or mesh_path.read_text(encoding="utf-8") != MESH:
        mesh_path.write_text(MESH, encoding="utf-8", newline="\n")
    for name, spec in crossers.items():
        old = world.find(f"model[@name='{name}']")
        if old is not None:
            world.remove(old)
        kind = "person" if name.startswith("person") else "dog"
        width, height = PANEL_SIZES[kind]
        yaw = math.copysign(math.pi / 2, spec["end_y"] - spec["start_y"])
        actor = element(world, "model", name=name)
        element(actor, "pose", (spec["x"], spec["start_y"], HIDDEN_Z, 0, 0, yaw))
        link = element(actor, "link", name="body")
        element(link, "kinematic", "true")
        element(link, "gravity", "false")
        inertia(link, 10, (1, 1, 1))
        collision = element(link, "collision", name="panel_collision")
        element(collision, "pose", (0, 0, height / 2, 0, 0, 0))
        box(collision, (width, 0.06, height))
        visual = element(link, "visual", name="photograph")
        element(visual, "cast_shadows", "false")
        mesh = element(element(visual, "geometry"), "mesh")
        element(mesh, "uri", str(mesh_path))
        element(mesh, "scale", (width, 1, height))
        material = element(visual, "material")
        element(material, "ambient", (1, 1, 1, 1))
        element(material, "diffuse", (1, 1, 1, 1))
        metal = element(element(material, "pbr"), "metal")
        element(metal, "albedo_map", str(ASSETS / f"{kind}.jpg"))
        element(metal, "metalness", 0)
        element(metal, "roughness", 1)
        plugin(
            actor,
            "VelocityControl",
            "velocity-control",
            {"topic": f"/model/{name}/cmd_vel"},
        )
