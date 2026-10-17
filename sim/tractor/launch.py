"""Launch only this simulation's processes, and clean them up on exit."""

from __future__ import annotations

import argparse
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from build_model import ROOT, main as build, plugin, write_xml


def simulation_environment(
    obstacles: bool = False, route: bool = False
) -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("GZ_SIM_RESOURCE_PATH", "")
    env["GZ_SIM_RESOURCE_PATH"] = str(ROOT / "models") + (
        os.pathsep + existing if existing else ""
    )
    env.setdefault(
        "GZ_PARTITION",
        "los-podadores-tractor-route"
        if route
        else "los-podadores-tractor-obstacles"
        if obstacles
        else "los-podadores-tractor",
    )
    return env


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
            process.wait(timeout=5)
            return
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            continue


def check_runtime(headless: bool = False) -> None:
    if sys.platform != "linux":
        raise RuntimeError(
            "Ejecuta esta simulación en Ubuntu 24.04 / WSL 2. Consulta README.md."
        )
    if not shutil.which("gz"):
        raise RuntimeError("Falta Gazebo. Ejecuta: bash sim/tractor/setup_ubuntu.sh")
    version = subprocess.run(
        ["gz", "sim", "--versions"], capture_output=True, text=True, timeout=15
    )
    if version.returncode or not any(
        line.strip().startswith("8.") for line in version.stdout.splitlines()
    ):
        raise RuntimeError(
            "Se requiere Gazebo Harmonic (gz-sim 8). Consulta README.md."
        )
    imports = "from gz.transport13 import Node; from gz.msgs10.twist_pb2 import Twist"
    if not headless:
        imports += "; import tkinter"
        if not os.environ.get("DISPLAY"):
            raise RuntimeError(
                "No hay pantalla gráfica. En Windows comprueba WSLg (wsl --update)."
            )
    result = subprocess.run(
        [sys.executable, "-c", imports], capture_output=True, text=True
    )
    if result.returncode:
        raise RuntimeError(
            "Faltan dependencias del Python de Ubuntu. Usa /usr/bin/python3 y setup_ubuntu.sh.\n"
            + result.stderr
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Tractor 3D en Gazebo Harmonic")
    parser.add_argument(
        "--headless", action="store_true", help="Solo servidor; sin panel de teclado"
    )
    parser.add_argument(
        "--check", action="store_true", help="Comprueba instalación sin abrir ventanas"
    )
    parser.add_argument(
        "--obstacles",
        action="store_true",
        help="Persona/perro móviles y asistencia de parada",
    )
    parser.add_argument(
        "--demo", action="store_true", help="Prueba corta autónoma ante persona y perro"
    )
    parser.add_argument(
        "--route", action="store_true", help="110 m, curva, puerta y regreso"
    )
    parser.add_argument(
        "--camera",
        action="store_true",
        help="Activa cámara RGB-D; YOLO se ejecuta aparte como observador",
    )
    parser.add_argument(
        "--frames-dir", type=Path, default=ROOT.parents[1] / "runs/tractor_camera/live"
    )
    parser.add_argument(
        "--time-scale",
        type=float,
        default=1,
        help="Velocidad del tiempo simulado (1 a 4), solo --route",
    )
    args = parser.parse_args()
    if not math.isfinite(args.time_scale) or not 1 <= args.time_scale <= 4:
        parser.error("--time-scale debe estar entre 1 y 4")
    if args.time_scale != 1 and not args.route:
        parser.error("--time-scale requiere --route")
    if args.route and (args.demo or args.obstacles):
        parser.error("Usa --route solo; ya incluye persona y perro en ambos sentidos")
    if args.camera and not args.route:
        parser.error("--camera requiere --route")
    args.obstacles = args.obstacles or args.demo
    try:
        build()
        check_runtime(args.headless)
    except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.check:
        print("Gazebo Harmonic y Python del sistema disponibles.")
        return 0
    env = simulation_environment(args.obstacles, args.route)
    world_path = (
        ROOT
        / "worlds"
        / (
            "tractor_route.sdf"
            if args.route
            else "tractor_obstacles.sdf"
            if args.obstacles
            else "tractor_field.sdf"
        )
    )
    temporary_world = None
    if args.route and (args.time_scale != 1 or args.camera):
        document = ET.parse(world_path).getroot()
        document.find("world/physics/real_time_factor").text = str(args.time_scale)
        if args.camera:
            plugin(
                document.find("world"), "Sensors", "sensors", {"render_engine": "ogre2"}
            )
        document.find("world/include/uri").text = str(ROOT / "models/tractor/model.sdf")
        temporary_world = tempfile.TemporaryDirectory(prefix="tractor-route-")
        world_path = Path(temporary_world.name) / "tractor_route.sdf"
        write_xml(world_path, document)
    for path in (ROOT / "models/tractor/model.sdf", world_path):
        result = subprocess.run(
            ["gz", "sdf", "-k", str(path)], env=env, cwd=ROOT / "worlds"
        )
        if result.returncode:
            return result.returncode
    command = ["gz", "sim", "-r", "-v", "3"]
    if args.headless:
        command.append("-s")
    command.append(str(world_path))
    gazebo = subprocess.Popen(
        command, env=env, cwd=ROOT / "worlds", start_new_session=True
    )
    keyboard = None
    camera_bridge = None
    try:
        if args.camera:
            args.frames_dir.mkdir(parents=True, exist_ok=True)
            if (args.frames_dir / "stop.request").exists():
                raise RuntimeError(
                    "Esta sesión de cámara ya terminó; utiliza otra --frames-dir"
                )
            camera_bridge = subprocess.Popen(
                [
                    sys.executable,
                    str(ROOT / "camera_bridge.py"),
                    "--frames-dir",
                    str(args.frames_dir),
                ],
                env=env,
                start_new_session=True,
            )
        if args.headless and not (args.demo or args.route):
            return gazebo.wait()
        controller = [
            sys.executable,
            str(
                ROOT
                / (
                    "route_demo.py"
                    if args.route
                    else "demo.py"
                    if args.demo
                    else "teleop.py"
                )
            ),
        ]
        if args.route:
            controller.extend(["--time-scale", str(args.time_scale)])
        if (args.demo or args.route) and args.headless:
            controller.append("--headless")
        elif args.obstacles and not args.demo:
            controller.append("--obstacles")
        keyboard = subprocess.Popen(controller, env=env, start_new_session=True)
        while gazebo.poll() is None and keyboard.poll() is None:
            if camera_bridge is not None and camera_bridge.poll() is not None:
                return camera_bridge.returncode
            try:
                return_code = keyboard.wait(timeout=0.25)
                return return_code
            except subprocess.TimeoutExpired:
                continue
        return (
            (gazebo.returncode or 0)
            if gazebo.poll() is not None
            else (keyboard.returncode or 0)
        )
    except KeyboardInterrupt:
        return 0
    finally:
        # Close teleop first so it can send zero commands before Gazebo exits.
        if keyboard is not None:
            stop_process(keyboard)
        if camera_bridge is not None:
            stop_process(camera_bridge)
        stop_process(gazebo)
        if temporary_world is not None:
            temporary_world.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
