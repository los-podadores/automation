"""Launch only this simulation's processes, and clean them up on exit."""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys

from build_model import ROOT, main as build


def simulation_environment() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("GZ_SIM_RESOURCE_PATH", "")
    env["GZ_SIM_RESOURCE_PATH"] = str(ROOT / "models") + (
        os.pathsep + existing if existing else ""
    )
    env.setdefault("GZ_PARTITION", "los-podadores-tractor")
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
    args = parser.parse_args()
    try:
        build()
        check_runtime(args.headless)
    except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.check:
        print("Gazebo Harmonic y Python del sistema disponibles.")
        return 0
    env = simulation_environment()
    for path in (ROOT / "models/tractor/model.sdf", ROOT / "worlds/tractor_field.sdf"):
        result = subprocess.run(
            ["gz", "sdf", "-k", str(path)], env=env, cwd=ROOT / "worlds"
        )
        if result.returncode:
            return result.returncode
    command = ["gz", "sim", "-r", "-v", "3"]
    if args.headless:
        command.append("-s")
    command.append(str(ROOT / "worlds/tractor_field.sdf"))
    gazebo = subprocess.Popen(
        command, env=env, cwd=ROOT / "worlds", start_new_session=True
    )
    keyboard = None
    try:
        if args.headless:
            return gazebo.wait()
        keyboard = subprocess.Popen(
            [sys.executable, str(ROOT / "teleop.py")], env=env, start_new_session=True
        )
        while gazebo.poll() is None and keyboard.poll() is None:
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
        stop_process(gazebo)


if __name__ == "__main__":
    raise SystemExit(main())
