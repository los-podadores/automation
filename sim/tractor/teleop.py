"""Keyboard panel for the Gazebo tractor. Uses Ubuntu's system Python."""

from __future__ import annotations

import json
import math
import signal
import sys
import time

from build_model import ROOT, load_parameters
from control import KeyboardDrive

MOTION_KEYS = {"w", "a", "s", "d", "up", "down", "left", "right"}


class GazeboCommands:
    def __init__(self) -> None:
        try:
            from gz.msgs10.double_pb2 import Double
            from gz.msgs10.twist_pb2 import Twist
            from gz.transport13 import Node
        except ImportError as exc:
            raise RuntimeError(
                "Faltan los bindings de Gazebo Harmonic. En Ubuntu 24.04 ejecuta "
                "bash sim/tractor/setup_ubuntu.sh y usa /usr/bin/python3."
            ) from exc
        self.double_type = Double
        self.twist_type = Twist
        self.node = Node()
        self.steering_pub = self.node.advertise("/model/tractor/steering_angle", Double)
        self.drive_pub = self.node.advertise("/model/tractor/rear_axle/cmd_vel", Twist)
        if not self.steering_pub or not self.drive_pub:
            raise RuntimeError("No se pudieron anunciar los topics del tractor.")

    def publish(self, speed: float, steering: float, yaw_rate: float) -> None:
        angle = self.double_type()
        angle.data = steering
        twist = self.twist_type()
        twist.linear.x = speed
        twist.angular.z = yaw_rate
        steering_ok = self.steering_pub.publish(angle)
        drive_ok = self.drive_pub.publish(twist)
        if not steering_ok or not drive_ok:
            raise RuntimeError("Gazebo Transport rechazó una orden de movimiento.")

    def stop(self) -> None:
        self.publish(0.0, 0.0, 0.0)


def main() -> int:
    try:
        import tkinter as tk
        from tkinter import ttk

        p = load_parameters()["control"]
        g = json.loads((ROOT / "cad_dimensions.json").read_text(encoding="utf-8"))
        commands = GazeboCommands()
        drive = KeyboardDrive(
            wheelbase=g["wheelbase_m"],
            forward_speed=p["forward_speed_mps"],
            reverse_speed=p["reverse_speed_mps"],
            max_steering=p["max_steering_rad"],
            steering_rate=p["steering_rate_radps"],
            acceleration=p["acceleration_mps2"],
        )
    except (ImportError, RuntimeError, OSError, ValueError) as exc:
        print(f"No se puede iniciar el teclado: {exc}", file=sys.stderr)
        return 1
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"No se puede abrir el panel gráfico de WSLg: {exc}", file=sys.stderr)
        return 1
    root.title("Los Podadores · Teclado del tractor")
    root.resizable(False, False)
    panel = ttk.Frame(root, padding=20)
    panel.grid()
    ttk.Label(panel, text="TRACTOR · SIMULACIÓN 3D", font=("Sans", 15, "bold")).grid(
        pady=(0, 12)
    )
    ttk.Label(
        panel, text="ENTER habilita · Mantén W/S o ↑/↓ para avanzar/retroceder"
    ).grid()
    ttk.Label(panel, text="Mantén A/D o ←/→ para dirigir · Suelta para centrar").grid()
    ttk.Label(panel, text="ESPACIO detiene y deshabilita · ESC cierra").grid(
        pady=(0, 12)
    )
    status = tk.StringVar(value="Detenido · Pulsa ENTER para habilitar")
    values = tk.StringVar(value="Orden: 0.00 m/s · Dirección: 0.0°")
    ttk.Label(panel, textvariable=status).grid()
    ttk.Label(panel, textvariable=values).grid(pady=8)
    ttk.Label(panel, text="Mantén el foco en este panel para conducir.").grid()
    ttk.Label(panel, text="Valores mostrados: órdenes, no mediciones reales.").grid(
        pady=(0, 12)
    )

    armed = False
    exit_code = 0
    last_tick = time.monotonic()
    pending_releases: dict[str, str] = {}

    def disarm(_event=None) -> None:
        nonlocal armed
        armed = False
        drive.stop()
        status.set("Detenido · Pulsa ENTER para habilitar")
        values.set("Orden: 0.00 m/s · Dirección: 0.0°")
        commands.stop()

    def key_down(event) -> None:
        nonlocal armed
        key = event.keysym.lower()
        pending = pending_releases.pop(key, None)
        if pending is not None:
            root.after_cancel(pending)
        if key == "escape":
            root.quit()
        elif key == "space":
            disarm()
        elif key == "return":
            armed = True
            status.set("Habilitado · ESPACIO para detener")
        elif armed and key in MOTION_KEYS:
            drive.keys.add(key)

    def release(key: str) -> None:
        pending_releases.pop(key, None)
        drive.keys.discard(key)

    def key_up(event) -> None:
        key = event.keysym.lower()
        pending = pending_releases.pop(key, None)
        if pending is not None:
            root.after_cancel(pending)
        # X11 can emit release/press pairs during autorepeat. Cancel on press.
        pending_releases[key] = root.after(30, release, key)

    def tick() -> None:
        nonlocal last_tick
        now = time.monotonic()
        elapsed = now - last_tick
        last_tick = now
        if elapsed > 0.5:
            disarm()  # Require re-arming after a stalled GUI/event loop.
        speed, angle, yaw_rate = drive.step(max(elapsed, 1e-6))
        commands.publish(speed, angle, yaw_rate)
        values.set(f"Orden: {speed:+.2f} m/s · Dirección: {math.degrees(angle):+.1f}°")
        root.after(max(1, round(1000 / p["publish_hz"])), tick)

    def callback_error(_kind, value, _traceback) -> None:
        nonlocal exit_code
        exit_code = 1
        print(f"Error del panel: {value}", file=sys.stderr)
        root.quit()

    ttk.Button(panel, text="DETENER — ESPACIO", command=disarm).grid(sticky="ew")
    root.bind("<KeyPress>", key_down)
    root.bind("<KeyRelease>", key_up)
    root.bind("<FocusOut>", disarm)
    root.protocol("WM_DELETE_WINDOW", root.quit)
    root.report_callback_exception = callback_error
    signal.signal(signal.SIGTERM, lambda *_: root.quit())
    try:
        commands.stop()
        root.after(50, tick)
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        # Let the final zero messages leave the native transport publisher.
        for _ in range(3):
            try:
                commands.stop()
            except RuntimeError:
                break
            time.sleep(0.05)
        root.destroy()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
