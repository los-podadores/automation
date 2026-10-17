"""Short autonomous crossing test, using labelled Gazebo ground truth."""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time

from build_model import ROOT, load_parameters
from control import approach
from obstacle_runtime import CrossingScenario, SceneObserver
from obstacles import (
    Decision,
    ObstacleGuard,
    circle_rectangle_gap,
    straight_path_steering,
)
from teleop import GazeboCommands


class CrossingDemo:
    def __init__(self) -> None:
        self.geometry = json.loads(
            (ROOT / "cad_dimensions.json").read_text(encoding="utf-8")
        )
        self.parameters = load_parameters()["control"]
        self.commands = GazeboCommands()
        self.observer = SceneObserver(
            rear_radius=self.geometry["rear"]["wheel_radius_m"]
        )
        self.scenario = CrossingScenario(self.observer)
        self.guard = ObstacleGuard(self.geometry)
        self.enabled = False
        self.finished = False
        self.speed = 0.0
        self.steering = 0.0
        self.last = time.monotonic()
        self.events: list[dict] = []
        self.last_reason = None
        self.min_gap = float("inf")
        self.scenario.start()

    def stop(self) -> None:
        self.enabled = False
        self.speed = 0.0
        self.commands.stop()

    def tick(self) -> tuple[Decision, object]:
        now = time.monotonic()
        dt = min(0.1, now - self.last)
        self.last = now
        vehicle, observations, fresh = self.observer.snapshot()
        if vehicle is not None:
            for obs in observations:
                gap = circle_rectangle_gap(
                    obs.x,
                    obs.y,
                    obs.radius,
                    vehicle,
                    self.guard.front,
                    self.guard.rear,
                    self.guard.half_width,
                )
                self.min_gap = min(self.min_gap, gap)
        if self.enabled and fresh and vehicle is not None and vehicle.x >= 10.5:
            self.finished = True
            self.enabled = False
        if self.finished:
            decision = Decision(False, "FINALIZADO", "Fin de la prueba corta")
        elif not self.enabled:
            decision = Decision(False, "LISTO", "Pulsa Iniciar prueba")
        else:
            target = (
                straight_path_steering(vehicle, self.geometry["wheelbase_m"])
                if vehicle
                else 0.0
            )
            self.steering = approach(
                self.steering, target, self.parameters["steering_rate_radps"] * dt
            )
            decision = self.guard.decide(
                vehicle,
                observations,
                self.parameters["forward_speed_mps"],
                self.steering,
                now,
                fresh and not self.scenario.error,
            )
        if decision.allowed:
            self.speed = approach(
                self.speed,
                self.parameters["forward_speed_mps"],
                self.parameters["acceleration_mps2"] * dt,
            )
            yaw_rate = (
                self.speed * math.tan(self.steering) / self.geometry["wheelbase_m"]
            )
            self.commands.publish(self.speed, self.steering, yaw_rate)
        else:
            self.speed = 0.0
            self.commands.publish(0.0, self.steering, 0.0)
        key = (decision.state, decision.reason)
        if key != self.last_reason:
            self.last_reason = key
            event = {
                "state": decision.state,
                "reason": decision.reason,
                "sim_time": vehicle.sim_time if vehicle else None,
                "x_m": vehicle.x if vehicle else None,
            }
            self.events.append(event)
            print(json.dumps(event, ensure_ascii=False), flush=True)
        return decision, vehicle

    def close(self) -> None:
        try:
            self.stop()
        finally:
            self.scenario.close()
        output = ROOT.parents[1] / "runs" / "tractor_demo"
        output.mkdir(parents=True, exist_ok=True)
        (output / "last_run.json").write_text(
            json.dumps(
                {
                    "perception": "Gazebo labelled ground truth; not YOLO",
                    "finished": self.finished,
                    "passed": self.passed(),
                    "minimum_conservative_gap_m": self.min_gap
                    if self.min_gap != float("inf")
                    else None,
                    "events": self.events,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def passed(self) -> bool:
        encountered = {
            label
            for label in ("Persona", "Perro")
            if any(
                event["state"] in ("FRENAR", "ESPERAR") and label in event["reason"]
                for event in self.events
            )
        }
        return (
            self.finished
            and len(encountered) == 2
            and self.min_gap >= self.guard.clearance
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prueba de decisión ante persona y perro"
    )
    parser.add_argument(
        "--headless", action="store_true", help="Inicia automáticamente sin panel"
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    demo = CrossingDemo()
    deadline = time.monotonic() + args.timeout
    try:
        if args.headless:
            demo.enabled = True
            while time.monotonic() < deadline and not demo.finished:
                demo.tick()
                time.sleep(1 / demo.parameters["publish_hz"])
            return 0 if demo.passed() else 1
        import tkinter as tk
        from tkinter import ttk

        root = tk.Tk()
        root.title("Los Podadores · Decisiones ante obstáculos")
        panel = ttk.Frame(root, padding=20)
        panel.grid()
        ttk.Label(
            panel, text="PRUEBA DE PERSONA Y PERRO", font=("Sans", 15, "bold")
        ).grid(pady=(0, 12))
        ttk.Label(panel, text="Percepción simulada · Recorrido corto de prueba").grid()
        status = tk.StringVar(value="LISTO")
        reason = tk.StringVar(value="Pulsa Iniciar prueba y observa Gazebo")
        measured = tk.StringVar(value="Esperando al tractor")
        ttk.Label(panel, textvariable=status, font=("Sans", 16, "bold")).grid(pady=12)
        ttk.Label(panel, textvariable=reason).grid()
        ttk.Label(panel, textvariable=measured).grid(pady=12)

        def start():
            if not demo.finished:
                demo.enabled = True

        start_button = ttk.Button(panel, text="Iniciar prueba", command=start)
        start_button.grid(sticky="ew", pady=4)
        ttk.Button(panel, text="DETENER", command=demo.stop).grid(sticky="ew", pady=4)
        ttk.Label(
            panel, text="Puedes observar Gazebo: este modo no necesita mantener W."
        ).grid(pady=8)

        def tick():
            decision, vehicle = demo.tick()
            status.set(decision.state)
            reason.set(decision.reason)
            if demo.finished:
                start_button.configure(state="disabled")
            if vehicle:
                measured.set(
                    f"Posición: {vehicle.x:.1f} m · Velocidad medida: {vehicle.speed:.2f} m/s"
                )
            root.after(max(1, round(1000 / demo.parameters["publish_hz"])), tick)

        exit_code = 0

        def callback_error(_kind, value, _traceback):
            nonlocal exit_code
            exit_code = 1
            print(f"Error en prueba: {value}", file=sys.stderr)
            root.quit()

        root.bind("<space>", lambda _event: demo.stop())
        root.protocol("WM_DELETE_WINDOW", root.quit)
        root.report_callback_exception = callback_error
        signal.signal(signal.SIGTERM, lambda *_: root.quit())
        root.after(100, tick)
        root.mainloop()
        root.destroy()
        return exit_code
    except KeyboardInterrupt:
        return 0
    finally:
        demo.close()


if __name__ == "__main__":
    raise SystemExit(main())
