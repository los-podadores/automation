"""Full Gazebo round trip with measured pose feedback and simulated perception."""

from __future__ import annotations

import argparse
import json
import math
import signal
import time
from pathlib import Path

from build_model import ROOT, load_parameters
from control import approach
from obstacle_runtime import CrossingScenario, SceneObserver
from obstacles import Decision, ObstacleGuard, circle_rectangle_gap
from route import RoutePath, angle_error, load_route_config
from teleop import GazeboCommands


class RouteDemo:
    def __init__(self):
        self.geometry = json.loads(
            (ROOT / "cad_dimensions.json").read_text(encoding="utf-8")
        )
        self.config = load_route_config(self.geometry)
        self.route = RoutePath(self.geometry, self.config)
        self.crossers = self.route.crossers()
        self.observer = SceneObserver(
            "tractor_route", self.geometry["rear"]["wheel_radius_m"], self.crossers
        )
        self.scenario = CrossingScenario(self.observer)
        self.commands = GazeboCommands()
        self.guard = ObstacleGuard(
            self.geometry, self.config["motion"]["obstacle_clearance_m"]
        )
        self.parameters = load_parameters()["control"]
        self.gate_obstacles = self.route.gate_observations()
        self.enabled = False
        self.finished = False
        self.failure = ""
        self.progress = 0.0
        self.speed = self.steering = 0.0
        self.previous_sim_time = None
        self.previous_gate_y = None
        self.gate_crossings = []
        self.stopped_for = set()
        self.max_tracking_error = 0.0
        self.min_actor_gap = self.min_gate_gap = float("inf")
        self.vehicle = None
        self.events, self.trace = [], []
        self.last_event = None
        self.last_trace = -1.0
        self.phase = "IDA_RECTA"
        self.scenario.start()

    def stop(self):
        self.enabled = False
        self.speed = 0.0
        self.commands.stop()

    def tick(self):
        vehicle, observations, fresh = self.observer.snapshot()
        self.vehicle = vehicle
        sim_time = vehicle.sim_time if vehicle else 0.0
        dt = max(
            0.0,
            min(
                0.15,
                sim_time
                - (
                    self.previous_sim_time
                    if self.previous_sim_time is not None
                    else sim_time
                ),
            ),
        )
        self.previous_sim_time = sim_time
        if vehicle and fresh:
            new_progress, tracking_error = self.route.project(vehicle, self.progress)
            if self.enabled:
                self.progress = new_progress
                self.max_tracking_error = max(self.max_tracking_error, tracking_error)
                if tracking_error > self.config["motion"]["tracking_limit_m"]:
                    self.failure = "El tractor se desvió del camino; reinicia la prueba"
            self.phase = self.route.at(self.progress).phase
            self.scenario.progress = self.progress
            for group, is_gate in ((observations, False), (self.gate_obstacles, True)):
                gap = min(
                    (
                        circle_rectangle_gap(
                            obs.x,
                            obs.y,
                            obs.radius,
                            vehicle,
                            self.guard.front,
                            self.guard.rear,
                            self.guard.half_width,
                        )
                        for obs in group
                    ),
                    default=float("inf"),
                )
                if is_gate:
                    self.min_gate_gap = min(self.min_gate_gap, gap)
                else:
                    self.min_actor_gap = min(self.min_actor_gap, gap)
                if self.enabled and gap < self.guard.clearance - 1e-3:
                    self.failure = (
                        "Se perdió el margen de separación; reinicia la prueba"
                    )
            if (
                self.previous_gate_y is not None
                and (self.previous_gate_y - self.route.gate_y)
                * (vehicle.y - self.route.gate_y)
                < 0
                and abs(vehicle.x - self.route.gate_x)
                < self.config["layout"]["gate_width_m"] / 2
            ):
                if (
                    not self.gate_crossings
                    or self.progress - self.gate_crossings[-1]["progress_m"] > 5
                ):
                    self.gate_crossings.append(
                        dict(
                            progress_m=self.progress,
                            sim_time=sim_time,
                            x_m=vehicle.x,
                            y_m=vehicle.y,
                        )
                    )
            self.previous_gate_y = vehicle.y
            if sim_time - self.last_trace >= 0.25:
                self.last_trace = sim_time
                self.trace.append(
                    dict(
                        t=sim_time,
                        x=vehicle.x,
                        y=vehicle.y,
                        yaw=vehicle.yaw,
                        speed=vehicle.speed,
                        s=self.progress,
                        error=tracking_error,
                    )
                )

        if self.finished:
            decision = Decision(False, "FINALIZADO", "Regresó al punto de partida")
        elif self.failure:
            decision = Decision(False, "DETENIDO", self.failure)
        elif not self.enabled:
            decision = Decision(False, "LISTO", "Pulsa Iniciar recorrido")
        else:
            desired_speed, target_steering = (
                self.route.command(vehicle, self.progress) if vehicle else (0.0, 0.0)
            )
            self.steering = approach(
                self.steering,
                target_steering,
                self.parameters["steering_rate_radps"] * dt,
            )
            remaining = self.route.length - self.progress
            if (
                vehicle
                and fresh
                and remaining < 0.4
                and math.hypot(vehicle.x, vehicle.y)
                <= self.config["motion"]["goal_tolerance_m"]
            ):
                decision = Decision(
                    False, "LLEGANDO", "Deteniendo en el punto de partida"
                )
                if abs(vehicle.speed) < 0.03:
                    self.finished = True
                    self.enabled = False
                    decision = Decision(
                        False, "FINALIZADO", "Regresó al punto de partida"
                    )
            else:
                decision = self.guard.decide(
                    vehicle,
                    observations + self.gate_obstacles,
                    desired_speed,
                    self.steering,
                    sim_time,
                    fresh and not self.scenario.error,
                )
            if (
                decision.obstacle in self.crossers
                and decision.state == "ESPERAR"
                and vehicle
                and abs(vehicle.speed) < 0.03
            ):
                self.stopped_for.add(decision.obstacle)
        if decision.allowed:
            self.speed = approach(
                self.speed, desired_speed, self.parameters["acceleration_mps2"] * dt
            )
            self.commands.publish(
                self.speed,
                self.steering,
                self.speed * math.tan(self.steering) / self.geometry["wheelbase_m"],
            )
        else:
            self.speed = 0.0
            self.commands.publish(0.0, self.steering, 0.0)
        key = (self.phase, decision.state, decision.reason)
        if key != self.last_event:
            self.last_event = key
            event = dict(
                phase=self.phase,
                state=decision.state,
                reason=decision.reason,
                obstacle=decision.obstacle,
                sim_time=sim_time,
                progress_m=self.progress,
                x_m=vehicle.x if vehicle else None,
                y_m=vehicle.y if vehicle else None,
                speed_mps=vehicle.speed if vehicle else None,
            )
            self.events.append(event)
            print(json.dumps(event, ensure_ascii=False), flush=True)
        return decision, vehicle

    def report(self):
        vehicle = self.vehicle
        goal_error = math.hypot(vehicle.x, vehicle.y) if vehicle else None
        heading_error = abs(angle_error(vehicle.yaw - math.pi)) if vehicle else None
        passed = bool(
            self.finished
            and not self.failure
            and len(self.gate_crossings) == 2
            and self.stopped_for == set(self.crossers)
            and self.min_actor_gap >= self.guard.clearance
            and self.min_gate_gap >= self.guard.clearance
            and goal_error <= self.config["motion"]["goal_tolerance_m"]
            and heading_error < 0.15
        )
        return dict(
            perception="Gazebo labelled ground truth; not YOLO",
            layout="Provisional except approximate 110 m straight",
            config=self.config,
            finished=self.finished,
            passed=passed,
            failure=self.failure,
            route_length_m=self.route.length,
            progress_m=self.progress,
            goal_error_m=goal_error,
            heading_error_rad=heading_error,
            max_tracking_error_m=self.max_tracking_error,
            minimum_actor_gap_m=self.min_actor_gap
            if math.isfinite(self.min_actor_gap)
            else None,
            minimum_gate_gap_m=self.min_gate_gap
            if math.isfinite(self.min_gate_gap)
            else None,
            gate_crossings=self.gate_crossings,
            stopped_for=sorted(self.stopped_for),
            events=self.events,
        )

    def close(self):
        try:
            self.stop()
        finally:
            self.scenario.close()
            output = ROOT.parents[1] / "runs" / "tractor_route"
            output.mkdir(parents=True, exist_ok=True)
            (output / "last_run.json").write_text(
                json.dumps(self.report(), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            (output / "trajectory.json").write_text(
                json.dumps(self.trace) + "\n", encoding="utf-8"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Recorrido de 110 m, curva, puerta y regreso"
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--control-dir",
        type=Path,
        help="Controles en el visor Windows; sin panel Tk de Linux",
    )
    parser.add_argument("--time-scale", type=float, default=1)
    parser.add_argument(
        "--timeout",
        type=float,
        default=1200,
        help="Límite de segundos reales en modo sin ventanas",
    )
    args = parser.parse_args()
    if not math.isfinite(args.time_scale) or not 1 <= args.time_scale <= 4:
        parser.error("--time-scale debe estar entre 1 y 4")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout debe ser finito y positivo")
    demo = RouteDemo()
    try:
        if args.control_dir is not None:
            from session_control import RouteOperator

            operator = RouteOperator(args.control_dir)
            from gz.msgs10.cameratrack_pb2 import CameraTrack

            camera_publisher = demo.commands.node.advertise("/gui/track", CameraTrack)
            stopping = False

            def stop_session(*_):
                nonlocal stopping
                stopping = True

            signal.signal(signal.SIGTERM, stop_session)
            while not stopping and not (args.control_dir / "stop.request").exists():
                was_enabled = demo.enabled
                connected = operator.poll(demo)
                if demo.enabled and not was_enabled:
                    message = CameraTrack()
                    message.track_mode = CameraTrack.FOLLOW
                    message.follow_target.name = "tractor"
                    message.follow_offset.x = -8
                    message.follow_offset.y = -6
                    message.follow_offset.z = 5
                    message.follow_pgain = message.track_pgain = 0.04
                    camera_publisher.publish(message)
                decision, vehicle = demo.tick()
                if not connected and not demo.finished and not demo.failure:
                    decision = Decision(
                        False,
                        "SIN CONTROL",
                        "Se perdió la conexión con el visor; pulsa Iniciar"
                        if operator.receiver.received is not None
                        else "Esperando los controles del visor de cámara",
                    )
                operator.publish(demo, decision, vehicle)
                time.sleep(1 / demo.parameters["publish_hz"])
            return 0
        if args.headless:
            demo.enabled = True
            deadline = time.monotonic() + args.timeout
            while (
                time.monotonic() < deadline and not demo.finished and not demo.failure
            ):
                demo.tick()
                time.sleep(1 / (demo.parameters["publish_hz"] * args.time_scale))
            return 0 if demo.report()["passed"] else 1
        import tkinter as tk
        from tkinter import ttk
        from gz.msgs10.cameratrack_pb2 import CameraTrack

        root = tk.Tk()
        camera_publisher = demo.commands.node.advertise("/gui/track", CameraTrack)

        def follow_tractor():
            message = CameraTrack()
            message.track_mode = CameraTrack.FOLLOW
            message.follow_target.name = "tractor"
            message.follow_offset.x = -8
            message.follow_offset.y = -6
            message.follow_offset.z = 5
            message.follow_pgain = message.track_pgain = 0.04
            camera_publisher.publish(message)

        root.title("Los Podadores · Recorrido completo 110 m")
        panel = ttk.Frame(root, padding=18)
        panel.grid()
        ttk.Label(
            panel, text="110 m · CURVA · PUERTA · REGRESO", font=("Sans", 15, "bold")
        ).grid(pady=8)
        ttk.Label(
            panel, text="Escenario aproximado · Puerta provisional de 3,5 m"
        ).grid()
        phase = tk.StringVar(value="IDA_RECTA")
        status = tk.StringVar(value="LISTO")
        reason = tk.StringVar(value="Pulsa Iniciar recorrido")
        measured = tk.StringVar(value="Esperando Gazebo")
        ttk.Label(panel, textvariable=phase, font=("Sans", 12, "bold")).grid(pady=8)
        ttk.Label(panel, textvariable=status, font=("Sans", 15, "bold")).grid()
        ttk.Label(panel, textvariable=reason).grid(pady=6)
        ttk.Label(panel, textvariable=measured).grid(pady=6)
        progress = ttk.Progressbar(panel, length=430, maximum=demo.route.length)
        progress.grid(pady=8)
        # Fixed overview of the whole route, plus live measured tractor position.
        overview = tk.Canvas(
            panel, width=500, height=195, background="#edf0e9", highlightthickness=0
        )
        overview.grid(pady=6)
        points = demo.route.points
        min_x, max_x = min(p.x for p in points) - 3, max(p.x for p in points) + 3
        min_y, max_y = -6, max(p.y for p in points) + 5
        scale = min(470 / (max_x - min_x), 165 / (max_y - min_y))

        def screen(x, y):
            return 15 + (x - min_x) * scale, 180 - (y - min_y) * scale

        coords = [value for p in points[::3] for value in screen(p.x, p.y)]
        overview.create_line(*coords, fill="#788278", width=3)
        gx, gy = screen(demo.route.gate_x, demo.route.gate_y)
        overview.create_text(gx - 25, gy - 12, text="Puerta", fill="#725315")
        sx, sy = screen(0, 0)
        overview.create_text(sx + 25, sy + 15, text="Inicio / fin", fill="#28516a")
        dot = overview.create_oval(0, 0, 0, 0, fill="#d2691e", outline="")

        def start():
            if not demo.finished and not demo.failure:
                follow_tractor()
                demo.enabled = True

        start_button = ttk.Button(
            panel, text="Iniciar recorrido / Reanudar", command=start
        )
        start_button.grid(sticky="ew", pady=4)
        ttk.Button(panel, text="DETENER — ESPACIO", command=demo.stop).grid(
            sticky="ew", pady=4
        )
        ttk.Button(
            panel, text="Seguir tractor con la cámara", command=follow_tractor
        ).grid(sticky="ew", pady=4)
        ttk.Label(
            panel,
            text="No necesitas mantener W · El marcador muestra la posición medida",
        ).grid(pady=6)
        exit_code = 0

        def tick():
            decision, vehicle = demo.tick()
            phase.set(demo.phase.replace("_", " "))
            status.set(decision.state)
            reason.set(decision.reason)
            progress["value"] = demo.progress
            if demo.finished or demo.failure:
                start_button.configure(state="disabled")
            if vehicle:
                measured.set(
                    f"Recorrido: {demo.progress:.1f} / {demo.route.length:.1f} m · {vehicle.speed:.2f} m/s · Puerta: {len(demo.gate_crossings)}/2"
                )
                x, y = screen(vehicle.x, vehicle.y)
                overview.coords(dot, x - 4, y - 4, x + 4, y + 4)
            root.after(
                max(1, round(1000 / (demo.parameters["publish_hz"] * args.time_scale))),
                tick,
            )

        def callback_error(_kind, error, _traceback):
            nonlocal exit_code
            exit_code = 1
            demo.failure = str(error)
            root.quit()

        # Handle Space before ttk's button bindings, including key release:
        # a focused Start button must never restart the vehicle after a stop.
        def emergency_stop(_event):
            demo.stop()
            return "break"

        root.bind_class("TractorEmergencyStop", "<KeyPress-space>", emergency_stop)
        root.bind_class(
            "TractorEmergencyStop", "<KeyRelease-space>", lambda _event: "break"
        )

        def install_stop_binding(widget):
            widget.bindtags(("TractorEmergencyStop", *widget.bindtags()))
            for child in widget.winfo_children():
                install_stop_binding(child)

        install_stop_binding(root)
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
