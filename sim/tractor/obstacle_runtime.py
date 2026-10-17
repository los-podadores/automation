"""Gazebo ground-truth adapter and crossing scenario. No image recognition here."""

from __future__ import annotations

import math
import threading
import time

from obstacles import CROSSERS, Observation, VehicleState


class SceneObserver:
    def __init__(
        self, world: str = "tractor_obstacles", rear_radius: float = 0.5
    ) -> None:
        from gz.msgs10.model_pb2 import Model
        from gz.msgs10.pose_v_pb2 import Pose_V
        from gz.transport13 import Node

        self.lock = threading.Lock()
        self.rear_radius = rear_radius
        self.vehicle: VehicleState | None = None
        self.received = 0.0
        self.actors: dict[str, tuple[Observation, float]] = {}
        self.node = Node()
        if not self.node.subscribe(
            Model, "/model/tractor/joint_state", self.on_vehicle
        ):
            raise RuntimeError("No se pudo observar el tractor")
        if not self.node.subscribe(Pose_V, f"/world/{world}/pose/info", self.on_poses):
            raise RuntimeError("No se pudo observar el escenario")

    def on_vehicle(self, msg) -> None:
        q = msg.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y**2 + q.z**2))
        velocities = [
            joint.axis1.velocity
            for joint in msg.joint
            if joint.name in ("rear_left_spin", "rear_right_spin")
        ]
        speed = sum(velocities) * self.rear_radius / 2 if len(velocities) == 2 else 0.0
        sim_time = msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9
        with self.lock:
            self.vehicle = VehicleState(
                msg.pose.position.x, msg.pose.position.y, yaw, speed, sim_time
            )
            self.received = time.monotonic()

    def on_poses(self, msg) -> None:
        now = time.monotonic()
        with self.lock:
            for pose in msg.pose:
                spec = CROSSERS.get(pose.name)
                if spec is not None:
                    self.actors[pose.name] = (
                        Observation(
                            pose.name,
                            spec["label"],
                            pose.position.x,
                            pose.position.y,
                            spec["radius"],
                        ),
                        now,
                    )

    def snapshot(self):
        now = time.monotonic()
        with self.lock:
            fresh = (
                now - self.received < 0.6
                and len(self.actors) == len(CROSSERS)
                and all(now - stamp < 0.6 for _obs, stamp in self.actors.values())
            )
            return self.vehicle, [obs for obs, _stamp in self.actors.values()], fresh


class CrossingScenario:
    def __init__(self, observer: SceneObserver) -> None:
        self.observer = observer
        self.stopped = threading.Event()
        self.error = ""
        self.thread = threading.Thread(
            target=self.run, daemon=True, name="crossing-scenario"
        )

    def start(self) -> None:
        self.thread.start()

    def run(self) -> None:
        from gz.msgs10.twist_pb2 import Twist
        from gz.transport13 import Node

        node = Node()
        publishers = {
            name: node.advertise(f"/model/{name}/cmd_vel", Twist) for name in CROSSERS
        }
        active: set[str] = set()
        try:
            while not self.stopped.wait(0.05):
                vehicle, observations, fresh = self.observer.snapshot()
                positions = {obs.name: obs for obs in observations}
                self.error = ""
                for name, spec in CROSSERS.items():
                    if vehicle is not None and vehicle.x >= spec["trigger_x"]:
                        active.add(name)
                    message = Twist()
                    if fresh and name in active and name in positions:
                        direction = math.copysign(1.0, spec["end_y"] - spec["start_y"])
                        remaining = direction * (spec["end_y"] - positions[name].y)
                        if remaining > 0.03:
                            # VelocityControl uses the model frame; +X points across the lane.
                            message.linear.x = min(spec["speed"], 2 * remaining)
                    if not publishers[name].publish(message):
                        self.error = "No se pudo actualizar un obstáculo"
        except Exception as exc:
            self.error = f"El escenario de obstáculos se detuvo: {exc}"
        finally:
            for publisher in publishers.values():
                publisher.publish(Twist())

    def close(self) -> None:
        self.stopped.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1)
