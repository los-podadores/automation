"""Session-local operator controls between Windows and WSL; no YOLO decisions.

An expiring operator heartbeat stops the route if the Windows control window
disappears. Renewing the old heartbeat cannot restart it: a new Start is needed.
"""

from __future__ import annotations

import json
import math
import os
import time
import uuid
from pathlib import Path

LEASE_SECONDS = 2.0


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pending.json")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for attempt in range(20):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            # A WSL reader or Windows antivirus can briefly hold the NTFS file.
            # Keep the previous complete JSON until atomic replacement succeeds.
            if attempt == 19:
                raise
            time.sleep(0.01)


def read_json(path: Path):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except (OSError, ValueError):
        return None


class PacketReceiver:
    """Expire locally observed packets; Windows/WSL wall clocks may differ."""

    def __init__(self, field):
        self.field = field
        self.order = -1
        self.received = None
        self.payload = None

    def receive(self, payload, now):
        order = payload.get(self.field) if isinstance(payload, dict) else None
        if type(order) is int and order > self.order:
            self.order, self.received, self.payload = order, now, payload
        if self.received is None or not 0 <= now - self.received <= LEASE_SECONDS:
            return None
        return self.payload


class OperatorClient:
    def __init__(self, folder: Path):
        self.folder = folder
        self.action = "stop"
        self.request_id = uuid.uuid4().hex
        self.request_order = time.monotonic_ns()
        self.heartbeat_order = self.request_order
        self.status_receiver = PacketReceiver("status_order")

    def send(self, action: str):
        if action not in ("start", "stop"):
            raise ValueError("Acción de operador inválida")
        self.action, self.request_id = action, uuid.uuid4().hex
        self.request_order = max(self.request_order + 1, time.monotonic_ns())
        self.heartbeat()

    def heartbeat(self):
        self.heartbeat_order = max(self.heartbeat_order + 1, time.monotonic_ns())
        write_json(
            self.folder / "operator.json",
            dict(
                action=self.action,
                request_id=self.request_id,
                request_order=self.request_order,
                heartbeat_order=self.heartbeat_order,
                written_at=time.time(),
            ),
        )

    def status(self):
        payload = read_json(self.folder / "route_status.json")
        return self.status_receiver.receive(payload, time.monotonic())


class RouteOperator:
    def __init__(self, folder: Path):
        self.folder = folder
        self.last_request = None
        self.last_order = -1
        self.receiver = PacketReceiver("heartbeat_order")
        self.last_status = None

    def update(self, demo, payload, now):
        valid = payload is None or (
            isinstance(payload, dict)
            and payload.get("action") in ("start", "stop")
            and isinstance(payload.get("request_id"), str)
            and len(payload["request_id"]) >= 8
            and type(payload.get("request_order")) is int
            and payload["request_order"] > 0
            and type(payload.get("heartbeat_order")) is int
            and payload["heartbeat_order"] > 0
            and isinstance(payload.get("written_at"), (int, float))
            and math.isfinite(payload["written_at"])
        )
        if not valid:
            if demo.enabled:
                demo.stop()
            return False
        # Ignore older commands even if a cached file is read after a newer one.
        if payload is not None and payload["request_order"] < self.last_order:
            payload = None
        payload = self.receiver.receive(payload, now)
        if payload is None:
            if demo.enabled:
                demo.stop()
            return False
        if payload["request_order"] > self.last_order:
            self.last_request = payload["request_id"]
            self.last_order = payload["request_order"]
            print(
                json.dumps(
                    dict(
                        operator_action=payload["action"], request_order=self.last_order
                    )
                ),
                flush=True,
            )
            if payload["action"] == "start" and not demo.finished and not demo.failure:
                demo.enabled = True
            else:
                demo.stop()
        return True

    def poll(self, demo):
        payload = read_json(self.folder / "operator.json")
        return self.update(demo, payload, time.monotonic())

    def publish(self, demo, decision, vehicle):
        key = (decision.state, decision.reason, demo.enabled)
        if key != self.last_status:
            self.last_status = key
            print(
                json.dumps(
                    dict(
                        route_state=decision.state,
                        reason=decision.reason,
                        enabled=demo.enabled,
                        progress_m=demo.progress,
                    ),
                    ensure_ascii=False,
                ),
                flush=True,
            )
        future = [
            spec for spec in demo.crossers.values() if spec["trigger_s"] > demo.progress
        ]
        next_crossing = (
            min(future, key=lambda spec: spec["trigger_s"]) if future else None
        )
        write_json(
            self.folder / "route_status.json",
            dict(
                written_at=time.time(),
                status_order=time.monotonic_ns(),
                state=decision.state,
                reason=decision.reason,
                phase=demo.phase,
                enabled=demo.enabled,
                finished=demo.finished,
                failure=demo.failure,
                progress_m=demo.progress,
                length_m=demo.route.length,
                speed_mps=vehicle.speed if vehicle else None,
                sim_time_s=vehicle.sim_time if vehicle else None,
                commanded_speed_mps=demo.speed,
                geometric_allowed=demo.geometric_decision.allowed
                if demo.geometric_decision
                else None,
                geometric_state=demo.geometric_decision.state
                if demo.geometric_decision
                else None,
                vision_state=demo.vision_decision.state
                if demo.vision_decision
                else None,
                active_crossers=sorted(demo.scenario.active),
                x_m=vehicle.x if vehicle else None,
                y_m=vehicle.y if vehicle else None,
                gate_crossings=len(demo.gate_crossings),
                next_crossing=dict(
                    label=next_crossing["label"],
                    distance_m=next_crossing["trigger_s"] - demo.progress,
                )
                if next_crossing
                else None,
            ),
        )
