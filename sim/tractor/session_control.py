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


def recent(payload, now):
    if not isinstance(payload, dict):
        return False
    stamp = payload.get("written_at")
    return (
        isinstance(stamp, (int, float))
        and math.isfinite(stamp)
        and -0.5 <= now - stamp <= LEASE_SECONDS
    )


class OperatorClient:
    def __init__(self, folder: Path):
        self.folder = folder
        self.action = "stop"
        self.request_id = uuid.uuid4().hex
        self.request_order = time.monotonic_ns()

    def send(self, action: str):
        if action not in ("start", "stop"):
            raise ValueError("Acción de operador inválida")
        self.action, self.request_id = action, uuid.uuid4().hex
        self.request_order = max(self.request_order + 1, time.monotonic_ns())
        self.heartbeat()

    def heartbeat(self):
        write_json(
            self.folder / "operator.json",
            dict(
                action=self.action,
                request_id=self.request_id,
                request_order=self.request_order,
                written_at=time.time(),
            ),
        )

    def status(self):
        payload = read_json(self.folder / "route_status.json")
        return payload if recent(payload, time.time()) else None


class RouteOperator:
    def __init__(self, folder: Path):
        self.folder = folder
        self.last_request = None
        self.last_order = -1
        self.last_payload = None

    def update(self, demo, payload, now):
        valid = (
            recent(payload, now)
            and payload.get("action") in ("start", "stop")
            and isinstance(payload.get("request_id"), str)
            and len(payload["request_id"]) >= 8
            and type(payload.get("request_order")) is int
            and payload["request_order"] > 0
        )
        if not valid:
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
        now = time.time()
        if payload is not None and type(payload.get("request_order")) is not int:
            return self.update(demo, payload, now)
        if recent(payload, now) and (
            self.last_payload is None
            or payload.get("request_order", -1)
            >= self.last_payload.get("request_order", -1)
        ):
            self.last_payload = payload
        # A brief NTFS read lock does not mean the operator disappeared. Retain
        # the previous heartbeat only until its original two-second expiry.
        if payload is None or (
            self.last_payload is not None
            and payload.get("request_order", -1)
            < self.last_payload.get("request_order", -1)
        ):
            payload = self.last_payload
        return self.update(demo, payload, now)

    def publish(self, demo, decision, vehicle):
        write_json(
            self.folder / "route_status.json",
            dict(
                written_at=time.time(),
                state=decision.state,
                reason=decision.reason,
                phase=demo.phase,
                enabled=demo.enabled,
                finished=demo.finished,
                failure=demo.failure,
                progress_m=demo.progress,
                length_m=demo.route.length,
                speed_mps=vehicle.speed if vehicle else None,
                x_m=vehicle.x if vehicle else None,
                y_m=vehicle.y if vehicle else None,
                gate_crossings=len(demo.gate_crossings),
            ),
        )
