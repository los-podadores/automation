"""Conservative front-camera interlock for the simulator, with no drive publisher.

Any Person/Dog anywhere in the front image blocks motion. No bounding-box depth
is used to assert clearance. The route's geometric guard remains independent.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

from obstacles import Decision
from session_control import read_json

RESULT_TIMEOUT = 1.0
MAX_SIM_AGE = 0.8
CLEAR_SECONDS = 1.2
MIN_CLEAR_FRAMES = 3


class VisionBrake:
    def __init__(self, folder: Path, session_id: str):
        if not session_id:
            raise ValueError("Falta el identificador de la sesión de cámara")
        self.folder = folder
        self.session_id = session_id
        self.sequence = -1
        self.stamp = -1.0
        self.received = None
        self.labels = []
        self.clear_since = None
        self.clear_frames = 0

    def unavailable(self, reason):
        self.clear_since = None
        self.clear_frames = 0
        return Decision(False, "SIN YOLO", reason)

    def poll(self, sim_time):
        return self.update(
            read_json(self.folder / "detections.json"), sim_time, time.monotonic()
        )

    def update(self, payload, sim_time, now):
        """Use local receipt time plus Gazebo capture time; never compare clocks."""
        if self.received is not None and not 0 <= now - self.received <= RESULT_TIMEOUT:
            self.clear_since = None
            self.clear_frames = 0
        try:
            if (
                not isinstance(payload, dict)
                or payload.get("mode") != "vision_brake"
                or payload.get("source") != "Gazebo rendered RGB-D"
                or payload.get("session_id") != self.session_id
            ):
                return self.unavailable("Esperando YOLO de la cámara de esta sesión")
            sequence = payload["frame_sequence"]
            stamp = payload["frame_time_s"]
            boxes = payload["detections"]
            if (
                type(sequence) is not int
                or sequence < 0
                or sequence < self.sequence
                or type(stamp) not in (int, float)
                or not math.isfinite(stamp)
                or not math.isfinite(sim_time)
                or not -0.1 <= sim_time - stamp <= MAX_SIM_AGE
                or not isinstance(boxes, list)
            ):
                return self.unavailable("Resultado YOLO inválido o imagen atrasada")
            labels = []
            for box in boxes:
                if (
                    not isinstance(box, dict)
                    or box.get("label") not in ("person", "dog")
                    or type(box.get("confidence")) not in (int, float)
                    or not math.isfinite(box["confidence"])
                    or not 0 <= box["confidence"] <= 1
                ):
                    return self.unavailable("Detecciones YOLO inválidas")
                labels.append(box["label"])
            if sequence > self.sequence:
                if stamp <= self.stamp:
                    return self.unavailable("La cámara no avanzó a una imagen nueva")
                self.sequence, self.stamp, self.received = sequence, stamp, now
                self.labels = labels
                if labels:
                    self.clear_since = None
                    self.clear_frames = 0
                else:
                    if self.clear_since is None:
                        self.clear_since = now
                    self.clear_frames += 1
            elif stamp != self.stamp or labels != self.labels:
                return self.unavailable("Resultado modificado sin una imagen nueva")
            if self.received is None or not 0 <= now - self.received <= RESULT_TIMEOUT:
                return self.unavailable("YOLO dejó de entregar resultados actuales")
            if self.labels:
                names = " / ".join(
                    name
                    for label, name in (("person", "persona"), ("dog", "perro"))
                    if label in self.labels
                )
                return Decision(
                    False,
                    "FRENAR YOLO",
                    f"YOLO ve {names} en la cámara frontal",
                    "yolo",
                )
            if (
                self.clear_since is None
                or now - self.clear_since < CLEAR_SECONDS
                or self.clear_frames < MIN_CLEAR_FRAMES
            ):
                return Decision(
                    False, "ESPERAR YOLO", "Confirmando imágenes sin persona ni perro"
                )
            return Decision(
                True, "CONTINUAR", "YOLO sin detecciones; sujeto al respaldo geométrico"
            )
        except (KeyError, TypeError, ValueError):
            return self.unavailable("Resultado YOLO incompleto")
