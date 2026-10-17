"""Gazebo RGB-D -> latest NPZ frame for the Windows GPU inference process.

Subscribes only to rendered images, never to actor poses or class labels.
The exchange is local on the shared filesystem; no network service is exposed.
"""

from __future__ import annotations

import argparse
import json
import signal
import threading
import time
from pathlib import Path

from camera_io import decode_image, write_frame


class CameraFrames:
    def __init__(self):
        from gz.msgs10.image_pb2 import Image
        from gz.transport13 import Node

        self.lock = threading.Lock()
        self.rgb = self.depth = None
        self.error = ""
        self.received = 0.0
        self.node = Node()
        for suffix, is_depth in (("image", False), ("depth_image", True)):
            if not self.node.subscribe(
                Image, f"/tractor/front_camera/{suffix}", self.callback(is_depth)
            ):
                raise RuntimeError("No se pudo suscribir a la cámara RGB-D")

    def callback(self, is_depth):
        def receive(message):
            try:
                data = decode_image(message, is_depth)
                stamp = message.header.stamp.sec + message.header.stamp.nsec * 1e-9
                with self.lock:
                    if is_depth:
                        self.depth = data, stamp
                    else:
                        self.rgb = data, stamp
                    self.received = time.monotonic()
                    self.error = ""
            except (ValueError, TypeError) as error:
                self.error = str(error)

        return receive

    def snapshot(self):
        with self.lock:
            if (
                self.error
                or time.monotonic() - self.received > 1.0
                or self.rgb is None
                or self.depth is None
            ):
                return None
            if abs(self.rgb[1] - self.depth[1]) > 0.02:
                return None
            if self.rgb[0].shape[:2] != self.depth[0].shape:
                return None
            return self.rgb[0], self.depth[0], self.rgb[1]


def main():
    parser = argparse.ArgumentParser(description="Puente local de cámara RGB-D")
    parser.add_argument("--frames-dir", required=True, type=Path)
    args = parser.parse_args()
    args.frames_dir.mkdir(parents=True, exist_ok=True)
    observer = CameraFrames()
    sequence, last_stamp = 0, None
    stopping = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopping.set())
    signal.signal(signal.SIGINT, lambda *_: stopping.set())
    started = time.monotonic()
    last_frame = started
    try:
        while not stopping.wait(0.025):
            if (args.frames_dir / "stop.request").exists():
                return 0
            frame = observer.snapshot()
            if frame is not None and frame[2] != last_stamp:
                sequence += 1
                try:
                    write_frame(args.frames_dir, *frame, sequence)
                except PermissionError:
                    # A Windows reader may momentarily hold the destination.
                    continue
                last_stamp, last_frame = frame[2], time.monotonic()
                if sequence == 1:
                    print(
                        "Cámara RGB-D conectada: imagen real renderizada de Gazebo",
                        flush=True,
                    )
            if time.monotonic() - last_frame > (45 if sequence == 0 else 15):
                print(
                    "No llegan imágenes RGB-D actualizadas. Revisa el renderizado de Gazebo.",
                    flush=True,
                )
                return 1
        return 0
    finally:
        (args.frames_dir / "camera_status.json").write_text(
            json.dumps(
                dict(
                    frames=sequence,
                    error=observer.error,
                    source="Gazebo RGB-D rendered sensor",
                )
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    raise SystemExit(main())
