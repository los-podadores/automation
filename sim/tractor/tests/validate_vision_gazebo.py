"""Explicit integration test in Windows/WSL; starts its own headless Gazebo.

First runs real YOLO on rendered images. Then injects labelled test packets to
verify the control adapter, without claiming the geometric actors were detected.
Run manually: .venv/Scripts/python.exe sim/tractor/tests/validate_vision_gazebo.py
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camera_io import read_frame
from session_control import OperatorClient, read_json, write_json
from vision import PROJECT, VisionMonitor, launch_gazebo


def main():
    args = SimpleNamespace(
        distribution="Ubuntu-24.04",
        headless=True,
        vision_brake=True,
        weights=PROJECT / "models/translate/best.pt",
        confidence=0.45,
        source=None,
        frames_dir=PROJECT / "runs/tractor_camera/validation-unused",
        output_dir=None,
        timeout=90,
        start_seconds=0,
        stride=1,
        max_frames=0,
    )
    monitor = VisionMonitor(args)
    process = launch_gazebo(args)
    monitor.output = args.frames_dir
    worker = threading.Thread(target=monitor.run, daemon=True)
    worker.start()
    operator = OperatorClient(args.frames_dir)
    injection = False
    frozen = False
    label = None
    samples = {}

    def service():
        operator.heartbeat()
        if injection and not frozen:
            try:
                _, _, stamp, sequence = read_frame(args.frames_dir)
                write_json(
                    args.frames_dir / "detections.json",
                    dict(
                        mode="vision_brake",
                        source="Gazebo rendered RGB-D",
                        session_id=read_json(args.frames_dir / "session.json")[
                            "session_id"
                        ],
                        frame_sequence=sequence,
                        frame_time_s=stamp,
                        detections=[dict(label=label, confidence=0.9)] if label else [],
                        test_injection=True,
                    ),
                )
            except (OSError, ValueError):
                pass  # The controller must stop while camera data is unavailable.
        if process.poll() is not None:
            raise AssertionError(f"Gazebo terminó: {process.returncode}")
        if monitor.error:
            raise AssertionError(monitor.error)
        return operator.status()

    def wait_until(name, predicate, seconds=25):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            status = service()
            if status and status.get("speed_mps") is not None and predicate(status):
                samples[name] = status
                print(
                    f"PASS {name}: {status['state']} · {status['speed_mps']:.3f} m/s",
                    flush=True,
                )
                return status
            time.sleep(0.08)
        raise AssertionError(f"Timeout {name}: {operator.status()}")

    try:
        operator.send("start")
        wait_until(
            "real_yolo_camera_pipeline",
            lambda s: monitor.processed >= 30 and s["speed_mps"] > 0.5,
            65,
        )
        actual = read_json(args.frames_dir / "detections.json")
        assert actual["mode"] == "vision_brake" and not actual.get("test_injection")
        worker_stop = time.monotonic()
        monitor.stop_event.set()
        worker.join(timeout=10)
        assert not worker.is_alive()
        print(
            "Ahora: señales sintéticas de prueba; NO son detecciones reales.",
            flush=True,
        )
        injection = True
        wait_until("moving", lambda s: s["speed_mps"] > 0.5)
        for kind in ("person", "dog"):
            label = kind
            wait_until(
                f"stop_{kind}",
                lambda s: s["state"] == "FRENAR YOLO" and abs(s["speed_mps"]) < 0.03,
            )
            # A short run of empty detections must not release the brake.
            label = None
            deadline = time.monotonic() + 0.6
            while time.monotonic() < deadline:
                status = service()
                assert abs(status["speed_mps"]) < 0.03, (
                    "Reanudó sin completar la espera"
                )
                time.sleep(0.08)
            wait_until(f"resume_after_{kind}", lambda s: s["speed_mps"] > 0.4)
        frozen = True
        wait_until(
            "frozen_results_stop",
            lambda s: s["state"] == "SIN YOLO" and abs(s["speed_mps"]) < 0.03,
        )
        frozen = False
        wait_until("resume_after_recovery", lambda s: s["speed_mps"] > 0.4)
        operator.send("stop")
        wait_until(
            "manual_stop_priority",
            lambda s: not s["enabled"] and abs(s["speed_mps"]) < 0.03,
        )
        write_json(
            args.frames_dir / "brake_validation.json",
            dict(
                passed=True,
                real_yolo_frames=monitor.processed,
                real_yolo_result=actual,
                synthetic_packets_used_for_stop_tests=True,
                test_seconds=time.monotonic() - worker_stop,
                samples=samples,
            ),
        )
        print(
            f"PASS · Informe: {args.frames_dir / 'brake_validation.json'}", flush=True
        )
    finally:
        monitor.stop_event.set()
        worker.join(timeout=10)
        (args.frames_dir / "stop.request").write_text("close\n", encoding="utf-8")
        process.wait(timeout=20)


if __name__ == "__main__":
    main()
