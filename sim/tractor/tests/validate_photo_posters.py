"""Actual rendered-photo -> YOLO -> brake integration. No injected detections.

Run on Windows from automation. Default verifies both outbound crossings;
--full-route also verifies the return, both gate passes and the final pose.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from session_control import OperatorClient, write_json
from vision import PROJECT, VisionMonitor, launch_gazebo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-route", action="store_true")
    parser.add_argument("--time-scale", type=float, default=2)
    options = parser.parse_args()
    args = SimpleNamespace(
        distribution="Ubuntu-24.04",
        headless=True,
        vision_brake=True,
        photo_posters=True,
        time_scale=options.time_scale,
        weights=PROJECT / "models/translate/best.pt",
        confidence=0.45,
        source=None,
        frames_dir=PROJECT / "runs/tractor_camera/validation-unused",
        output_dir=None,
        timeout=900,
        start_seconds=0,
        stride=1,
        max_frames=0,
    )
    monitor = VisionMonitor(args)
    simulation = launch_gazebo(args)
    monitor.output = args.frames_dir
    operator = OperatorClient(args.frames_dir)
    worker = threading.Thread(target=monitor.run, daemon=True)
    worker.start()
    names = ["person_out", "dog_out"] + (
        ["person_return", "dog_return"] if options.full_route else []
    )
    records = {name: {} for name in names}
    last_status = None
    succeeded = False
    deadline = time.monotonic() + 850
    previous_state = None
    appeared = {}
    captured_middle = set()
    try:
        operator.send("start")
        while time.monotonic() < deadline:
            operator.heartbeat()
            status = operator.status()
            if monitor.error:
                raise AssertionError(monitor.error)
            if simulation.poll() is not None:
                raise AssertionError(f"Gazebo terminó: {simulation.returncode}")
            if status is None or status.get("speed_mps") is None:
                time.sleep(0.05)
                continue
            last_status = status
            if status["failure"]:
                raise AssertionError(status["failure"])
            state = (status["state"], tuple(status.get("active_crossers", [])))
            if state != previous_state:
                previous_state = state
                print(
                    f"{status['progress_m']:.1f} m · {state} · {status['speed_mps']:.2f} m/s",
                    flush=True,
                )
            with monitor.lock:
                latest = monitor.latest
            if latest is not None:
                annotated, result = latest
                assert result["source"] == "Gazebo rendered RGB-D" and not result.get(
                    "test_injection"
                )
                for name in names:
                    label = "person" if name.startswith("person") else "dog"
                    record = records[name]
                    boxes = [d for d in result["detections"] if d["label"] == label]
                    if name in status.get("active_crossers", []):
                        appeared.setdefault(name, result["frame_time_s"])
                        middle = 5.4 if label == "person" else 3.8
                        if (
                            name not in captured_middle
                            and result["frame_time_s"] - appeared[name] >= middle
                        ):
                            cv2.imwrite(
                                str(args.frames_dir / f"{name}_mid_crossing.jpg"),
                                annotated,
                            )
                            captured_middle.add(name)
                    if name in status.get("active_crossers", []) and boxes:
                        if "detected" not in record:
                            record["detected"] = result
                            cv2.imwrite(
                                str(args.frames_dir / f"{name}_detected.jpg"), annotated
                            )
                            print(
                                f"DETECCIÓN REAL {name}: {max(d['confidence'] for d in boxes):.3f}",
                                flush=True,
                            )
                        if (
                            status["state"] == "FRENAR YOLO"
                            and abs(status["speed_mps"]) < 0.03
                        ):
                            record.setdefault("stopped", status)
                            if status.get("geometric_allowed") is True:
                                record["vision_led_stop"] = True
                    if (
                        "stopped" in record
                        and "resumed" not in record
                        and status["state"] == "CONTINUAR"
                        and status["speed_mps"] > 0.25
                        and not boxes
                    ):
                        record["resumed"] = status
                        print(f"REANUDÓ {name}", flush=True)
            complete = all(
                all(
                    key in record
                    for key in ("detected", "stopped", "resumed", "vision_led_stop")
                )
                for record in records.values()
            )
            if complete and (
                status["finished"] if options.full_route else status["progress_m"] > 90
            ):
                succeeded = True
                break
            if status["finished"]:
                break
            if not options.full_route and status["progress_m"] > 95:
                break
            time.sleep(0.06)
        if not succeeded:
            raise AssertionError(
                f"Validación incompleta: { {name: list(r) for name, r in records.items()} }"
            )
    finally:
        operator.send("stop")
        monitor.stop_event.set()
        worker.join(timeout=10)
        (args.frames_dir / "stop.request").write_text("close\n", encoding="utf-8")
        simulation.wait(timeout=25)
        write_json(
            args.frames_dir / "photo_validation.json",
            dict(
                passed=succeeded,
                full_route=options.full_route,
                synthetic_detections=False,
                real_yolo_frames=monitor.processed,
                records=records,
                final_status=last_status,
                error=monitor.error,
            ),
        )
        print(f"Informe: {args.frames_dir / 'photo_validation.json'}", flush=True)
    if succeeded:
        print(
            "PASS: fotos renderizadas, detecciones reales, parada y reanudación",
            flush=True,
        )


if __name__ == "__main__":
    main()
