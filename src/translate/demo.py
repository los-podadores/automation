"""CLI demo for the obstacle detection and multi-lawn planner module.

Usage:
    uv run python -m src.translate.demo
    uv run python -m src.translate.demo --images 10
    uv run python -m src.translate.demo --image path/to/image.jpg
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import cv2

from .config import ROD_DATASET_DIR
from .detect import ObstacleDetector
from .planner import MultiLawnPlanner

DEMO_OUTPUT_DIR = Path("runs/translate/demo")


def find_test_images(n: int = 5) -> list[Path]:
    """Pick n random test images from the ROD dataset."""
    test_dir = ROD_DATASET_DIR / "ROD-Dataset" / "dataset" / "test" / "images"
    if not test_dir.exists():
        print(f"Error: test images not found at {test_dir}")
        sys.exit(1)

    images = sorted(test_dir.glob("*.jpg"))
    if not images:
        print(f"Error: no .jpg files in {test_dir}")
        sys.exit(1)

    if n >= len(images):
        return images
    return random.sample(images, n)


def run_detection(detector: ObstacleDetector, images: list[Path], output_dir: Path) -> None:
    """Run detection on each image, print results, save annotated frames."""
    output_dir.mkdir(parents=True, exist_ok=True)

    total_detections = 0
    total_obstacles = 0

    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  [skip] Could not read {img_path.name}")
            continue

        t0 = time.perf_counter()
        detections = detector.detect(frame)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        transit = detector.filter_transit_obstacles(detections)
        total_detections += len(detections)
        total_obstacles += len(transit)

        print(f"\n--- {img_path.name} ({frame.shape[1]}x{frame.shape[0]}) ---")
        print(f"  Inference: {elapsed_ms:.1f}ms")
        print(f"  Detected: {len(detections)} objects, {len(transit)} transit obstacles")

        if detections:
            print(f"  {'Class':<22} {'Conf':>5}  {'BBox (x1,y1,x2,y2)'}")
            print(f"  {'-'*22} {'-'*5}  {'-'*24}")
            for det in sorted(detections, key=lambda d: d.confidence, reverse=True):
                marker = " *" if det.is_transit_obstacle else ""
                print(
                    f"  {det.class_name:<22} {det.confidence:.2f}  "
                    f"({det.bbox[0]:4d},{det.bbox[1]:4d},{det.bbox[2]:4d},{det.bbox[3]:4d})"
                    f"{marker}"
                )

        annotated = detector.annotate(frame, detections)
        out_path = output_dir / f"det_{img_path.name}"
        cv2.imwrite(str(out_path), annotated)

    print(f"\n{'='*50}")
    print(f"Total: {total_detections} detections across {len(images)} images")
    print(f"Transit obstacles: {total_obstacles}")
    print(f"Annotated images saved to: {output_dir}/")


def run_planner_demo(detector: ObstacleDetector) -> None:
    """Demo the multi-lawn planner with a synthetic scenario."""
    print(f"\n{'='*50}")
    print("Multi-Lawn Planner Demo")
    print(f"{'='*50}")

    planner = MultiLawnPlanner(detector)

    planner.add_lawn("front-yard", center=(0.0, 0.0), radius=8.0)
    planner.add_lawn("back-yard", center=(0.0, 30.0), radius=10.0)
    planner.add_lawn("side-garden", center=(20.0, 15.0), radius=6.0)

    planner.add_transition("front-yard", "back-yard")
    planner.add_transition("front-yard", "side-garden")
    planner.add_transition("side-garden", "back-yard")

    print("\nLawn areas:")
    for name, lawn in planner.lawns.items():
        print(f"  {name}: center={lawn.center}, radius={lawn.radius}m")

    print("\nTransitions:")
    for src, neighbors in planner.transition_graph.items():
        for dst in neighbors:
            print(f"  {src} <-> {dst}")

    for src, dst in [("front-yard", "back-yard"), ("front-yard", "side-garden"), ("back-yard", "side-garden")]:
        path = planner.plan_transit(src, dst)
        if path:
            print(f"\nPath: {src} -> {dst}")
            print(f"  Obstacle-free: {path.obstacle_free}")
            if path.blocked_by:
                print(f"  Blocked by: {', '.join(path.blocked_by)}")
            if path.waypoints:
                print("  Waypoints:")
                for wp in path.waypoints:
                    print(f"    {wp.label}: ({wp.position[0]:.1f}, {wp.position[1]:.1f})")
            else:
                print("  Direct path (no intermediate waypoints)")
        else:
            print(f"\n  No route found: {src} -> {dst}")

    test_dir = ROD_DATASET_DIR / "ROD-Dataset" / "dataset" / "test" / "images"
    sample_frames = list(test_dir.glob("*.jpg"))[:3]
    if sample_frames:
        print(f"\nObstacle check with camera frame ({sample_frames[0].name}):")
        frame = cv2.imread(str(sample_frames[0]))
        if frame is not None:
            path = planner.plan_transit("front-yard", "back-yard", frame=frame)
            if path:
                status = "CLEAR" if path.obstacle_free else "BLOCKED"
                print(f"  Path status: {status}")
                if path.blocked_by:
                    print(f"  Obstacles detected: {', '.join(path.blocked_by)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Obstacle detection demo")
    parser.add_argument("--images", "-n", type=int, default=5, help="Number of test images to process")
    parser.add_argument("--image", "-i", type=str, default=None, help="Process a specific image instead of random test images")
    parser.add_argument("--no-planner", action="store_true", help="Skip the multi-lawn planner demo")
    args = parser.parse_args()

    print("Loading YOLOv8s model...")
    detector = ObstacleDetector()
    print("Model loaded.\n")

    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            print(f"Error: image not found: {img_path}")
            sys.exit(1)
        images = [img_path]
    else:
        print(f"Picking {args.images} random test images...")
        images = find_test_images(args.images)

    DEMO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_detection(detector, images, DEMO_OUTPUT_DIR)

    if not args.no_planner:
        run_planner_demo(detector)

    print(f"\nDone. Results in: {DEMO_OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
