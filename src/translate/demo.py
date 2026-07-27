"""CLI demo for the obstacle detection and multi-lawn planner module.

Usage:
    uv run python -m src.translate.demo
    uv run python -m src.translate.demo --images 10
    uv run python -m src.translate.demo --image path/to/image.jpg
    uv run python -m src.translate.demo --video path/to/video.mp4 --weights yolov8s.pt
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


def run_video_detection(
    detector: ObstacleDetector,
    video_path: Path,
    output_dir: Path,
    show: bool = False,
    skip_frames: int = 1,
    scale: float = 1.0,
) -> None:
    """Run detection on a video and save the annotated video."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: could not open video {video_path}")
        return
        
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    width = int(orig_width * scale)
    height = int(orig_height * scale)
    out_fps = fps / skip_frames
    
    out_path = output_dir / f"det_{video_path.name}"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(out_path), fourcc, out_fps, (width, height))
    
    print(f"\n--- Processing Video: {video_path.name} ---")
    print(f"  Original: {orig_width}x{orig_height} @ {fps:.1f} fps")
    if scale != 1.0 or skip_frames > 1:
        print(f"  Output: {width}x{height} @ {out_fps:.1f} fps (scale={scale}, skip={skip_frames})")
    
    frame_idx = 0
    total_detections = 0
    t0_total = time.perf_counter()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % skip_frames != 0:
            frame_idx += 1
            continue
            
        if scale != 1.0:
            frame = cv2.resize(frame, (width, height))
            
        detections = detector.detect(frame)
        total_detections += len(detections)
        
        annotated = detector.annotate(frame, detections)
        out.write(annotated)
        
        if show:
            cv2.imshow("Real-Time Detection", annotated)
            # Press 'q' to stop early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n  [Detenido por el usuario]")
                break
        
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  Processed frame {frame_idx}/{total_frames}...")
            
    cap.release()
    out.release()
    if show:
        cv2.destroyAllWindows()
    
    elapsed = time.perf_counter() - t0_total
    frames_processed = frame_idx // skip_frames
    avg_fps = frames_processed / elapsed if elapsed > 0 else 0
    
    print(f"\nVideo processing complete.")
    print(f"  Total frames processed: {frames_processed}/{total_frames}")
    print(f"  Total detections: {total_detections}")
    print(f"  Speed: {avg_fps:.1f} fps")
    print(f"  Saved to: {out_path}")


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
    parser.add_argument("--video", "-v", type=str, default=None, help="Process a specific video file")
    parser.add_argument("--weights", "-w", type=str, default=None, help="Path to custom YOLO weights (e.g. yolov8s.pt)")
    parser.add_argument("--show", action="store_true", help="Display the video in real-time during processing")
    parser.add_argument("--skip", type=int, default=1, help="Process every Nth frame to speed up (e.g., 2 to process half frames)")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale down video resolution (e.g., 0.5 for half resolution)")
    parser.add_argument("--no-planner", action="store_true", help="Skip the multi-lawn planner demo")
    args = parser.parse_args()

    print("Loading YOLOv8 model...")
    detector = ObstacleDetector(weights_path=args.weights)
    print("Model loaded.\n")

    DEMO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"Error: video not found: {video_path}")
            sys.exit(1)
        run_video_detection(
            detector, 
            video_path, 
            DEMO_OUTPUT_DIR, 
            show=args.show,
            skip_frames=args.skip,
            scale=args.scale,
        )
    else:
        if args.image:
            img_path = Path(args.image)
            if not img_path.exists():
                print(f"Error: image not found: {img_path}")
                sys.exit(1)
            images = [img_path]
        else:
            print(f"Picking {args.images} random test images...")
            images = find_test_images(args.images)
        run_detection(detector, images, DEMO_OUTPUT_DIR)

    if not args.no_planner:
        run_planner_demo(detector)

    print(f"\nDone. Results in: {DEMO_OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
