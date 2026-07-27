"""Launch TensorBoard to visualize YOLOv8 training in real time.

Usage:
    uv run python -m src.translate.visualize
    uv run python -m src.translate.visualize --logdir models/translate
    uv run python -m src.translate.visualize --port 6007
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DEFAULT_LOGDIR = Path("models/translate")


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch TensorBoard for training monitoring")
    parser.add_argument(
        "-d", "--logdir",
        type=Path,
        default=DEFAULT_LOGDIR,
        help=f"Directory containing training logs (default: {DEFAULT_LOGDIR})",
    )
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=6006,
        help="Port for TensorBoard server (default: 6006)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host for TensorBoard server (default: localhost)",
    )
    args = parser.parse_args()

    if not args.logdir.exists():
        print(f"Error: log directory not found at {args.logdir}")
        print("Train a model first: uv run python -m src.translate.train")
        sys.exit(1)

    print("Starting TensorBoard...")
    print(f"  Log directory: {args.logdir.resolve()}")
    print(f"  URL: http://{args.host}:{args.port}")
    print("\nPress Ctrl+C to stop.\n")

    os.execvp(
        "tensorboard",
        [
            "tensorboard",
            "--logdir", str(args.logdir),
            "--host", args.host,
            "--port", str(args.port),
        ],
    )


if __name__ == "__main__":
    main()
