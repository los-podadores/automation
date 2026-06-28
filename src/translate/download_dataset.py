"""Download a Kaggle dataset using kagglehub.

Usage:
    uv run python src/translate/download_dataset.py abtinzandi/obstacle-detection-dataset
    uv run python src/translate/download_dataset.py abtinzandi/obstacle-detection-dataset -o datasets/rod
"""

from __future__ import annotations

import argparse
import sys

DEFAULT_OUTPUT_DIR = "datasets/rod"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a Kaggle dataset using kagglehub.",
    )
    parser.add_argument(
        "dataset",
        help="Kaggle dataset handle (e.g. abtinzandi/obstacle-detection-dataset)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    try:
        import kagglehub
    except ImportError:
        print(
            "Error: kagglehub is not installed.\n"
            "Install it with: uv add kagglehub",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Downloading dataset: {args.dataset}")
    path = kagglehub.dataset_download(args.dataset, output_dir=args.output_dir)
    print(f"Dataset downloaded to: {path}")


if __name__ == "__main__":
    main()
