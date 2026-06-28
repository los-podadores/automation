"""Train YOLOv8s on the ROD-Dataset for obstacle detection.

Prerequisites:
    1. Download the dataset:
       uv run python src/translate/download_dataset.py abtinzandi/obstacle-detection-dataset

    2. Run training:
       uv run python -m src.translate.train
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def find_data_yaml(dataset_dir: Path) -> Path | None:
    """Locate data.yaml inside the dataset directory. Returns None if not found."""
    candidates = [
        dataset_dir / "ROD-Dataset" / "dataset" / "data.yaml",
        dataset_dir / "ROD-Dataset" / "data.yaml",
        dataset_dir / "data.yaml",
    ]
    for p in candidates:
        if p.exists():
            logger.info("Found data.yaml at %s", p)
            return p
    return None


def detect_device() -> str:
    """Auto-detect available device."""
    try:
        import torch

        if torch.cuda.is_available():
            device = "0"
            logger.info("GPU detected: %s", torch.cuda.get_device_name(0))
        else:
            device = "cpu"
            logger.info("No GPU found, using CPU (training will be slow)")
    except ImportError:
        device = "cpu"
        logger.info("PyTorch not found with CUDA, using CPU")
    return device


def main() -> None:
    from ultralytics import YOLO

    dataset_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None

    if dataset_dir is None:
        for candidate in [
            Path("datasets/rod"),
            Path("/kaggle/input/obstacle-detection-dataset"),
            Path("/content/datasets/rod"),
            Path("ROD-Dataset"),
        ]:
            if candidate.exists():
                dataset_dir = candidate
                break

    if dataset_dir is None or not dataset_dir.exists():
        logger.error(
            "Dataset not found.\n\n"
            "Download it first:\n"
            "  uv run python src/translate/download_dataset.py "
            "abtinzandi/obstacle-detection-dataset"
        )
        sys.exit(1)

    data_yaml = find_data_yaml(dataset_dir)
    if data_yaml is None:
        logger.error(
            "data.yaml not found inside %s.\n\n"
            "Make sure the dataset is properly downloaded:\n"
            "  uv run python src/translate/download_dataset.py "
            "abtinzandi/obstacle-detection-dataset",
            dataset_dir,
        )
        sys.exit(1)

    device = detect_device()

    output_base = (
        Path("/kaggle/working/models") if Path("/kaggle").exists() else Path("models")
    )
    output_dir = output_base / "translate"
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = 100 if device != "cpu" else 50
    logger.info("Starting YOLOv8s training for %d epochs on %s", epochs, device)

    model = YOLO("yolov8s.pt")

    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=640,
        batch=16 if device != "cpu" else 8,
        device=device,
        workers=2 if device != "cpu" else 0,
        patience=20,
        project=str(output_base),
        name="translate",
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        verbose=True,
        seed=42,
        deterministic=True,
        plots=True,
    )

    best_weights = output_dir / "best.pt"
    if not best_weights.exists():
        logger.warning("Training finished but best.pt not found in %s", output_dir)
        return

    logger.info("Training complete. Best weights: %s", best_weights)

    # Copy best.pt to the working directory (models/translate/)
    final_dest = Path("models/translate/best.pt")
    final_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, final_dest)
    logger.info("Copied best.pt to %s", final_dest)

    logger.info("Running validation on test split...")
    model.val(data=str(data_yaml), split="test")

    logger.info("Exporting to ONNX for edge deployment...")
    model.export(format="onnx", imgsz=640)


if __name__ == "__main__":
    main()
