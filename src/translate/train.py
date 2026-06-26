"""Train YOLOv8s on the ROD-Dataset for obstacle detection.

No GPU? Use one of these free options:

    Option A — Kaggle (recommended, free 30h/week GPU):
        1. Go to kaggle.com, create a notebook
        2. Add the ROD-Dataset as input
        3. Copy this script into a cell and run it
        4. Download best.pt from the output

    Option B — Google Colab (free T4 GPU):
        1. Open colab.research.google.com
        2. Runtime > Change runtime type > T4 GPU
        3. !pip install ultralytics
        4. Upload this script or paste the training code
        5. Download best.pt from output

    Option C — Local CPU training (slow, ~24-48h for 100 epochs):
        uv run python -m src.translate.train

Prerequisites (local or Colab)::

    pip install ultralytics
    # Download the dataset:
    kaggle datasets download -d abtinzandi/obstacle-detection-dataset -p datasets/rod --unzip
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def get_data_yaml(dataset_dir: Path) -> Path:
    """Locate or generate data.yaml for the ROD dataset."""
    candidates = [
        dataset_dir / "ROD-Dataset" / "data.yaml",
        dataset_dir / "data.yaml",
    ]
    for p in candidates:
        if p.exists():
            logger.info("Found data.yaml at %s", p)
            return p

    data_yaml = dataset_dir / "data.yaml"
    logger.info("Generating %s", data_yaml)

    yaml_content = f"""train: {(dataset_dir / "ROD-Dataset" / "train").resolve()}
val: {(dataset_dir / "ROD-Dataset" / "valid").resolve()}
test: {(dataset_dir / "ROD-Dataset" / "test").resolve()}

nc: 25
names:
  0: Car
  1: Bus
  2: Truck
  3: Motorcycle
  4: Bike
  5: Person
  6: Dog
  7: Building
  8: Tree
  9: Stairs
  10: Manhole
  11: Guard rail
  12: Pedestrian crosswalk
  13: Road
  14: Dustbin
  15: Bench
  16: Chair
  17: Plant Pot
  18: Electrical Pole
  19: Electrical Box
  20: Bicycle Rack
  21: Traffic Cone
  22: Traffic Barrel
  23: Traffic Sign
  24: Fire Hydrant
"""
    data_yaml.write_text(yaml_content)
    return data_yaml


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

    # Support running from Kaggle/Colab where dataset is at a different path
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
            "Dataset not found. Expected one of:\n"
            "  - datasets/rod/ROD-Dataset/\n"
            "  - /kaggle/input/obstacle-detection-dataset/ROD-Dataset/\n"
            "  - ROD-Dataset/\n\n"
            "Download with:\n"
            "  kaggle datasets download -d abtinzandi/obstacle-detection-dataset "
            "-p datasets/rod --unzip"
        )
        sys.exit(1)

    data_yaml = get_data_yaml(dataset_dir)
    device = detect_device()

    # Output directory — adapts to Kaggle/Colab/local
    output_base = (
        Path("/kaggle/working/models") if Path("/kaggle").exists() else Path("models")
    )
    output_dir = output_base / "translate"
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = 100 if device != "cpu" else 50  # fewer epochs on CPU
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
    if best_weights.exists():
        logger.info("Training complete. Best weights: %s", best_weights)
    else:
        logger.warning("Training finished but best.pt not found in %s", output_dir)

    logger.info("Running validation on test split...")
    model.val(data=str(data_yaml), split="test")

    logger.info("Exporting to ONNX for edge deployment...")
    model.export(format="onnx", imgsz=640)


if __name__ == "__main__":
    main()
