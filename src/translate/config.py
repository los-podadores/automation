"""Configuration constants for the translate module."""

from __future__ import annotations

from pathlib import Path
from typing import Final

# --- Paths ---
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent.parent
MODELS_DIR: Final[Path] = PROJECT_ROOT / "models" / "translate"
DEFAULT_WEIGHTS: Final[Path] = MODELS_DIR / "best.pt"
ROD_DATASET_DIR: Final[Path] = PROJECT_ROOT / "datasets" / "rod"

# --- YOLO inference ---
CONFIDENCE_THRESHOLD: Final[float] = 0.45
IOU_THRESHOLD: Final[float] = 0.50
IMAGE_SIZE: Final[int] = 640

# --- ROD-Dataset class names (25 classes) ---
ROD_CLASSES: Final[list[str]] = [
    "Car",
    "Bus",
    "Truck",
    "Motorcycle",
    "Bike",
    "Person",
    "Dog",
    "Building",
    "Tree",
    "Stairs",
    "Manhole",
    "Guard rail",
    "Pedestrian crosswalk",
    "Road",
    "Dustbin",
    "Bench",
    "Chair",
    "Plant Pot",
    "Electrical Pole",
    "Electrical Box",
    "Bicycle Rack",
    "Traffic Cone",
    "Traffic Barrel",
    "Traffic Sign",
    "Fire Hydrant",
]

# Classes that block the robot path during inter-lawn transit
TRANSIT_OBSTACLE_CLASSES: Final[list[str]] = [
    "Car",
    "Bus",
    "Truck",
    "Person",
    "Dog",
    "Bench",
    "Chair",
    "Traffic Cone",
    "Traffic Barrel",
    "Fire Hydrant",
    "Electrical Pole",
    "Electrical Box",
    "Dustbin",
    "Bicycle Rack",
]

# Classes that mark navigable terrain
NAVIGABLE_CLASSES: Final[list[str]] = ["Road", "Pedestrian crosswalk", "Manhole"]

# --- Training ---
TRAIN_EPOCHS: Final[int] = 100
TRAIN_BATCH_SIZE: Final[int] = 16
TRAIN_IMG_SIZE: Final[int] = 640
TRAIN_DEVICE: Final[str] = "0"  # GPU index, or "cpu"
TRAIN_WORKERS: Final[int] = 8
TRAIN_PATIENCE: Final[int] = 20  # early stopping patience
