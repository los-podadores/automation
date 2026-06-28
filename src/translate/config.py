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
# Order must match the dataset's data.yaml / class_distribution.csv
ROD_CLASSES: Final[list[str]] = [
    "Bike",                # 0
    "Building",            # 1
    "Car",                 # 2
    "Person",              # 3
    "Stairs",              # 4
    "Traffic Sign",        # 5
    "Electrical Pole",     # 6
    "Road",                # 7
    "Motorcycle",          # 8
    "Dustbin",             # 9
    "Dog",                 # 10
    "Manhole",             # 11
    "Tree",                # 12
    "Guard rail",          # 13
    "Pedestrian crosswalk",# 14
    "Truck",               # 15
    "Bus",                 # 16
    "Bench",               # 17
    "Traffic Cone",        # 18
    "Fire Hydrant",        # 19
    "Traffic Barrel",      # 20
    "Plant Pot",           # 21
    "Electrical Box",      # 22
    "Chair",               # 23
    "Bicycle Rack",        # 24
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
