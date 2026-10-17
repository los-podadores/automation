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

# --- Autonomous Vehicle Dimensions & Kinematics (Full-Scale Utility Car) ---
CAR_LENGTH: Final[float] = 3.80  # Overall longitudinal length (meters)
CAR_WIDTH: Final[float] = 1.90  # Overall lateral width (meters)
CAR_HEIGHT: Final[float] = 1.40  # Overall height (meters)
WHEELBASE: Final[float] = 2.50  # Distance between front and rear axles (meters)
TRACK_WIDTH: Final[float] = 1.60  # Lateral distance between left and right wheels (meters)
WHEEL_RADIUS: Final[float] = 0.38  # Wheel radius (meters)
VEHICLE_MASS_KG: Final[float] = 950.0  # Vehicle mass (kg)
COLLISION_RADIUS: Final[float] = 1.25  # Bounding collision radius (meters)

# Non-holonomic Ackermann Dynamic Limits
MAX_STEER_RAD: Final[float] = 0.5585  # ~32 degrees maximum front steering angle
MAX_SPEED_MPS: Final[float] = 3.0  # 10.8 km/h max forward velocity
MIN_SPEED_MPS: Final[float] = -1.2  # -4.3 km/h max reverse velocity
MAX_ACCEL_MPSS: Final[float] = 1.5  # Max forward acceleration (m/s^2)
MAX_BRAKE_MPSS: Final[float] = 2.5  # Max service braking deceleration (m/s^2)
DT_SECONDS: Final[float] = 0.05  # 20 Hz simulation / control loop

# --- Surround 4-Pod Camera-LiDAR Perception ---
LIDAR_MAX_RANGE: Final[float] = 15.0  # Max LiDAR range in meters (near/mid-field solid state)
LIDAR_NUM_RAYS: Final[int] = 72  # 72 radial rays covering full 360-degree perimeter (5 deg res)
LIDAR_FOV_RAD: Final[float] = 6.283185307179586  # 2 * pi (full 360-degree surround)

