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

# --- Autonomous Vehicle Dimensions & Kinematics (STEP CAD Ground Truth: TRACTOR.stp, uniformly scaled) ---
VEHICLE_SIZE_SCALE: Final[float] = 1.5  # Uniform rigid-body scale factor over CAD ground truth
WHEELBASE: Final[float] = 2.00 * VEHICLE_SIZE_SCALE  # Distance between front and rear axles (meters)
REAR_TRACK_WIDTH: Final[float] = 1.40 * VEHICLE_SIZE_SCALE  # Lateral distance between rear wheels (meters)
FRONT_TRACK_WIDTH: Final[float] = 1.29 * VEHICLE_SIZE_SCALE  # Lateral distance between front wheels (meters)
TRACK_WIDTH: Final[float] = 1.40 * VEHICLE_SIZE_SCALE  # Dominant lateral track width (meters)
REAR_WHEEL_RADIUS: Final[float] = 0.50 * VEHICLE_SIZE_SCALE  # Rear drive wheel radius (meters)
FRONT_WHEEL_RADIUS: Final[float] = 0.37 * VEHICLE_SIZE_SCALE  # Front steering wheel radius (meters)
WHEEL_RADIUS: Final[float] = 0.50 * VEHICLE_SIZE_SCALE  # Drive wheel radius for odometry (meters)
CAR_LENGTH: Final[float] = 2.90 * VEHICLE_SIZE_SCALE  # Overall longitudinal footprint with wheels (meters)
CAR_WIDTH: Final[float] = 1.72 * VEHICLE_SIZE_SCALE  # Overall lateral width with rear wheels and steering envelope (meters)
CAR_HEIGHT: Final[float] = 1.25 * VEHICLE_SIZE_SCALE  # Overall height (meters)
VEHICLE_MASS_KG: Final[float] = 414.0  # Total vehicle mass (kg): 300 chassis + 2x35 rear + 2x20 front + 2x2 knuckle
COLLISION_RADIUS: Final[float] = 1.15 * VEHICLE_SIZE_SCALE  # Bounding collision radius (meters)

# Non-holonomic Ackermann Dynamic Limits
MAX_STEER_RAD: Final[float] = 0.40  # ~23 degrees maximum front steering angle (joint limit)
MAX_SPEED_MPS: Final[float] = 2.0  # 7.2 km/h max forward velocity
MIN_SPEED_MPS: Final[float] = -1.0  # -3.6 km/h max reverse velocity
MAX_ACCEL_MPSS: Final[float] = 1.5  # Max forward acceleration (m/s^2)
MAX_BRAKE_MPSS: Final[float] = 2.5  # Max service braking deceleration (m/s^2)
DT_SECONDS: Final[float] = 0.05  # 20 Hz simulation / control loop

# --- Surround 4-Pod Camera-LiDAR Perception ---
LIDAR_MAX_RANGE: Final[float] = 15.0  # Max LiDAR range in meters (near/mid-field solid state)
LIDAR_NUM_RAYS: Final[int] = 72  # 72 radial rays covering full 360-degree perimeter (5 deg res)
LIDAR_FOV_RAD: Final[float] = 6.283185307179586  # 2 * pi (full 360-degree surround)

# --- Dual Directional 3D LiDAR & Ground Segmentation ---
LIDAR_DIRECTIONAL_PITCH_RAD: Final[float] = 0.349066  # ~20 degrees downward tilt
LIDAR_VERTICAL_SAMPLES: Final[int] = 16  # 16-channel vertical resolution
LIDAR_VERTICAL_MIN_RAD: Final[float] = -0.436332  # -25 degrees (ground plane sweep)
LIDAR_VERTICAL_MAX_RAD: Final[float] = 0.261799  # +15 degrees (obstacle sweep)
GROUND_EPSILON_METERS: Final[float] = 0.05  # 5 cm ground plane boundary
SMALL_OBSTACLE_MIN_HEIGHT: Final[float] = 0.06  # 6 cm minimum protrusion for animal/curb
MAX_OBSTACLE_HEIGHT: Final[float] = 2.0  # 2.0 m ceiling for vehicle obstacle envelope
DITCH_NEGATIVE_THRESHOLD: Final[float] = -0.12  # -12 cm step drop for negative obstacles/holes
MAX_TRAVERSABLE_SLOPE_RAD: Final[float] = 0.349066  # 20 degrees maximum terrain slope

# --- Physical Directional 3D LiDAR Mounting Extrinsics (scaled with vehicle) ---
# Tractor chassis body length = CAR_LENGTH centered at x=0; bumpers at x=+/-CAR_LENGTH/2
FRONT_LIDAR_X: Final[float] = CAR_LENGTH / 2.0  # Front bumper leading face (meters ahead of vehicle center)
FRONT_LIDAR_Y: Final[float] = 0.00  # Centered laterally (meters)
FRONT_LIDAR_Z: Final[float] = 0.60 * VEHICLE_SIZE_SCALE  # Front bumper mount height (meters above ground)
FRONT_LIDAR_PITCH_RAD: Final[float] = 0.349066  # ~20 degrees downward tilt
FRONT_LIDAR_YAW_RAD: Final[float] = 0.00  # Forward facing

REAR_LIDAR_X: Final[float] = -CAR_LENGTH / 2.0  # Rear hitch/bumper trailing face (meters behind vehicle center)
REAR_LIDAR_Y: Final[float] = 0.00  # Centered laterally (meters)
REAR_LIDAR_Z: Final[float] = 0.60 * VEHICLE_SIZE_SCALE  # Rear bumper mount height (meters above ground)
REAR_LIDAR_PITCH_RAD: Final[float] = 0.349066  # ~20 degrees downward tilt
REAR_LIDAR_YAW_RAD: Final[float] = 3.141592653589793  # 180 degrees (rear facing)

