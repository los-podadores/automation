# YOLOv8s Obstacle Detection — Training & Execution

## Overview

This module detects real-world obstacles for multi-lawn navigation using YOLOv8s trained on the [ROD-Dataset](https://www.kaggle.com/datasets/abtinzandi/obstacle-detection-dataset) (25 obstacle classes, 24k+ images).

## Prerequisites

```bash
uv sync
```

## 1. Download the Dataset

```bash
# Install kaggle CLI if not already installed
uv pip install kaggle

# Set up API credentials (download kaggle.json from kaggle.com → Settings → API)
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download and unzip
kaggle datasets download -d abtinzandi/obstacle-detection-dataset -p datasets/rod --unzip
```

After download, the structure should be:
```
datasets/rod/
└── ROD-Dataset/
    ├── data.yaml
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
```

## 2. Train the Model

```bash
uv run python -m src.translate.train
```

This will:
- Auto-detect GPU/CPU
- Train YOLOv8s for 100 epochs on GPU (50 on CPU)
- Save weights to `models/translate/best.pt`
- Run validation on the test split
- Export to ONNX

### Training Parameters

| Parameter | Value | Notes |
|---|---|---|
| Model | YOLOv8s (small) | Good balance of speed/accuracy |
| Epochs | 100 (GPU) / 50 (CPU) | |
| Image size | 640x640 | |
| Batch size | 16 (GPU) / 8 (CPU) | |
| Early stopping | 20 epochs patience | Stops if no improvement |
| Pretrained | Yes (COCO weights) | Transfer learning |

### Monitor Training

```python
# In a notebook or after training, plot results:
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("models/translate/results.csv")
df.columns = df.columns.str.strip()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].plot(df["epoch"], df["train/box_loss"], label="train")
axes[0].plot(df["epoch"], df["val/box_loss"], label="val")
axes[0].set_title("Box Loss")
axes[0].legend()

axes[1].plot(df["epoch"], df["metrics/precision(B)"], label="precision")
axes[1].plot(df["epoch"], df["metrics/recall(B)"], label="recall")
axes[1].set_title("Precision & Recall")
axes[1].legend()

axes[2].plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP50")
axes[2].plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP50-95")
axes[2].set_title("mAP")
axes[2].legend()

plt.tight_layout()
plt.show()
```

## 3. Execute / Use the Model

### Obstacle Detection

```python
from src.translate import ObstacleDetector

detector = ObstacleDetector()  # loads models/translate/best.pt

# Run on a camera frame (BGR numpy array from OpenCV)
import cv2
frame = cv2.imread("test_image.jpg")
detections = detector.detect(frame)

for det in detections:
    print(f"{det.class_name}: {det.confidence:.2f} at {det.bbox}")

# Filter only obstacles that block transit
obstacles = detector.filter_transit_obstacles(detections)

# Annotate frame with bounding boxes
annotated = detector.annotate(frame, detections)
cv2.imwrite("output.jpg", annotated)
```

### Multi-Lawn Path Planning

```python
from src.translate import ObstacleDetector, MultiLawnPlanner

detector = ObstacleDetector()
planner = MultiLawnPlanner(detector)

# Define lawn areas
planner.add_lawn("front-yard", center=(0, 0), radius=10.0)
planner.add_lawn("back-yard", center=(0, 25), radius=8.0)
planner.add_lawn("side-garden", center=(15, 12), radius=6.0)

# Connect lawns
planner.add_transition("front-yard", "back-yard")
planner.add_transition("front-yard", "side-garden")

# Plan path with obstacle detection
frame = cv2.imread("camera_view.jpg")
path = planner.plan_transit("front-yard", "back-yard", frame=frame)

if path and path.obstacle_free:
    for wp in path.waypoints:
        print(f"Navigate to: {wp.position}")
else:
    print(f"Path blocked by: {path.blocked_by}")
```

## 4. Available Classes

The model detects 25 obstacle classes:

**Vehicles:** Car, Bus, Truck, Motorcycle, Bike
**People:** Person, Dog
**Structures:** Building, Tree, Stairs, Road
**Street furniture:** Manhole, Guard rail, Pedestrian crosswalk, Dustbin, Bench, Chair, Plant Pot, Electrical Pole, Electrical Box, Bicycle Rack, Traffic Cone, Traffic Barrel, Traffic Sign, Fire Hydrant

### Class Groups (in config.py)

- `TRANSIT_OBSTACLE_CLASSES` — objects that block the robot path during inter-lawn transit
- `NAVIGABLE_CLASSES` — terrain the robot can cross (Road, Pedestrian crosswalk, Manhole)

## File Structure

```
src/translate/
├── __init__.py      # Module exports
├── config.py        # Constants, paths, class definitions
├── detect.py        # ObstacleDetector class (YOLOv8s wrapper)
├── train.py         # Training script
├── planner.py       # MultiLawnPlanner class (BFS path planning)
└── INSTRUCTIONS.md  # This file
```

## Troubleshooting

**"Weights not found" error:**
Make sure `models/translate/best.pt` exists after training.

**CUDA out of memory:**
Reduce batch size in `train.py` (e.g., `batch=8` or `batch=4`).

**Low mAP after training:**
- Increase epochs (try 200)
- Check dataset: `!find datasets/rod -name "*.txt" | wc -l` should show ~24k label files
- Verify data.yaml paths are correct

**Slow inference on CPU:**
Normal — YOLOv8s takes ~30-50ms per frame on CPU, which is fine for real-time (30fps = 33ms budget).
