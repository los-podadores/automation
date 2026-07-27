# YOLOv8s Obstacle Detection — Setup & Usage

## Overview

This module detects real-world obstacles for multi-lawn navigation using YOLOv8s trained on the [ROD-Dataset](https://www.kaggle.com/datasets/abtinzandi/obstacle-detection-dataset) (25 obstacle classes, 24k+ images).

## Prerequisites

```bash
direnv allow   # activates Nix flake with SDL2, uv, ruff
uv sync        # install Python dependencies (includes kagglehub, ultralytics)
```

You need a Kaggle account and API token. Set it up once:

```bash
# Download your API token from kaggle.com → Settings → API → Create New Token
# Then either:
#   Option A: Place kaggle.json at ~/.kaggle/kaggle.json
#   Option B: Export the token
export KAGGLE_API_TOKEN=your_token_here
```

## 1. Download the Dataset

```bash
uv run python src/translate/download_dataset.py abtinzandi/obstacle-detection-dataset
```

This downloads the ROD-Dataset to `datasets/rod/` using `kagglehub`.

After download, the structure should be:
```
datasets/rod/
└── ROD-Dataset/
    └── dataset/
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
- Find `data.yaml` inside the downloaded dataset (exits with error if not found)
- Train YOLOv8s for 100 epochs on GPU (50 on CPU)
- Save training outputs to `models/translate/`
- Copy `best.pt` to `models/translate/best.pt` (the working location)
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

In a separate terminal, while training is running:

```bash
uv run python -m src.translate.visualize
```

Opens TensorBoard at http://localhost:6006 with live loss curves, metrics, and learning rate.

## 3. Run the Demo

```bash
uv run python -m src.translate.demo              # 5 random test images
uv run python -m src.translate.demo -n 10        # more images
uv run python -m src.translate.demo -i photo.jpg # specific image
uv run python -m src.translate.demo --no-planner  # skip planner demo
```

Annotated images are saved to `runs/translate/demo/`.

## File Structure

```
src/translate/
├── __init__.py          # Module exports (ObstacleDetector, MultiLawnPlanner)
├── config.py            # Constants, paths, class definitions
├── detect.py            # ObstacleDetector class (YOLOv8s wrapper)
├── planner.py           # MultiLawnPlanner class (BFS path planning)
├── train.py             # Training script
├── download_dataset.py  # Dataset download via kagglehub
├── visualize.py         # Launch TensorBoard for live training monitoring
├── demo.py              # CLI demo script
└── INSTRUCTIONS.md      # This file
```

## Troubleshooting

**"data.yaml not found" during training:**
Run the download script first:
```bash
uv run python src/translate/download_dataset.py abtinzandi/obstacle-detection-dataset
```

**"Weights not found" during demo/inference:**
Train a model first (see step 2), or ensure `models/translate/best.pt` exists.

**CUDA out of memory:**
Reduce batch size in `train.py` (e.g., `batch=8` or `batch=4`).

**Low mAP after training:**
- Increase epochs (try 200)
- Check dataset: the download should produce ~24k label files
- Verify `data.yaml` paths are correct inside the dataset

**Slow inference on CPU:**
Normal — YOLOv8s takes ~30-50ms per frame on CPU, which is fine for real-time (30fps = 33ms budget).
