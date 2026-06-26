"""YOLOv8s obstacle detector for real-world navigation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import (
    CONFIDENCE_THRESHOLD,
    DEFAULT_WEIGHTS,
    IMAGE_SIZE,
    IOU_THRESHOLD,
    NAVIGABLE_CLASSES,
    ROD_CLASSES,
    TRANSIT_OBSTACLE_CLASSES,
)

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """A single obstacle detection."""

    class_name: str
    class_id: int
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels
    center: tuple[int, int]  # cx, cy in pixels
    area: int  # bounding box area in pixels

    @property
    def is_transit_obstacle(self) -> bool:
        return self.class_name in TRANSIT_OBSTACLE_CLASSES

    @property
    def is_navigable(self) -> bool:
        return self.class_name in NAVIGABLE_CLASSES


class ObstacleDetector:
    """Wraps a YOLOv8s model for obstacle detection on camera frames.

    Usage::

        detector = ObstacleDetector()               # loads default weights
        detections = detector.detect(camera_frame)  # np.ndarray (BGR)
        obstacles = detector.filter_transit_obstacles(detections)
    """

    def __init__(
        self,
        weights_path: str | Path | None = None,
        confidence: float = CONFIDENCE_THRESHOLD,
        iou: float = IOU_THRESHOLD,
        device: str | None = None,
    ) -> None:
        from ultralytics import YOLO

        self.confidence = confidence
        self.iou = iou

        if weights_path is None:
            weights_path = DEFAULT_WEIGHTS
        weights_path = Path(weights_path)

        if not weights_path.exists():
            raise FileNotFoundError(
                f"Weights not found at {weights_path}. "
                "Train a model first with: uv run python -m src.translate.train"
            )

        logger.info("Loading YOLOv8s weights from %s", weights_path)
        self.model = YOLO(str(weights_path))
        self._device = device

    def detect(
        self,
        frame: np.ndarray,
        *,
        classes: list[int] | None = None,
    ) -> list[Detection]:
        """Run inference on a single BGR frame and return structured detections."""
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou,
            imgsz=IMAGE_SIZE,
            classes=classes,
            device=self._device,
            verbose=False,
        )

        detections: list[Detection] = []
        if not results:
            return detections

        result = results[0]
        if result.boxes is None:
            return detections

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = (
                ROD_CLASSES[cls_id] if cls_id < len(ROD_CLASSES) else f"cls_{cls_id}"
            )

            detections.append(
                Detection(
                    class_name=class_name,
                    class_id=cls_id,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    center=((x1 + x2) // 2, (y1 + y2) // 2),
                    area=(x2 - x1) * (y2 - y1),
                )
            )

        logger.debug("Detected %d objects", len(detections))
        return detections

    def filter_transit_obstacles(self, detections: list[Detection]) -> list[Detection]:
        """Return only detections that are obstacles for transit navigation."""
        return [d for d in detections if d.is_transit_obstacle]

    def filter_navigable(self, detections: list[Detection]) -> list[Detection]:
        """Return only detections that indicate navigable terrain."""
        return [d for d in detections if d.is_navigable]

    def annotate(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        *,
        show_conf: bool = True,
    ) -> np.ndarray:
        """Draw bounding boxes and labels on a copy of the frame."""
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = (0, 255, 0) if det.is_navigable else (0, 0, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = det.class_name
            if show_conf:
                label += f" {det.confidence:.2f}"
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
        return annotated
