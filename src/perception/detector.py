"""
YOLOv8 object detector.

Usage::

    detector = YOLODetector(model="yolov8n.pt", confidence=0.3)
    detections = detector.detect(rgb_image)
    for d in detections:
        print(d.label, d.confidence, d.screen_position)
"""

import numpy as np

from src.common.types import BBox, Detection
from src.common.logger import setup_logger

logger = setup_logger("detector")

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False
    YOLO = None  # type: ignore

# Legacy alias — prefer class_config.load_config() instead.
# Kept for backward-compat with any code importing this directly.
def _load_default_config():
    from src.perception.class_config import load_config
    cfg = load_config()
    # Build reverse mapping: yolo_name → thor_name
    result: dict[str, str] = {}
    for thor, yolo in cfg.thor_to_names.items():
        result[yolo.lower()] = thor
    return result

COCO_TO_THOR: dict[str, str] = {}  # populated lazily below

def _init_coco_to_thor():
    global COCO_TO_THOR
    if not COCO_TO_THOR:
        try:
            COCO_TO_THOR.update(_load_default_config())
        except Exception:
            pass

_init_coco_to_thor()


class YOLODetector:
    """Wrapper around a YOLOv8 model for indoor object detection."""

    # Known model variants and their sizes
    MODELS = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]

    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.3,
                 iou_threshold: float = 0.45):
        if not _YOLO_AVAILABLE:
            raise ImportError(
                "ultralytics is not installed.  Run:  pip install ultralytics\n"
                "Then download the model: the first call will auto-download yolov8n.pt"
            )
        self.model = YOLO(model_name)
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        logger.info("YOLO detector loaded: %s (conf=%.2f, iou=%.2f)",
                     model_name, confidence, iou_threshold)

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        """Run detection on an RGB image (H x W x 3, uint8).

        Returns a list of Detection objects sorted by confidence (high first).
        """
        results = self.model(rgb, conf=self.confidence, iou=self.iou_threshold,
                             verbose=False)
        detections: list[Detection] = []

        img_h, img_w = rgb.shape[:2]

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                label = result.names[int(box.cls[0])]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections.append(Detection(
                    label=label,
                    bbox=BBox(x1=float(x1), y1=float(y1),
                              x2=float(x2), y2=float(y2)),
                    confidence=conf,
                    screen_position=self._screen_pos(x1, x2, img_w),
                ))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    @staticmethod
    def _screen_pos(x1: float, x2: float, img_w: int) -> str:
        cx = (x1 + x2) / 2
        if cx < img_w / 3:
            return "left"
        elif cx < 2 * img_w / 3:
            return "center"
        return "right"
