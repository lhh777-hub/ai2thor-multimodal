"""
YOLOv8 / YOLO-World object detectors.

Usage::

    # Standard YOLO (closed-set, needs fine-tuning for custom classes)
    detector = YOLODetector(model="yolov8n.pt", confidence=0.3)
    detections = detector.detect(rgb_image)

    # YOLO-World (open-vocabulary, no fine-tuning needed)
    detector = YOLOWorldDetector(model="yolov8s-worldv2.pt", confidence=0.2)
    detector.set_classes(["chair", "refrigerator", "door", "window"])
    detections = detector.detect(rgb_image)

Reference:
    Cheng et al., "YOLO-World: Real-Time Open-Vocabulary Object Detection", CVPR 2024.
    https://arxiv.org/abs/2401.17270
"""

import numpy as np

from src.common.types import BBox, Detection
from src.common.logger import setup_logger
from src.common.utils import screen_pos, box_iou

logger = setup_logger("detector")

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False
    YOLO = None  # type: ignore

try:
    from ultralytics import YOLOWorld
    _YOLO_WORLD_AVAILABLE = True
except ImportError:
    _YOLO_WORLD_AVAILABLE = False
    YOLOWorld = None  # type: ignore


# Known YOLO-World model variants (will auto-download on first use)
YOLO_WORLD_MODELS = [
    "yolov8s-world.pt", "yolov8s-worldv2.pt",
    "yolov8m-world.pt", "yolov8m-worldv2.pt",
    "yolov8l-world.pt", "yolov8l-worldv2.pt",
    "yolov8x-world.pt", "yolov8x-worldv2.pt",
]


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
                    screen_position=screen_pos(x1, x2, img_w),
                ))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections


# ---------------------------------------------------------------------------
# YOLO-World — open-vocabulary detection (CVPR 2024)
# ---------------------------------------------------------------------------

class YOLOWorldDetector:
    """Open-vocabulary object detector using YOLO-World.

    Unlike ``YOLODetector`` (closed-set, 80 COCO classes), this detector
    accepts an arbitrary list of class names at runtime via ``set_classes()``.
    No fine-tuning or data collection is needed to detect new classes.

    Reference:
        Cheng, Song, Ge, Liu, Wang, Shan.
        "YOLO-World: Real-Time Open-Vocabulary Object Detection." CVPR 2024.

    Usage::

        detector = YOLOWorldDetector(model="yolov8s-worldv2.pt", confidence=0.2)
        detector.set_classes(["chair", "refrigerator", "door", "cabinet"])
        dets = detector.detect(rgb_image)
    """

    def __init__(self, model_name: str = "yolov8s-worldv2.pt",
                 confidence: float = 0.2, iou_threshold: float = 0.45):
        if not _YOLO_WORLD_AVAILABLE:
            raise ImportError(
                "YOLOWorld requires ultralytics>=8.1.0.  Run:  pip install -U ultralytics\n"
                "The model will auto-download on first use (~200 MB for yolov8s-worldv2.pt)."
            )
        if model_name not in YOLO_WORLD_MODELS:
            logger.warning("Unknown YOLO-World variant: %s (expected one of %s)",
                           model_name, YOLO_WORLD_MODELS)
        self.model = YOLOWorld(model_name)
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self._classes: list[str] = []
        self._num_classes = 0
        logger.info("YOLO-World loaded: %s (conf=%.2f, iou=%.2f)",
                     model_name, confidence, iou_threshold)

    # ------------------------------------------------------------------
    # Class configuration
    # ------------------------------------------------------------------

    def set_classes(self, class_names: list[str]) -> None:
        """Set the vocabulary of classes to detect.

        This is the key advantage of YOLO-World: you can change what the
        model detects without any re-training.  The CLIP text encoder
        produces embeddings for each class name, and the detection head
        uses those embeddings instead of fixed class weights.

        Call this once after construction (or re-call to switch vocabulary).
        """
        if not class_names:
            logger.warning("set_classes() called with empty list")
            return
        self._classes = list(class_names)
        self._num_classes = len(class_names)
        self.model.set_classes(class_names)
        logger.info("YOLO-World vocabulary set: %d classes", self._num_classes)

    def set_classes_from_config(self, classes_path: str | None = None,
                                 extra: list[str] | None = None) -> list[str]:
        """Load class names from ``classes.yaml`` and apply them.

        This is the most common setup for AI2-THOR: read every class defined
        in the YAML config so the detector covers all in-scene objects.

        *extra* classes (e.g. ``["door", "window"]``) are appended if not
        already present.

        Returns the final list of class names.
        """
        from src.perception.class_config import load_config
        cfg = load_config(classes_path)
        names = list(cfg.class_names)
        if extra:
            for e in extra:
                if e not in names:
                    names.append(e)
        self.set_classes(names)
        return names

    def set_classes_from_scene(self, controller, extra: list[str] | None = None) -> list[str]:
        """Narrow the vocabulary to only objects present in the current AI2-THOR scene.

        Queries ``controller.get_object_map()``, maps each AI2-THOR object type
        through ``classes.yaml``, and sets the YOLO-World vocabulary to exactly
        those labels.  This eliminates false positives for objects that don't
        exist in the scene (e.g. no "toilet" detections in a kitchen).

        *extra* labels (e.g. ``["door", "window", "wall"]``) are always included —
        these structural elements often appear visually but aren't listed in
        the AI2-THOR object map.

        Returns the final list of class names.
        """
        from src.perception.class_config import load_config
        cfg = load_config()

        # Collect all object types present in the scene
        obj_map = controller.get_object_map()
        labels: set[str] = set()

        for thor_type in obj_map:
            cls_id = cfg.thor_to_class_id(thor_type)
            if cls_id is not None:
                labels.add(cfg.class_names[cls_id])
            else:
                # Unmapped AI2-THOR type — try using it directly
                labels.add(thor_type.lower().replace(" ", "_"))

        # Always include structural objects not in the object map
        structural = {"door", "window", "wall", "floor", "ceiling",
                       "cabinet", "counter", "drawer", "shelf", "light"}
        labels.update(structural)
        if extra:
            labels.update(extra)

        names = sorted(labels)
        self.set_classes(names)
        logger.info("YOLO-World scene vocabulary: %d classes (from %d scene objects)",
                     len(names), len(obj_map))
        return names

    @property
    def classes(self) -> list[str]:
        return list(self._classes)

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        """Run open-vocabulary detection on an RGB image.

        Returns a list of Detection objects sorted by confidence (high first).
        If ``set_classes()`` was not called yet, returns an empty list.
        """
        if self._num_classes == 0:
            logger.warning("YOLO-World: set_classes() not called — no classes to detect")
            return []

        results = self.model.predict(rgb, conf=self.confidence,
                                      iou=self.iou_threshold, verbose=False)
        detections: list[Detection] = []
        img_h, img_w = rgb.shape[:2]

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id >= len(self._classes):
                    continue
                label = self._classes[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections.append(Detection(
                    label=label,
                    bbox=BBox(x1=float(x1), y1=float(y1),
                              x2=float(x2), y2=float(y2)),
                    confidence=conf,
                    screen_position=screen_pos(x1, x2, img_w),
                ))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections


# ---------------------------------------------------------------------------
# Hybrid detector — YOLO-finetuned + YOLO-World
# ---------------------------------------------------------------------------

class HybridDetector:
    """Combine fine-tuned YOLO (high precision on trained classes) with
    YOLO-World (open-vocabulary coverage for unseen objects).

    Both detectors run on every frame.  The merge strategy is **per-class
    gating**: for any class that the fine-tuned model detects in this frame,
    YOLO-World detections of the SAME class are suppressed entirely.
    YOLO-World only contributes detections for classes the fine-tuned model
    cannot see — typically structural objects (door, window, mirror) that
    were not in the training set.

    Usage::

        yolo_f = YOLODetector("runs/.../best.pt", confidence=0.3)
        yolo_w = YOLOWorldDetector("yolov8s-worldv2.pt", confidence=0.15)
        yolo_w.set_classes_from_scene(controller)
        hybrid = HybridDetector(yolo_f, yolo_w)
        dets = hybrid.detect(rgb)
    """

    def __init__(self, finetuned: YOLODetector, world: YOLOWorldDetector,
                 merge_iou: float = 0.5):
        self.finetuned = finetuned
        self.world = world
        self.merge_iou = merge_iou

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        """Run both detectors and merge with same-class IoU gating.

        Strategy:
        1. Run both detectors, keep all fine-tuned detections.
        2. For each YOLO-World detection, check if it overlaps (IoU ≥ merge_iou)
           with a fine-tuned detection of the **same class**.  If so, the
           fine-tuned version takes priority (it's more reliable on its trained
           classes) and the YOLO-World version is discarded.
        3. YOLO-World detections of classes the fine-tuned model can't detect
           (door, window, mirror, etc.) are always kept — unless they spatially
           overlap a fine-tuned detection of a different class (cross-class NMS).

        This preserves YOLO-World's ability to find additional instances of a
        class that fine-tuned YOLO partially detected (e.g. a second chair that
        fine-tuned missed).
        """
        dets_f = self.finetuned.detect(rgb)
        dets_w = self.world.detect(rgb)

        if not dets_w:
            return dets_f
        if not dets_f:
            return dets_w

        # Group fine-tuned boxes by class for same-class IoU check
        f_by_class: dict[str, list[tuple[float, float, float, float]]] = {}
        for d in dets_f:
            f_by_class.setdefault(d.label.lower(), []).append(
                (d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2))

        # All fine-tuned boxes for cross-class IoU fallback
        f_all = [(d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2) for d in dets_f]

        merged = list(dets_f)
        for d in dets_w:
            wb = (d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2)
            w_cls = d.label.lower()
            keep = True

            # Gate 1: same-class overlap → fine-tuned wins
            if w_cls in f_by_class:
                for fb in f_by_class[w_cls]:
                    if box_iou(wb, fb) >= self.merge_iou:
                        keep = False
                        break

            # Gate 2: cross-class spatial overlap (e.g. YOLO-W "cabinet"
            # at same spot as fine-tuned "refrigerator")
            if keep:
                for fb in f_all:
                    if box_iou(wb, fb) >= self.merge_iou:
                        keep = False
                        break

            if keep:
                merged.append(d)

        merged.sort(key=lambda d: d.confidence, reverse=True)
        return merged



# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_hybrid(
    finetuned_model: str = "runs/detect/runs/train/weights/best.pt",
    finetuned_conf: float = 0.3,
    world_model: str = "yolov8s-worldv2.pt",
    world_conf: float = 0.15,
    controller=None,
    merge_iou: float = 0.5,
) -> HybridDetector:
    """Create a ready-to-use HybridDetector.

    If *controller* is provided, the YOLO-World vocabulary is automatically
    narrowed to objects present in the current scene.
    """
    yolo_f = YOLODetector(model_name=finetuned_model, confidence=finetuned_conf)
    yolo_w = YOLOWorldDetector(model_name=world_model, confidence=world_conf)
    if controller is not None:
        yolo_w.set_classes_from_scene(controller)
    else:
        yolo_w.set_classes_from_config()
    return HybridDetector(yolo_f, yolo_w, merge_iou=merge_iou)

def create_detector(model_name: str, confidence: float = 0.3,
                    iou_threshold: float = 0.45,
                    classes_path: str | None = None,
                    extra_classes: list[str] | None = None):
    """Factory that returns the right detector based on the model name.

    - If *model_name* contains ``"world"`` → ``YOLOWorldDetector``
    - Otherwise → ``YOLODetector``

    For YOLO-World, if *classes_path* is provided the vocabulary is
    auto-populated from that YAML file.
    """
    if "world" in model_name.lower():
        det = YOLOWorldDetector(model_name=model_name, confidence=confidence,
                                 iou_threshold=iou_threshold)
        # Always set a vocabulary — YOLO-World returns nothing without it.
        # Default to the project's class config so the detector works out of the box.
        det.set_classes_from_config(classes_path or "config/classes.yaml",
                                     extra=extra_classes)
        return det
    return YOLODetector(model_name=model_name, confidence=confidence,
                        iou_threshold=iou_threshold)
