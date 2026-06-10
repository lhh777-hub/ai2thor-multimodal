"""
Distance estimation from monocular RGB.

Two strategies:
- ``HeuristicDepth`` — bbox area ratio + pinhole-camera distance estimate
- ``DepthAnythingV2`` — monocular depth model (more accurate, requires GPU + extra deps)

Each ``estimate()`` returns ``(level, metres)`` where *level* is one of
``"NEAR"`` / ``"MEDIUM"`` / ``"FAR"`` and *metres* is the estimated distance
(or 0.0 when unavailable).
"""

import math
from abc import ABC, abstractmethod

import numpy as np

from src.common.types import Detection
from src.common.logger import setup_logger

logger = setup_logger("depth")

# Default distance thresholds (metres)
_NEAR_LIMIT = 1.5   # ≤ 1.5 m  → NEAR
_FAR_LIMIT  = 3.0   # ≥ 3.0 m  → FAR  (1.5–3.0 → MEDIUM)


# Typical real-world sizes in metres (longest dimension).
# HeuristicDepth uses these per-class instead of a single assumed_size,
# which dramatically improves distance accuracy.
_OBJECT_SIZES: dict[str, float] = {
    # Large furniture / appliances (≥ 0.8 m)
    "refrigerator": 1.8, "fridge": 1.8,
    "bed": 2.0,
    "couch": 2.0, "sofa": 2.0,
    "dining table": 1.5,
    "bathtub": 1.2,
    "wardrobe": 1.0,
    "oven": 0.9,
    "tv": 0.9, "television": 0.9,
    "bookshelf": 0.9,
    "cabinet": 0.8,
    "shelf": 0.8,
    "curtain": 0.7,
    "sink": 0.7,
    "mirror": 0.6,
    "toilet": 0.6,
    "suitcase": 0.6,
    "counter": 0.8,
    "countertop": 0.8,

    # Medium objects (0.3 – 0.7 m)
    "chair": 0.5, "armchair": 0.6,
    "microwave": 0.4,
    "laptop": 0.35,
    "keyboard": 0.4,
    "backpack": 0.4,
    "potted plant": 0.5,
    "vase": 0.3,
    "teddy bear": 0.3,
    "sports ball": 0.22,
    "handbag": 0.3,
    "umbrella": 0.4,
    "tennis racket": 0.5,
    "baseball bat": 0.7,
    "skateboard": 0.6,
    "surfboard": 1.5,
    "broom": 0.9,
    "mop": 0.9,
    "vacuum": 0.4,
    "trash can": 0.4,
    "garbage bin": 0.4,
    "pillow": 0.4,
    "towel": 0.5,
    "lamp": 0.4,
    "toaster": 0.25,
    "stool": 0.35,

    # Small objects (< 0.3 m)
    "cup": 0.1, "mug": 0.1,
    "bottle": 0.2, "wine bottle": 0.3,
    "bowl": 0.15,
    "book": 0.2,
    "cell phone": 0.08, "phone": 0.08,
    "apple": 0.08,
    "banana": 0.15,
    "orange": 0.08,
    "sandwich": 0.15,
    "donut": 0.08,
    "cake": 0.2,
    "carrot": 0.12,
    "hot dog": 0.12,
    "pizza": 0.3,
    "knife": 0.25,
    "spoon": 0.18,
    "fork": 0.18,
    "wine glass": 0.2,
    "mouse": 0.06,
    "remote": 0.15,
    "clock": 0.2,
    "scissors": 0.18,
    "pen": 0.12,
    "candle": 0.1,
    "soap": 0.08,
    "sponge": 0.1,
    "credit card": 0.06,
    "paper": 0.3,
    "newspaper": 0.35,
    "plate": 0.25,
    "pan": 0.3,
    "pot": 0.3, "cooking pot": 0.3,
    "kettle": 0.25,
    "teapot": 0.2,
    "box": 0.3,
    "tissue box": 0.15,
    "shoe": 0.25,
    "glasses": 0.12,
    "watch": 0.04,
    "key": 0.05,
    "hair drier": 0.2,
    "toothbrush": 0.15,
}


def _level_from_metres(metres: float, near_limit: float = _NEAR_LIMIT,
                       far_limit: float = _FAR_LIMIT) -> str:
    if metres <= near_limit:
        return "NEAR"
    elif metres <= far_limit:
        return "MEDIUM"
    return "FAR"


# ---------------------------------------------------------------------------
# Abstract
# ---------------------------------------------------------------------------

class DepthEstimator(ABC):
    """Abstract interface for distance estimation."""

    @abstractmethod
    def estimate(self, rgb: np.ndarray, detection: Detection) -> tuple[str, float]:
        """Return ``(level, metres)`` for *detection*."""
        ...


# ---------------------------------------------------------------------------
# Heuristic (bbox area ratio + pinhole model)
# ---------------------------------------------------------------------------

class HeuristicDepth(DepthEstimator):
    """Estimate distance from the fraction of the image a bounding box covers.

    Uses a pinhole-camera model to produce a rough metric estimate, then
    thresholds it to NEAR / MEDIUM / FAR.  Assumes a typical indoor-object
    size of *assumed_size* metres.
    """

    def __init__(self, default_size: float = 0.4,
                 near_limit: float = _NEAR_LIMIT,
                 far_limit: float = _FAR_LIMIT,
                 fov_deg: float = 90.0):
        self.default_size = default_size
        self.near_limit = near_limit
        self.far_limit = far_limit
        self.fov_deg = fov_deg

    def estimate(self, rgb: np.ndarray, detection: Detection) -> tuple[str, float]:
        metres = self._estimate_metres(rgb, detection)
        level = _level_from_metres(metres, self.near_limit, self.far_limit)
        logger.debug("%s: %.2f m (size=%.2f) → %s",
                     detection.label, metres,
                     _OBJECT_SIZES.get(detection.label, self.default_size),
                     level)
        return level, metres

    def _estimate_metres(self, rgb: np.ndarray, detection: Detection) -> float:
        """Pinhole-camera distance from bbox height.

            distance = (real_size × focal_px) / bbox_height_px

        Uses a per-class assumed size when available; falls back to
        *default_size* for unknown classes.
        """
        img_h = rgb.shape[0]
        bbox_h = detection.bbox.height
        if bbox_h <= 0:
            return 0.0

        # Per-class object size, with case-insensitive fallback
        assumed = _OBJECT_SIZES.get(detection.label)
        if assumed is None:
            assumed = _OBJECT_SIZES.get(detection.label.lower(), self.default_size)

        fov_rad = math.radians(self.fov_deg)
        focal_px = (img_h / 2.0) / math.tan(fov_rad / 2.0)
        metres = (assumed * focal_px) / bbox_h
        return round(float(metres), 2)


# ---------------------------------------------------------------------------
# Depth Anything V2  (optional, lazy-load)
# ---------------------------------------------------------------------------

class DepthAnythingV2(DepthEstimator):
    """Monocular depth via Depth Anything V2.

    Lazy-loads the model on first call.  Falls back to ``HeuristicDepth`` if
    the model cannot be loaded (no GPU, missing package, etc.).
    """

    def __init__(self, near_threshold: float = 0.4, far_threshold: float = 0.7,
                 model_name: str = "depth_anything_v2_vits"):
        self.near_threshold = near_threshold
        self.far_threshold = far_threshold
        self.model_name = model_name
        self._model = None
        self._fallback = HeuristicDepth()
        self._tried_load = False

    def estimate(self, rgb: np.ndarray, detection: Detection) -> tuple[str, float]:
        self._try_load()
        if self._model is None:
            return self._fallback.estimate(rgb, detection)

        import torch
        with torch.no_grad():
            depth_map = self._model.infer_image(rgb)

        # Average depth inside the bbox
        x1 = max(0, int(detection.bbox.x1))
        y1 = max(0, int(detection.bbox.y1))
        x2 = min(depth_map.shape[1], int(detection.bbox.x2))
        y2 = min(depth_map.shape[0], int(detection.bbox.y2))

        if x2 <= x1 or y2 <= y1:
            return "FAR", 0.0

        bbox_depth = depth_map[y1:y2, x1:x2]
        avg = float(np.mean(bbox_depth))
        full_mean = float(np.mean(depth_map))

        norm = avg / full_mean if full_mean > 0 else 0.5
        # Metric distance is approximate — DA V2 produces relative depth.
        # We map the normalised value to a rough metre scale (0–5 m range).
        metres = round(norm * 5.0, 2)

        if norm < self.near_threshold:
            return "NEAR", metres
        elif norm < self.far_threshold:
            return "MEDIUM", metres
        return "FAR", metres

    def _try_load(self):
        if self._tried_load:
            return
        self._tried_load = True

        try:
            import torch
            from depth_anything_v2.dpt import DepthAnythingV2 as DAV2

            if not torch.cuda.is_available():
                logger.warning("DepthAnything: no CUDA, using heuristic fallback")
                return

            model_configs = {
                "depth_anything_v2_vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
                "depth_anything_v2_vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
                "depth_anything_v2_vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
            }
            cfg = model_configs[self.model_name]
            self._model = DAV2(**cfg)
            self._model.load_state_dict(torch.hub.load_state_dict_from_url(
                f"https://huggingface.co/depth-anything/Depth-Anything-V2-{cfg['encoder'].replace('vit','ViT')}/resolve/main/{self.model_name}.pth",
                map_location="cpu",
            ))
            self._model = self._model.to("cuda").eval()
            logger.info("Depth Anything V2 loaded: %s on cuda", self.model_name)

        except Exception as e:
            logger.warning("DepthAnything load failed (%s), using heuristic fallback", e)
            self._model = None
