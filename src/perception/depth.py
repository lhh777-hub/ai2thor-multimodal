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


def _level_from_metres(metres: float) -> str:
    if metres <= _NEAR_LIMIT:
        return "NEAR"
    elif metres <= _FAR_LIMIT:
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

    def __init__(self, assumed_size: float = 0.4,
                 near_limit: float = _NEAR_LIMIT,
                 far_limit: float = _FAR_LIMIT,
                 fov_deg: float = 90.0):
        self.assumed_size = assumed_size
        self.near_limit = near_limit
        self.far_limit = far_limit
        self.fov_deg = fov_deg

    def estimate(self, rgb: np.ndarray, detection: Detection) -> tuple[str, float]:
        metres = self._estimate_metres(rgb, detection)
        level = _level_from_metres(metres)
        logger.debug("%s: %.2f m → %s", detection.label, metres, level)
        return level, metres

    def _estimate_metres(self, rgb: np.ndarray, detection: Detection) -> float:
        """Pinhole-camera distance from bbox height.

            distance = (real_size × focal_px) / bbox_height_px

        *focal_px* is derived from the vertical FOV and image height.
        """
        img_h = rgb.shape[0]
        bbox_h = detection.bbox.height
        if bbox_h <= 0:
            return 0.0

        fov_rad = math.radians(self.fov_deg)
        focal_px = (img_h / 2.0) / math.tan(fov_rad / 2.0)
        metres = (self.assumed_size * focal_px) / bbox_h
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
