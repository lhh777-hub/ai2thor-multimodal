"""
Distance estimation from monocular RGB.

Two strategies:
- ``HeuristicDepth`` — bbox area ratio (fast, no GPU, good enough for most cases)
- ``DepthAnythingV2`` — monocular depth model (more accurate, requires GPU + extra deps)
"""

from abc import ABC, abstractmethod

import numpy as np

from src.common.types import Detection
from src.common.logger import setup_logger

logger = setup_logger("depth")


class DepthEstimator(ABC):
    """Abstract interface for distance estimation."""

    @abstractmethod
    def estimate(self, rgb: np.ndarray, detection: Detection) -> str:
        """Return 'NEAR', 'MEDIUM', or 'FAR' for *detection*."""
        ...


# ---------------------------------------------------------------------------
# Heuristic (bbox area ratio)
# ---------------------------------------------------------------------------

class HeuristicDepth(DepthEstimator):
    """Estimate distance from the fraction of the image that a bounding box covers.

    A large bbox → object is NEAR; a small bbox → object is FAR.
    """

    def __init__(self, near_ratio: float = 0.15, far_ratio: float = 0.05):
        self.near_ratio = near_ratio
        self.far_ratio = far_ratio

    def estimate(self, rgb: np.ndarray, detection: Detection) -> str:
        img_area = rgb.shape[0] * rgb.shape[1]
        bbox_area = detection.bbox.width * detection.bbox.height
        ratio = bbox_area / img_area

        if ratio > self.near_ratio:
            level = "NEAR"
        elif ratio > self.far_ratio:
            level = "MEDIUM"
        else:
            level = "FAR"

        logger.debug("%s: ratio=%.4f → %s", detection.label, ratio, level)
        return level


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

    def estimate(self, rgb: np.ndarray, detection: Detection) -> str:
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
            return "FAR"

        bbox_depth = depth_map[y1:y2, x1:x2]
        avg = float(np.mean(bbox_depth))
        full_mean = float(np.mean(depth_map))

        norm = avg / full_mean if full_mean > 0 else 0.5
        if norm < self.near_threshold:
            return "NEAR"
        elif norm < self.far_threshold:
            return "MEDIUM"
        return "FAR"

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
