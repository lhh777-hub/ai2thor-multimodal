"""
Common utility functions shared across the project.

Previously these were duplicated in 3-4 files each.  This module is the
single source of truth — import from here instead of copy-pasting.

Usage::

    from src.common.utils import img_to_b64, draw_annotations, box_iou
"""

from __future__ import annotations

import base64
import os
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from src.common.types import Detection, Vec3


# ===========================================================================
# Image encoding
# ===========================================================================

def img_to_b64(rgb: np.ndarray, quality: int = 60) -> str:
    """Convert an RGB numpy array (H×W×3 uint8) to a base64 JPEG string.

    *quality* controls JPEG compression (1-100, default 60).  Lower values
    = smaller payload = faster API calls.  At quality=60, a 640×480 frame
    is ~50 KB vs ~350 KB at quality=95.
    """
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                          [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode("utf-8")


# ===========================================================================
# Annotation drawing
# ===========================================================================

def draw_annotations(rgb: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw detection bounding-boxes and labels on *rgb*.  Returns a copy.

    Colour coding: green (conf ≥ 0.5), yellow (conf ≥ 0.3), red (low).
    """
    img = rgb.copy()
    h, w = img.shape[:2]

    for d in detections:
        b = d.bbox
        x1, y1 = max(0, int(b.x1)), max(0, int(b.y1))
        x2, y2 = min(w, int(b.x2)), min(h, int(b.y2))

        # Colour by confidence
        if d.confidence >= 0.5:
            colour = (0, 255, 0)       # green
        elif d.confidence >= 0.3:
            colour = (0, 255, 255)     # yellow
        else:
            colour = (0, 0, 255)       # red

        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)

        dist_str = f"{d.distance_meters:.1f}m" if d.distance_meters else d.distance_level
        label = f"{d.label} {d.confidence:.2f} [{dist_str}]"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x1, max(y1 - th - 6, 0)),
                      (x1 + tw + 4, y1), colour, -1)
        cv2.putText(img, label, (x1 + 2, max(y1 - 4, th + 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    return img


# ===========================================================================
# Screen position
# ===========================================================================

def screen_pos(x1: float, x2: float, img_w: int) -> str:
    """Classify a bounding box's horizontal position in the image.

    Returns ``"left"``, ``"center"``, or ``"right"``.
    """
    cx = (x1 + x2) / 2.0
    if cx < img_w / 3:
        return "left"
    elif cx < 2 * img_w / 3:
        return "center"
    return "right"


# ===========================================================================
# IoU computation
# ===========================================================================

def box_iou(a: tuple[float, float, float, float],
            b: tuple[float, float, float, float]) -> float:
    """Intersection-over-Union for two (x1, y1, x2, y2) boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


# ===========================================================================
# Class name mapping
# ===========================================================================

def thor_to_yolo(thor_name: str) -> str:
    """Map an AI2-THOR object type to its YOLO class name via classes.yaml.

    Falls back to lowercased, underscore-replaced name if the config is
    unavailable.
    """
    try:
        from src.perception.class_config import load_config
        cfg = load_config()
        cid = cfg.thor_to_class_id(thor_name)
        if cid is not None:
            return cfg.class_names[cid]
    except Exception:
        pass
    return thor_name.lower().replace("_", " ")


# ===========================================================================
# Instance-level distance evaluation
# ===========================================================================

def eval_distance(ctrl, target_name: str,
                  target_pos: Vec3 | None = None) -> float | None:
    """Distance from agent to the target *instance*, for success evaluation.

    When *target_pos* (from spatial memory) is provided, measures against
    the EXACT instance the agent navigated to — avoiding false positives
    from same-type objects elsewhere in the scene.

    Falls back to ``ctrl.distance_to()`` (closest-instance query) otherwise.
    """
    if target_pos is not None:
        agent = ctrl.agent_state.position
        return round(float(np.sqrt(
            (agent.x - target_pos.x) ** 2 + (agent.z - target_pos.z) ** 2)), 2)
    return ctrl.distance_to(target_name)


# ===========================================================================
# Safe file-name sanitizer
# ===========================================================================

def safe_filename(text: str, max_len: int = 30) -> str:
    """Replace non-alphanumeric characters with underscores for file names."""
    import re
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_")[:max_len]
