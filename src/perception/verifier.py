"""
CLIP-based detection verifier — re-scores YOLO detections using
vision-language similarity, helping to suppress false positives.

Usage::

    verifier = CLIPVerifier()
    verified = verifier.verify(rgb, detections, "a photo of a chair")
    # Each detection now has clip_score filled in.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from src.common.types import Detection
from src.common.logger import setup_logger

logger = setup_logger("verifier")

try:
    import clip
    import torch
    _CLIP_AVAILABLE = True
except ImportError:
    _CLIP_AVAILABLE = False


class CLIPVerifier:
    """Uses CLIP to compute text–image similarity for each detection crop."""

    def __init__(self, model_name: str = "ViT-B/32", device: str | None = None):
        if not _CLIP_AVAILABLE:
            raise ImportError("openai-clip is not installed.  Run:  pip install openai-clip")
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        logger.info("CLIP verifier loaded: %s on %s", model_name, self.device)

    def verify(self, rgb: np.ndarray, detections: list[Detection],
               prompt: str) -> list[Detection]:
        """Compute CLIP similarity of each detection crop against *prompt*.

        ``clip_score`` is filled in on every ``Detection``.
        """
        if not detections:
            return detections

        text_tokens = clip.tokenize([prompt]).to(self.device)
        with torch.no_grad():
            text_feat = self.model.encode_text(text_tokens)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        h, w = rgb.shape[:2]
        for det in detections:
            crop = self._crop(rgb, det)
            if crop is None:
                det.clip_score = 0.0
                continue

            img_input = self.preprocess(crop).unsqueeze(0).to(self.device)
            with torch.no_grad():
                img_feat = self.model.encode_image(img_input)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                det.clip_score = float((img_feat @ text_feat.T).squeeze())

        return detections

    def _crop(self, rgb: np.ndarray, det: Detection):
        b = det.bbox
        h, w = rgb.shape[:2]
        x1, y1 = max(0, int(b.x1)), max(0, int(b.y1))
        x2, y2 = min(w, int(b.x2)), min(h, int(b.y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return Image.fromarray(rgb[y1:y2, x1:x2])


def filter_by_clip(detections: list[Detection], threshold: float = 0.20) -> list[Detection]:
    """Keep only detections with ``clip_score >= threshold``."""
    return [d for d in detections if d.clip_score >= threshold]


def rerank_by_clip(detections: list[Detection]) -> list[Detection]:
    """Re-sort by clip_score descending."""
    return sorted(detections, key=lambda d: d.clip_score, reverse=True)
