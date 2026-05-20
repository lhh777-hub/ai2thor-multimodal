"""
Perception pipeline — detection + depth + optional CLIP verification + scene prior.

Usage::

    pipeline = PerceptionPipeline(detector, depth, verifier=clip, prior=scene_prior)
    detections = pipeline.process(rgb_image)
"""

from __future__ import annotations

import numpy as np

from src.common.types import Detection
from src.perception.detector import YOLODetector
from src.perception.depth import DepthEstimator
from src.common.logger import setup_logger

logger = setup_logger("perception")


class PerceptionPipeline:
    """Detect objects, estimate distances, optionally verify with CLIP."""

    def __init__(self, detector: YOLODetector, depth_estimator: DepthEstimator,
                 verifier=None, prior=None):
        self.detector = detector
        self.depth = depth_estimator
        self.verifier = verifier   # CLIPVerifier or None
        self.prior = prior         # ScenePrior or None

    def process(self, rgb: np.ndarray,
                clip_prompt: str = "indoor objects",
                clip_threshold: float = 0.0,
                ) -> list[Detection]:
        """Run the full perception pipeline on *rgb*.

        Steps:
        1. YOLO detection
        2. Scene prior weighting (if configured)
        3. CLIP verification + filtering (if configured)
        4. Distance estimation

        Returns the final list, sorted by confidence (high first, after prior).
        """
        # 1. Detect
        detections = self.detector.detect(rgb)

        # 2. Scene prior
        if self.prior is not None:
            detections = self.prior.apply(detections)
            detections.sort(key=lambda d: d.confidence, reverse=True)

        # 3. CLIP verification
        if self.verifier is not None:
            detections = self.verifier.verify(rgb, detections, clip_prompt)
            if clip_threshold > 0:
                from src.perception.verifier import filter_by_clip
                detections = filter_by_clip(detections, clip_threshold)

        # 4. Distance
        for d in detections:
            d.distance_level = self.depth.estimate(rgb, d)

        return detections
