"""
Perception pipeline — detection + depth + optional CLIP verification + scene prior.

Usage::

    pipeline = PerceptionPipeline(detector, depth, verifier=clip, prior=scene_prior)
    detections = pipeline.process(rgb_image)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.common.types import Detection
from src.perception.detector import YOLODetector
from src.perception.depth import DepthEstimator, _level_from_metres
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
                controller: Any | None = None,
                ) -> list[Detection]:
        """Run the full perception pipeline on *rgb*.

        Steps:
        1. YOLO detection
        2. Scene prior weighting (if configured)
        3. CLIP verification + filtering (if configured)
        4. Distance estimation (heuristic / depth-model)
        5. Ground-truth distance override (if *controller* is provided)

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

        # 4. Distance (heuristic or depth-model)
        for d in detections:
            level, metres = self.depth.estimate(rgb, d)
            d.distance_level = level
            d.distance_meters = metres

        # 5. Ground-truth override from AI2-THOR (exact, when available)
        if controller is not None:
            self._apply_ground_truth_distance(detections, controller)

        return detections

    # ------------------------------------------------------------------
    # Ground-truth distance (AI2-THOR)
    # ------------------------------------------------------------------

    def _apply_ground_truth_distance(self, detections: list[Detection],
                                     controller: Any) -> None:
        """Override ``distance_meters`` and ``distance_level`` with exact
        AI2-THOR object positions.

        Matches each detection label to the closest scene object of the
        corresponding type via ``ThorController.distance_to()``.
        """
        for d in detections:
            metres = self._match_and_query(d.label, controller)
            if metres is not None:
                d.distance_meters = metres
                d.distance_level = _level_from_metres(metres)

    @staticmethod
    def _match_and_query(label: str, controller: Any) -> float | None:
        """Map a YOLO label to an AI2-THOR object type and query distance."""
        # Try the controller's distance_to() which does fuzzy matching
        dist = controller.distance_to(label)
        if dist is not None:
            return round(dist, 2)
        return None
