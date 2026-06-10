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
from src.perception.verifier import filter_by_clip
from src.common.logger import setup_logger

logger = setup_logger("perception")


class PerceptionPipeline:
    """Detect objects, estimate distances, optionally verify with CLIP."""

    def __init__(self, detector: YOLODetector, depth_estimator: DepthEstimator,
                 verifier=None, prior=None, clip_detector=None):
        self.detector = detector
        self.depth = depth_estimator
        self.verifier = verifier           # CLIPVerifier or None
        self.prior = prior                 # ScenePrior or None
        self.clip_detector = clip_detector # CLIPDetector or None
        self._class_config = None          # cached class config for reverse lookup

    def process(self, rgb: np.ndarray,
                clip_prompt: str | None = None,
                clip_threshold: float = 0.0,
                clip_targets: list[str] | None = None,
                clip_detect_threshold: float = 0.28,
                clip_detect_iou: float = 0.5,
                controller: Any | None = None,
                ) -> list[Detection]:
        """Run the full perception pipeline on *rgb*.

        Steps:
        1. YOLO detection
        2. CLIP zero-shot detection (if *clip_targets* provided)
        3. Scene prior weighting (if configured)
        4. CLIP verification + filtering (if configured)
        5. Distance estimation (heuristic / depth-model)
        6. Ground-truth distance override (if *controller* is provided)

        Returns the final list, sorted by confidence (high first, after prior).
        """
        # 1. YOLO detection
        detections = self.detector.detect(rgb)

        # 2. CLIP zero-shot detection for extra targets not in YOLO training
        if self.clip_detector is not None and clip_targets:
            extra = self.clip_detector.detect(
                rgb, clip_targets,
                threshold=clip_detect_threshold,
                iou_threshold=clip_detect_iou)
            if extra:
                logger.info("CLIPDetector found %d extra detections", len(extra))
                detections.extend(extra)

        # 3. Scene prior
        if self.prior is not None:
            detections = self.prior.apply(detections)
            detections.sort(key=lambda d: d.confidence, reverse=True)

        # 4. CLIP verification (per-label prompts by default for discrimination)
        if self.verifier is not None:
            detections = self.verifier.verify(rgb, detections, prompt=clip_prompt)
            if clip_threshold > 0:
                detections = filter_by_clip(detections, clip_threshold)

        # 5. Distance (heuristic or depth-model)
        for d in detections:
            level, metres = self.depth.estimate(rgb, d)
            d.distance_level = level
            d.distance_meters = metres

        # 6. Ground-truth override from AI2-THOR (exact, when available)
        if controller is not None:
            self._apply_ground_truth_distance(detections, controller)

        return detections

    # ------------------------------------------------------------------
    # Coverage summary (diagnostic)
    # ------------------------------------------------------------------

    def coverage_summary(self, detections: list[Detection],
                         controller: Any | None = None) -> str:
        """Return a one-line coverage summary comparing detections to scene objects.

        Example: ``"12/23 scene objects detected (52%), missed: bed, toilet, mirror"``
        """
        if controller is None:
            return ""
        try:
            obj_map = controller.get_object_map()
            if not obj_map:
                return ""
            # Map AI2-THOR types to YOLO labels via cached config
            if self._class_config is None:
                from src.perception.class_config import load_config
                self._class_config = load_config()
            cfg = self._class_config

            scene_labels: set[str] = set()
            for thor_type in obj_map:
                cid = cfg.thor_to_class_id(thor_type)
                if cid is not None:
                    scene_labels.add(cfg.class_names[cid])

            if not scene_labels:
                return ""

            detected = {d.label.lower() for d in detections}
            matched = {s for s in scene_labels if s.lower() in detected}
            missed = sorted(scene_labels - matched)
            pct = len(matched) / len(scene_labels) * 100 if scene_labels else 0
            missed_str = ", ".join(missed[:5])
            if len(missed) > 5:
                missed_str += f" +{len(missed) - 5} more"
            return f"{len(matched)}/{len(scene_labels)} scene objects detected ({pct:.0f}%)" \
                   + (f", missed: {missed_str}" if missed_str else "")
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Ground-truth distance (AI2-THOR)
    # ------------------------------------------------------------------

    def _apply_ground_truth_distance(self, detections: list[Detection],
                                     controller: Any) -> None:
        """Override ``distance_meters`` and ``distance_level`` with exact
        AI2-THOR object positions, and penalise detections whose heuristic
        distance is inconsistent with ground truth (likely false positives).
        """
        for d in detections:
            gt_metres = self._match_and_query(d.label, controller)
            if gt_metres is None:
                continue

            # Distance consistency check: if the heuristic estimate
            # (based on bbox size) differs wildly from the ground-truth
            # position, this detection is probably a false positive
            # (e.g. bread misclassified as "chair" at close range
            #  while the real chair is 3 m away).
            heuristic = d.distance_meters
            if heuristic > 0 and gt_metres > 0:
                ratio = max(heuristic, gt_metres) / max(min(heuristic, gt_metres), 0.01)
                delta = abs(heuristic - gt_metres)
                if ratio > 2.5 and delta > 1.0:
                    d.confidence = round(d.confidence * 0.4, 4)
                    logger.debug("%s: heuristic=%.2f gt=%.2f ratio=%.1f → "
                                 "penalised (conf=%.3f)",
                                 d.label, heuristic, gt_metres, ratio, d.confidence)

            d.distance_meters = gt_metres
            d.distance_level = _level_from_metres(gt_metres)

    def _match_and_query(self, label: str, controller: Any) -> float | None:
        """Map a YOLO label to an AI2-THOR object type and query distance.

        Tries the raw YOLO label first, then reverse-maps through the
        cached class config (e.g. "refrigerator" → "Fridge").
        """
        dist = controller.distance_to(label)
        if dist is not None:
            return round(dist, 2)

        # Lazy-load and cache the class config (avoid re-parsing YAML per detection)
        if self._class_config is None:
            try:
                from src.perception.class_config import load_config
                self._class_config = load_config()
            except Exception:
                return None

        for thor_type, yolo_name in self._class_config.thor_to_names.items():
            if yolo_name.lower() == label.lower():
                dist = controller.distance_to(thor_type)
                if dist is not None:
                    return round(dist, 2)
        return None
