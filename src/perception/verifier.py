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

from src.common.types import BBox, Detection
from src.common.logger import setup_logger
from src.common.utils import box_iou, screen_pos

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
               prompt: str | None = None) -> list[Detection]:
        """Compute CLIP similarity for each detection crop.

        When *prompt* is ``None`` (default), each detection is scored against
        ``"a photo of a {label}"`` — this gives meaningful per-label
        discrimination (a real chair scores high against "a photo of a chair";
        a false-positive chair scores low).

        When *prompt* is a string, all detections are scored against that
        single prompt (legacy / generic mode).

        ``clip_score`` is filled in on every ``Detection``.
        """
        if not detections:
            return detections

        h, w = rgb.shape[:2]

        # 1. Crop + preprocess all valid detections in one pass
        crops: list = []
        valid_dets: list[Detection] = []
        for det in detections:
            crop = self._crop(rgb, det)
            if crop is None:
                det.clip_score = 0.0
                continue
            crops.append(self.preprocess(crop))
            valid_dets.append(det)

        if not crops:
            return detections

        img_batch = torch.stack(crops).to(self.device)  # [N, 3, 224, 224]

        # 2. Batch-encode all image crops
        with torch.no_grad():
            img_feats = self.model.encode_image(img_batch)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)  # [N, D]

        # 3. Text encoding + scoring
        if prompt is not None:
            # Legacy mode: single prompt for all detections
            text_tokens = clip.tokenize([prompt]).to(self.device)
            with torch.no_grad():
                text_feat = self.model.encode_text(text_tokens)
                text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            scores = (img_feats @ text_feat.T).squeeze(-1)  # [N]
            for det, s in zip(valid_dets, scores.tolist()):
                det.clip_score = round(float(s), 4)
        else:
            # Per-label mode: "a photo of a {label}" for each unique label
            unique_labels = list({d.label for d in valid_dets})
            label_prompts = [f"a photo of a {lbl}" for lbl in unique_labels]
            text_tokens = clip.tokenize(label_prompts).to(self.device)
            with torch.no_grad():
                text_feats = self.model.encode_text(text_tokens)
                text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)  # [L, D]

            label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
            for i, det in enumerate(valid_dets):
                j = label_to_idx[det.label]
                det.clip_score = round(float((img_feats[i] @ text_feats[j]).item()), 4)

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


# ---------------------------------------------------------------------------
# Helpers for CLIPDetector
# ---------------------------------------------------------------------------

def _generate_windows(h: int, w: int, grid_rows: int, grid_cols: int) -> list[tuple[int, int, int, int]]:
    """Generate window (x1, y1, x2, y2) coordinates by dividing the image into a grid."""
    cell_h, cell_w = h / grid_rows, w / grid_cols
    windows: list[tuple[int, int, int, int]] = []
    for r in range(grid_rows):
        for c in range(grid_cols):
            x1 = int(c * cell_w)
            y1 = int(r * cell_h)
            x2 = int((c + 1) * cell_w)
            y2 = int((r + 1) * cell_h)
            windows.append((x1, y1, x2, y2))
    return windows


def _greedy_nms(boxes: list[tuple[int, int, int, int]],
                scores: list[float], iou_threshold: float = 0.5) -> list[int]:
    """Greedy NMS: sort by score descending, suppress overlapping boxes. Returns kept indices."""
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    suppressed = [False] * len(boxes)
    keep: list[int] = []
    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        for j in order:
            if j == i or suppressed[j]:
                continue
            if box_iou(boxes[i], boxes[j]) > iou_threshold:
                suppressed[j] = True
    return keep


# ---------------------------------------------------------------------------
# CLIPDetector — zero-shot sliding-window detector
# ---------------------------------------------------------------------------

class CLIPDetector:
    """Zero-shot object detection using CLIP with multi-scale sliding windows.

    Unlike ``CLIPVerifier`` which re-scores existing YOLO detections, this
    detector searches the entire image for objects matching arbitrary text
    descriptions — useful for objects that YOLO was never trained on
    (e.g. doors that are just wall textures in AI2-THOR).

    Usage::

        detector = CLIPDetector(verifier=clip_verifier)  # share model
        doors = detector.detect(rgb, ["door", "window"], threshold=0.25)
    """

    def __init__(self, verifier: CLIPVerifier | None = None,
                 model_name: str = "ViT-B/32", device: str | None = None):
        if verifier is not None:
            self.model = verifier.model
            self.preprocess = verifier.preprocess
            self.device = verifier.device
        else:
            if not _CLIP_AVAILABLE:
                raise ImportError("openai-clip is not installed.  Run:  pip install openai-clip")
            if device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device
            self.model, self.preprocess = clip.load(model_name, device=self.device)
            self.model.eval()
        logger.info("CLIPDetector ready on %s", self.device)

    def detect(self, rgb: np.ndarray, targets: list[str],
               scales: list[tuple[int, int]] | None = None,
               threshold: float = 0.25, iou_threshold: float = 0.5) -> list[Detection]:
        """Search *rgb* for objects matching each target text via sliding windows.

        Args:
            rgb: H×W×3 uint8 image.
            targets: List of object names (e.g. ``["door", "window"]``).
            scales: Grid sizes as ``(rows, cols)`` tuples.
                    Default: ``[(3, 2), (4, 3), (6, 4)]``.
            threshold: Minimum CLIP cosine similarity to keep a window.
            iou_threshold: NMS overlap threshold.

        Returns:
            List of ``Detection`` objects with ``clip_score`` filled in.
        """
        if scales is None:
            scales = [(3, 2), (4, 3), (6, 4)]

        h, w = rgb.shape[:2]

        # 1. Generate all window proposals
        all_windows: list[tuple[int, int, int, int]] = []
        for rows, cols in scales:
            all_windows.extend(_generate_windows(h, w, rows, cols))

        if not all_windows:
            return []

        # 2. Crop + preprocess all windows
        crops: list = []
        valid_indices: list[int] = []
        for i, (x1, y1, x2, y2) in enumerate(all_windows):
            crop = Image.fromarray(rgb[y1:y2, x1:x2])
            try:
                crops.append(self.preprocess(crop))
                valid_indices.append(i)
            except Exception:
                continue

        if not crops:
            return []

        img_batch = torch.stack(crops).to(self.device)  # [N, 3, 224, 224]

        # 3. Batch-encode all image crops
        with torch.no_grad():
            img_feats = self.model.encode_image(img_batch)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)  # [N, D]

        # 4. Batch-encode all target texts at once
        prompts = [f"a photo of a {t}" for t in targets]
        text_tokens = clip.tokenize(prompts).to(self.device)  # [T, 77]
        with torch.no_grad():
            text_feats = self.model.encode_text(text_tokens)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)  # [T, D]

        # 5. Per-target scoring + NMS
        # sims = [N, T] — one column per target
        sims = (img_feats @ text_feats.T)  # [N, T]

        all_detections: list[Detection] = []

        for t_idx, target in enumerate(targets):
            col = sims[:, t_idx]  # [N]

            # Collect candidates above threshold
            candidates: list[tuple[int, float]] = []  # (window_index, score)
            for j, score in enumerate(col.tolist()):
                if score >= threshold:
                    candidates.append((valid_indices[j], score))

            if not candidates:
                continue

            cand_boxes = [all_windows[idx] for idx, _ in candidates]
            cand_scores = [s for _, s in candidates]
            keep = _greedy_nms(cand_boxes, cand_scores, iou_threshold)

            for k in keep:
                idx, score = candidates[k]
                x1, y1, x2, y2 = all_windows[idx]
                pos = screen_pos(x1, x2, w)
                all_detections.append(Detection(
                    label=target,
                    bbox=BBox(x1=float(x1), y1=float(y1),
                              x2=float(x2), y2=float(y2)),
                    confidence=round(score, 4),
                    screen_position=pos,
                    clip_score=round(score, 4),
                ))

        return all_detections
