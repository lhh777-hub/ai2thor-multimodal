"""
One-shot detection capture — save annotated frame for report figures.

Usage:
    # Pretrained model
    python scripts/capture_detections.py --scene FloorPlan1 --model yolov8n.pt --output outputs/fig_pretrained.png

    # Fine-tuned model
    python scripts/capture_detections.py --scene FloorPlan1 --model runs/detect/runs/train/weights/best.pt --output outputs/fig_finetuned.png

    # Multi-scene batch capture
    python scripts/capture_detections.py --scenes FloorPlan1,FloorPlan3,FloorPlan5 --model yolov8n.pt --outdir outputs/fig_pretrained
    python scripts/capture_detections.py --scenes FloorPlan1,FloorPlan3,FloorPlan5 --model runs/detect/runs/train/weights/best.pt --outdir outputs/fig_finetuned
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.controller.thor import ThorController
from src.perception.detector import YOLODetector
from src.perception.depth import HeuristicDepth
from src.perception.pipeline import PerceptionPipeline


def _draw_detections(rgb, detections):
    """Draw detection boxes + labels on rgb. Returns annotated copy."""
    img = rgb.copy()
    h, w = img.shape[:2]
    for d in detections:
        b = d.bbox
        x1, y1 = max(0, int(b.x1)), max(0, int(b.y1))
        x2, y2 = min(w, int(b.x2)), min(h, int(b.y2))
        if d.confidence >= 0.5:
            colour = (0, 255, 0)
        elif d.confidence >= 0.3:
            colour = (0, 255, 255)
        else:
            colour = (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
        dist_str = f"{d.distance_meters:.1f}m" if d.distance_meters else d.distance_level
        label = f"{d.label} {d.confidence:.2f} [{dist_str}]"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x1, max(y1 - th - 6, 0)),
                      (x1 + tw + 4, y1), colour, -1)
        cv2.putText(img, label, (x1 + 2, max(y1 - 4, th + 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    return img


def capture(scene: str, model_path: str, output_path: str, confidence: float = 0.3):
    """Capture one annotated frame from *scene* using *model_path*."""
    detector = YOLODetector(model_name=model_path, confidence=confidence)
    pipeline = PerceptionPipeline(detector, HeuristicDepth())

    with ThorController(width=800, height=600, render_depth=False) as ctrl:
        ctrl.load_scene(scene, seed=42)
        view = ctrl.get_current_view()
        dets = pipeline.process(view.sensor_data.rgb, controller=ctrl)
        annotated = _draw_detections(view.sensor_data.rgb, dets)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        n = len(dets)
        labels = ", ".join(d.label for d in dets[:8])
        print(f"[{scene}] {model_path}: {n} detections → {output_path}")
        if labels:
            print(f"  top: {labels}")


def main():
    p = argparse.ArgumentParser(
        description="One-shot detection capture for report figures")
    p.add_argument("--scene", default="FloorPlan1")
    p.add_argument("--scenes", default=None,
                   help="Comma-separated scenes (overrides --scene)")
    p.add_argument("--model", required=True,
                   help="Path to YOLO model (yolov8n.pt or best.pt)")
    p.add_argument("--output", default=None,
                   help="Single output path (use with --scene)")
    p.add_argument("--outdir", default="outputs/figures",
                   help="Output directory for --scenes mode")
    p.add_argument("--confidence", type=float, default=0.3)
    args = p.parse_args()

    if args.scenes:
        scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
        os.makedirs(args.outdir, exist_ok=True)
        for scene in scenes:
            model_tag = os.path.splitext(os.path.basename(args.model))[0]
            out = os.path.join(args.outdir, f"{scene}_{model_tag}.png")
            capture(scene, args.model, out, args.confidence)
    else:
        out = args.output or f"outputs/{args.scene}_detect.png"
        capture(args.scene, args.model, out, args.confidence)


if __name__ == "__main__":
    main()
