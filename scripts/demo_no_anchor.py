"""
Plain approach demo (NO GeoAnchor) — walk toward a target and log raw YOLO labels.

Usage:
    python scripts/demo_no_anchor.py --scene FloorPlan1 --target microwave
    python scripts/demo_no_anchor.py --scene FloorPlan1 --target refrigerator
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.controller.thor import ThorController
from src.perception.detector import YOLODetector
from src.perception.depth import HeuristicDepth
from src.perception.pipeline import PerceptionPipeline
from src.common.types import Vec3


def _find_object(ctrl, target: str) -> Vec3 | None:
    obj_map = ctrl.get_object_map()
    for name, info in obj_map.items():
        if target.lower() in name.lower() or name.lower() in target.lower():
            return info["position"]
    try:
        from src.perception.class_config import load_config
        cfg = load_config()
        for thor_name, yolo_name in cfg.thor_to_names.items():
            if yolo_name.lower() == target.lower() and thor_name in obj_map:
                return obj_map[thor_name]["position"]
    except Exception:
        pass
    return None


def _draw_frame(rgb, dets):
    img = rgb.copy()
    h, w = img.shape[:2]
    for d in dets:
        b = d.bbox
        x1, y1 = max(0, int(b.x1)), max(0, int(b.y1))
        x2, y2 = min(w, int(b.x2)), min(h, int(b.y2))
        colour = (0, 255, 0) if d.confidence >= 0.5 else (0, 255, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
        dist_str = f"{d.distance_meters:.1f}m" if d.distance_meters else "?"
        label = f"{d.label} {d.confidence:.2f} [{dist_str}]"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x1, max(y1 - th - 6, 0)),
                      (x1 + tw + 4, y1), colour, -1)
        cv2.putText(img, label, (x1 + 2, max(y1 - 4, th + 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.putText(img, "GeoAnchor: OFF", (10, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (128, 128, 128), 2)
    return img


def main():
    p = argparse.ArgumentParser(description="Plain approach — no GeoAnchor")
    p.add_argument("--scene", default="FloorPlan1")
    p.add_argument("--target", default="microwave")
    p.add_argument("--model", default="runs/detect/runs/train/weights/best.pt")
    p.add_argument("--outdir", default="outputs/no_anchor_demo")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    detector = YOLODetector(model_name=args.model, confidence=0.3)
    pipeline = PerceptionPipeline(detector, HeuristicDepth())

    with ThorController(width=800, height=600, render_depth=False) as ctrl:
        ctrl.load_scene(args.scene, seed=42)

        obj_pos = _find_object(ctrl, args.target)
        if obj_pos is None:
            print(f"Object '{args.target}' not found in {args.scene}")
            return

        agent = ctrl.agent_state.position
        dx = agent.x - obj_pos.x
        dz = agent.z - obj_pos.z
        dist = (dx**2 + dz**2)**0.5
        if dist > 0.01:
            approach = Vec3(
                x=obj_pos.x + dx / dist * 3.0,
                y=obj_pos.y, z=obj_pos.z + dz / dist * 3.0)
            for _ in ctrl.navigate_to(approach):
                pass

        print(f"\n  Plain approach (NO GeoAnchor): '{args.target}' in {args.scene}")
        print(f"  Frames: {args.outdir}/")
        print(f"  {'Step':<5s} {'Dist':<8s} {'YOLO nearby':<45s}")
        print(f"  {'-'*5} {'-'*8} {'-'*45}")

        for step in range(10):
            view = ctrl.get_current_view()
            dets = pipeline.process(view.sensor_data.rgb, controller=ctrl)

            gt_dist = ctrl.distance_to(args.target) or 999
            nearby = [d for d in dets
                      if d.distance_meters > 0 and d.distance_meters < 3.0]
            nearby_str = ", ".join(
                f"{d.label}({d.distance_meters:.1f}m)" for d in nearby[:4]
            ) or "(nothing)"
            print(f"  {step:<5d} {gt_dist:<8.2f} {nearby_str:<45s}")

            annotated = _draw_frame(view.sensor_data.rgb, dets)
            cv2.imwrite(os.path.join(args.outdir, f"step_{step:02d}.png"),
                        cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

            ctrl.step("MOVE_FORWARD")

    print(f"\n  ▶ Frames: {args.outdir}/\n")


if __name__ == "__main__":
    main()
