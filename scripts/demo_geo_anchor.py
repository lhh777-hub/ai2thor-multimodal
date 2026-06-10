"""
GeoAnchor label-switching demo — approach a target object step by step,
print YOLO labels vs anchor status, and save annotated frames.

Usage:
    python scripts/demo_geo_anchor.py --scene FloorPlan1 --target microwave
    python scripts/demo_geo_anchor.py --scene FloorPlan1 --target refrigerator
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
from src.perception.geo_anchor import GeometricAnchor, pixel_to_world
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


def _draw_frame(rgb: np.ndarray, dets, anchor_label: str = "",
                anchor_locked: bool = False) -> np.ndarray:
    """Draw detection boxes on rgb.  Anchor-remapped dets get a gold border."""
    img = rgb.copy()
    h, w = img.shape[:2]
    for d in dets:
        b = d.bbox
        x1, y1 = max(0, int(b.x1)), max(0, int(b.y1))
        x2, y2 = min(w, int(b.x2)), min(h, int(b.y2))

        # Gold border if this detection was remapped by anchor
        is_remapped = anchor_locked and d.label.lower() == anchor_label.lower()
        border = (0, 215, 255) if is_remapped else (
            (0, 255, 0) if d.confidence >= 0.5 else (0, 255, 255))

        cv2.rectangle(img, (x1, y1), (x2, y2), border, 2)
        dist_str = f"{d.distance_meters:.1f}m" if d.distance_meters else "?"
        label = f"{d.label} {d.confidence:.2f} [{dist_str}]"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x1, max(y1 - th - 6, 0)),
                      (x1 + tw + 4, y1), border, -1)
        cv2.putText(img, label, (x1 + 2, max(y1 - 4, th + 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    # Overlay anchor status
    status = f"GeoAnchor: LOCKED '{anchor_label}'" if anchor_locked else "GeoAnchor: —"
    colour = (0, 215, 255) if anchor_locked else (128, 128, 128)
    cv2.putText(img, status, (10, h - 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, colour, 2)
    return img


def main():
    p = argparse.ArgumentParser(description="GeoAnchor label-switching demo")
    p.add_argument("--scene", default="FloorPlan1")
    p.add_argument("--target", default="microwave")
    p.add_argument("--model", default="runs/detect/runs/train/weights/best.pt")
    p.add_argument("--outdir", default="outputs/geo_anchor_demo")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    detector = YOLODetector(model_name=args.model, confidence=0.3)
    pipeline = PerceptionPipeline(detector, HeuristicDepth())

    with ThorController(width=800, height=600, render_depth=True) as ctrl:
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
                y=obj_pos.y,
                z=obj_pos.z + dz / dist * 3.0,
            )
            for _ in ctrl.navigate_to(approach):
                pass

        anchor = GeometricAnchor(distance_threshold=0.3)

        print(f"\n{'='*70}")
        print(f"  GeoAnchor Demo: '{args.target}' in {args.scene}")
        print(f"  Frames saved to: {args.outdir}/")
        print(f"{'='*70}")
        print(f"  {'Step':<5s} {'Dist':<8s} {'YOLO nearby':<45s} {'Anchor':<10s}")
        print(f"  {'-'*5} {'-'*8} {'-'*45} {'-'*10}")

        for step in range(10):
            view = ctrl.get_current_view()
            dets = pipeline.process(view.sensor_data.rgb, controller=ctrl)

            # Compute 3D world positions
            depth = view.sensor_data.depth
            if depth is not None:
                h, w = depth.shape[:2]
                s = view.agent_state
                for d in dets:
                    cx = max(0, min(w - 1, int(d.bbox.center_x)))
                    cy = max(0, min(h - 1, int(d.bbox.center_y)))
                    dv = float(depth[cy, cx])
                    if 0 < dv < 20:
                        d._world_xyz = pixel_to_world(
                            cx, cy, dv, width=w, height=h,
                            agent_x=s.position.x, agent_y=s.position.y,
                            agent_z=s.position.z,
                            heading_deg=s.heading_deg, horizon_deg=s.horizon_deg,
                        )

            # GeoAnchor
            if not anchor.is_locked:
                anchor.try_lock(args.target, dets, min_conf=0.5, min_distance=0.5)
            else:
                anchor.remap_by_proximity(args.target, dets)

            if anchor.is_locked and anchor.closest_distance < 0.5:
                anchor.unlock()

            # Print summary
            gt_dist = ctrl.distance_to(args.target) or 999
            nearby = [d for d in dets
                      if d.distance_meters > 0 and d.distance_meters < 3.0]
            nearby_str = ", ".join(
                f"{d.label}({d.distance_meters:.1f}m)" for d in nearby[:4]
            ) or "(nothing)"
            anchor_str = f"LOCKED@{anchor.closest_distance:.1f}m" if anchor.is_locked else "—"
            print(f"  {step:<5d} {gt_dist:<8.2f} {nearby_str:<45s} {anchor_str:<10s}")

            # Save annotated frame
            annotated = _draw_frame(
                view.sensor_data.rgb, dets,
                anchor_label=args.target if anchor.is_locked else "",
                anchor_locked=anchor.is_locked,
            )
            fname = f"step_{step:02d}.png" if not anchor.is_locked else \
                    f"step_{step:02d}_ANCHORED.png"
            cv2.imwrite(os.path.join(args.outdir, fname),
                        cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

            ctrl.step("MOVE_FORWARD")

    print(f"\n  ▶ Frames: {args.outdir}/")
    print(f"  ▶ Files with '_ANCHORED' suffix = GeoAnchor locked & remapping labels")
    print(f"  ▶ Gold-bordered boxes = remapped by anchor\n")


if __name__ == "__main__":
    main()
