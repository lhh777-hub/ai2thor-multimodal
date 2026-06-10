"""
Interactive GeoAnchor comparison — manual control with anchor toggle.

Commands:
    w/s       move forward / back
    a/d       turn left / right (90 deg)
    q/e       fine turn (30 deg)
    anchor    toggle GeoAnchor ON / OFF
    detect    show current detections
    info      agent pose
    quit

Usage:
    python scripts/interactive_anchor.py --scene FloorPlan1 --target microwave
    python scripts/interactive_anchor.py --scene FloorPlan1 --target refrigerator
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
from src.perception.geo_anchor import GeometricAnchor, pixel_to_world


SHORTCUT: dict[str, str] = {
    "w": "MOVE_FORWARD", "s": "MOVE_BACK",
    "a": "TURN_LEFT", "d": "TURN_RIGHT",
    "q": "TURN_LEFT_SMALL", "e": "TURN_RIGHT_SMALL",
}


def _draw_frame(rgb, dets, anchor_on, anchor_label, anchor_locked, closest_dist):
    img = rgb.copy()
    h, w = img.shape[:2]
    for d in dets:
        b = d.bbox
        x1, y1 = max(0, int(b.x1)), max(0, int(b.y1))
        x2, y2 = min(w, int(b.x2)), min(h, int(b.y2))
        remapped = anchor_locked and d.label.lower() == anchor_label.lower()
        border = (0, 215, 255) if remapped else (
            (0, 255, 0) if d.confidence >= 0.5 else (0, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), border, 2)
        dist_str = f"{d.distance_meters:.1f}m" if d.distance_meters else "?"
        label = f"{d.label} {d.confidence:.2f} [{dist_str}]"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x1, max(y1 - th - 6, 0)),
                      (x1 + tw + 4, y1), border, -1)
        cv2.putText(img, label, (x1 + 2, max(y1 - 4, th + 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    s = "LOCKED" if anchor_locked else "scanning"
    status = f"GeoAnchor: {'ON' if anchor_on else 'OFF'} [{s}]"
    colour = (0, 215, 255) if anchor_locked else (128, 128, 128)
    cv2.putText(img, status, (10, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="FloorPlan1")
    p.add_argument("--target", default="microwave")
    p.add_argument("--model", default="runs/detect/runs/train/weights/best.pt")
    p.add_argument("--outdir", default="outputs/interactive_anchor")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    detector = YOLODetector(model_name=args.model, confidence=0.3)
    pipeline = PerceptionPipeline(detector, HeuristicDepth())
    anchor = GeometricAnchor(distance_threshold=0.3)
    anchor_on = True
    frame_idx = 0

    with ThorController(width=800, height=600, render_depth=True) as ctrl:
        ctrl.load_scene(args.scene, seed=42)

        print(f"\n  Interactive GeoAnchor Demo — '{args.target}' in {args.scene}")
        print(f"  GeoAnchor: ON  |  Frames: {args.outdir}/")
        print(f"  wasd=move  qe=fine-turn  anchor=toggle  detect=show  info=pose  quit\n")

        while True:
            try:
                cmd = input("  > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not cmd:
                continue

            head = cmd.lower()

            if head in ("quit", "exit", "Q"):
                break

            if head == "anchor":
                anchor_on = not anchor_on
                if not anchor_on:
                    anchor.reset()
                print(f"  GeoAnchor: {'ON' if anchor_on else 'OFF'}")
                continue

            if head == "detect":
                view = ctrl.get_current_view()
                dets = pipeline.process(view.sensor_data.rgb, controller=ctrl)
                _process_anchor(view, dets, anchor, anchor_on, args.target)
                nearby = [d for d in dets
                          if d.distance_meters > 0 and d.distance_meters < 3.0]
                if nearby:
                    for d in sorted(nearby, key=lambda x: x.distance_meters)[:6]:
                        print(f"    {d.label:<18s} conf={d.confidence:.2f}  "
                              f"dist={d.distance_meters:.2f}m  pos={d.screen_position}")
                else:
                    print("    (nothing nearby)")
                annotated = _draw_frame(view.sensor_data.rgb, dets, anchor_on,
                                        args.target, anchor.is_locked,
                                        anchor.closest_distance)
                fname = f"frame_{frame_idx:04d}.png"
                cv2.imwrite(os.path.join(args.outdir, fname),
                            cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                print(f"    saved → {fname}")
                frame_idx += 1
                continue

            if head == "info":
                s = ctrl.agent_state
                print(f"  pos=({s.position.x:.2f},{s.position.y:.2f},{s.position.z:.2f})  "
                      f"heading={s.heading_deg:.0f}deg  colliding={s.is_colliding}")
                continue

            if head in SHORTCUT:
                ctrl.step(SHORTCUT[head])
                view = ctrl.get_current_view()
                dets = pipeline.process(view.sensor_data.rgb, controller=ctrl)
                _process_anchor(view, dets, anchor, anchor_on, args.target)
                annotated = _draw_frame(view.sensor_data.rgb, dets, anchor_on,
                                        args.target, anchor.is_locked,
                                        anchor.closest_distance)
                fname = f"frame_{frame_idx:04d}.png"
                cv2.imwrite(os.path.join(args.outdir, fname),
                            cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                frame_idx += 1
                continue

            print(f"  Unknown: '{cmd}'.  Try: w/a/s/d/q/e  anchor  detect  info  quit")

    print(f"  Frames saved: {frame_idx} → {args.outdir}/\n")


def _process_anchor(view, dets, anchor, anchor_on, target):
    if not anchor_on:
        return
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
                    heading_deg=s.heading_deg, horizon_deg=s.horizon_deg)
    if not anchor.is_locked:
        anchor.try_lock(target, dets, min_conf=0.5, min_distance=0.5)
    else:
        anchor.remap_by_proximity(target, dets)
    if anchor.is_locked and anchor.closest_distance < 0.5:
        anchor.unlock()


if __name__ == "__main__":
    main()
