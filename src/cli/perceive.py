"""
Perception demo — YOLO detection + depth + CLIP verification + scene priors.

Usage::

    python src/cli/perceive.py --scene FloorPlan1
    python src/cli/perceive.py --scene FloorPlan1 --model yolov8m.pt --clip
    python src/cli/perceive.py --scene FloorPlan1 --confidence 0.2 --clip-threshold 0.18

Commands::

    wasd/qe    Move / turn
    detect      Run perception on current view
    clip        Toggle CLIP verification
    prior       Toggle scene prior weighting
    auto        Toggle auto-detect after every move
    info        Scene info
    quit        Exit
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.controller.thor import ThorController
from src.perception.detector import YOLODetector
from src.perception.depth import HeuristicDepth
from src.perception.pipeline import PerceptionPipeline
from src.recording.collector import FrameCollector
from src.common.logger import setup_logger

logger = setup_logger("perceive")

SHORTCUT: dict[str, str] = {
    "w": "MOVE_FORWARD", "s": "MOVE_BACK",
    "a": "TURN_LEFT", "d": "TURN_RIGHT",
    "q": "TURN_LEFT_SMALL", "e": "TURN_RIGHT_SMALL",
}


def _print_detections(detections) -> None:
    if not detections:
        print("  (nothing detected)")
        return
    header = f"  {'Label':<18s} {'Conf':>6s}"
    cols = 1
    if any(getattr(d, "clip_score", 0) > 0 for d in detections):
        header += f"  {'CLIP':>6s}"
        cols += 1
    header += f"  {'Pos':<8s}  {'Dist':>8s}  {'Dist(m)':>8s}"
    print(header)
    sep = f"  {'-'*18} {'-'*6}" + (f"  {'-'*6}" if cols > 1 else "")
    sep += f"  {'-'*8}  {'-'*8}  {'-'*8}"
    print(sep)
    for d in detections:
        line = f"  {d.label:<18s} {d.confidence:5.2f}"
        if cols > 1:
            line += f"   {d.clip_score:5.2f}"
        line += f"   {d.screen_position:<8s}  {d.distance_level:<8s}  {d.distance_meters:>6.2f}"
        print(line)


def _check_classes(scene: str, classes_path: str | None = None) -> None:
    """Diagnostic: list all AI2-THOR objects in *scene* and their class mapping."""
    from src.perception.class_config import load_config

    cfg = load_config(classes_path)

    with ThorController(width=640, height=480) as ctrl:
        ctrl.load_scene(scene)
        obj_map = ctrl.get_object_map()

    print(f"\n  Scene: {scene}")
    print(f"  Config: {cfg.source_path}")
    print(f"  Total object types in scene: {len(obj_map)}\n")

    mapped, unmapped = [], []
    for name in sorted(obj_map):
        cls_id = cfg.thor_to_class_id(name)
        if cls_id is not None:
            mapped.append((name, cls_id, cfg.class_names[cls_id]))
        else:
            unmapped.append(name)

    if mapped:
        print(f"  Mapped ({len(mapped)}):")
        print(f"  {'AI2-THOR':<25s} {'class_id':>8s}  YOLO name")
        print(f"  {'-'*25} {'-'*8}  {'-'*20}")
        for name, cid, yname in mapped:
            print(f"  {name:<25s} {cid:>8d}  {yname}")

    if unmapped:
        print(f"\n  UNMAPPED ({len(unmapped)}) — YOLO won't detect these:")
        for name in unmapped:
            print(f"    {name}")
        print(f"\n  To fix: add these lines to {cfg.source_path}:\n")
        print(f"  # In the 'names' list (at the end):")
        for name in unmapped:
            yname = name.lower().replace(" ", "_")
            print(f"  # - {yname}")
        print(f"\n  # In 'thor_to_names':")
        for name in unmapped:
            yname = name.lower().replace(" ", "_")
            print(f"  {name}: {yname}")
    else:
        print(f"\n  All objects mapped.")

    print(f"\n  Total YOLO classes: {cfg.num_classes}")
    print(f"  Classes source: {cfg.source_path}\n")


def _verify_labels(scene: str, classes_path: str | None = None) -> None:
    """Collect 20 frames with ground-truth labels, scanning 360° at each position.

    Output goes to ``outputs/verify/`` — open the saved PNGs to check
    whether bounding boxes and class names look correct.
    """
    import cv2

    from src.controller.thor import ThorController
    from src.perception.class_config import load_config

    cfg = load_config(classes_path)
    out_dir = "outputs/verify"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n  Collecting from {scene} ...")
    print(f"  Classes: {cfg.num_classes} | Pattern: scan 360° → move → repeat\n")

    with ThorController(width=640, height=480, visibility_distance=10.0,
                        render_instance_seg=True) as ctrl:
        ctrl.load_scene(scene, seed=42)
        # Show what's in the scene vs what can be labeled
        obj_map = ctrl.get_object_map()
        total_obj_types = len(obj_map)
        mappable = sum(1 for name in obj_map if cfg.thor_to_class_id(name) is not None)
        print(f"  Scene objects: {total_obj_types} types, {mappable} mappable to YOLO classes\n")
        n_saved = 0
        total_labels = 0
        waypoint = 0
        objects_ever_labeled: set[str] = set()

        while n_saved < 20 and waypoint < 30:
            # Scan 360° at current position
            for _ in range(4):
                result = ctrl.step("TURN_LEFT")
                seg = getattr(result.raw_event, "instance_segmentation_frame", None)
                if seg is None:
                    continue

                colour_to_id = getattr(result.raw_event, "color_to_object_id", {})
                if not colour_to_id:
                    continue

                rgb = result.sensor_data.rgb.copy()
                drawn, obj_types = _draw_bboxes_on_frame_manual(
                    rgb, seg, colour_to_id, result.raw_event, cfg)
                objects_ever_labeled.update(obj_types)
                if drawn > 0:
                    path = os.path.join(out_dir, f"{scene}_w{waypoint:02d}_{n_saved:03d}.png")
                    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    n_saved += 1
                    total_labels += drawn
                    print(f"  [{n_saved:02d}] {path}  —  {drawn} labels")
                if n_saved >= 20:
                    break

            # Move forward
            result = ctrl.step("MOVE_FORWARD")
            waypoint += 1
            if not result.success:
                ctrl.step("TURN_LEFT")
                ctrl.step("TURN_LEFT")

            # Save one frame after moving
            seg = getattr(result.raw_event, "instance_segmentation_frame", None)
            if seg is not None:
                colour_to_id = getattr(result.raw_event, "color_to_object_id", {})
                if colour_to_id:
                    rgb = result.sensor_data.rgb.copy()
                    drawn, obj_types = _draw_bboxes_on_frame_manual(
                        rgb, seg, colour_to_id, result.raw_event, cfg)
                    objects_ever_labeled.update(obj_types)
                    if drawn > 0:
                        path = os.path.join(out_dir, f"{scene}_w{waypoint:02d}_{n_saved:03d}.png")
                        cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                        n_saved += 1
                        total_labels += drawn

    # --- Diagnostic: which objects from the scene metadata NEVER appeared in
    #     any instance segmentation frame?  (These need manual investigation.)
    print(f"\n  --- Label Coverage Summary ---")
    never_seen = set()
    for name in obj_map:
        if name not in objects_ever_labeled:
            never_seen.add(name)

    if never_seen:
        print(f"  Objects NEVER labeled ({len(never_seen)}):")
        for name in sorted(never_seen):
            mapped = "✓" if cfg.thor_to_class_id(name) is not None else "✗ not in classes.yaml"
            print(f"    {name:<25s}  {mapped}")
        print(f"\n  Possible reasons:")
        print(f"    - AI2-THOR may not include this type in instance segmentation")
        print(f"    - Object is inside a cabinet/drawer (not visible)")
        print(f"    - Object occluded from all viewpoints")
    else:
        print(f"  All scene objects were labeled at least once.")

    labelled = sorted(objects_ever_labeled)
    print(f"\n  Objects labeled at least once ({len(labelled)}):")
    print(f"  {', '.join(labelled)}")

    print(f"\n  Done. {n_saved} frames ({total_labels} labels)")
    print(f"  Open outputs/verify/ — files named w00 (1st waypoint), w01 (2nd), etc.")
    print(f"\n  If boxes look correct → proceed:")
    print(f"    python src/perception/finetune/collect.py --scenes 10 --steps 200")
    print(f"  If not → check mapping:")
    print(f"    python src/cli/perceive.py --check-classes --scene {scene}\n")


def _draw_bboxes_on_frame_manual(rgb: np.ndarray, seg, colour_to_id: dict,
                                  event, cfg) -> tuple[int, set[str]]:
    """Draw ground-truth boxes on *rgb* using AI2-THOR event metadata.

    Returns ``(count, {object_type, ...})`` — the set is for coverage tracking.
    """
    import cv2
    seen = set()
    drawn = 0
    obj_types: set[str] = set()

    for colour, obj_id in colour_to_id.items():
        if obj_id in seen:
            continue
        seen.add(obj_id)

        colour_np = np.array(colour, dtype=np.uint8)
        mask = np.all(seg == colour_np, axis=-1)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue

        obj = event.get_object(obj_id)
        if obj is None:
            continue
        thor_type = obj.get("objectType", "")
        cls_id = cfg.thor_to_class_id(thor_type)
        if cls_id is None:
            continue

        obj_types.add(thor_type)

        x1, y1 = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max()), int(ys.max())
        label = cfg.class_names[cls_id]

        cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(rgb, label, (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        drawn += 1

    return drawn, obj_types


def run(scene: str = "FloorPlan1", width: int = 800, height: int = 600,
        model: str = "yolov8n.pt", confidence: float = 0.3, use_clip: bool = False,
        use_prior: bool = True, clip_threshold: float = 0.0):

    auto_detect = True
    detector = YOLODetector(model_name=model, confidence=confidence)
    depth = HeuristicDepth()

    # Optional modules
    verifier = None
    if use_clip:
        try:
            from src.perception.verifier import CLIPVerifier
            verifier = CLIPVerifier()
        except ImportError as e:
            print(f"  CLIP not available: {e}")
            use_clip = False

    prior = None
    if use_prior:
        from src.perception.prior import ScenePrior
        prior = ScenePrior(scene)

    pipeline = PerceptionPipeline(detector, depth, verifier=verifier, prior=prior)
    collector = FrameCollector(scene=scene)

    with ThorController(width=width, height=height) as ctrl:
        ctrl.load_scene(scene)
        result = ctrl.get_current_view()
        collector.record(result)

        room = prior.room_type if prior else "unknown"
        print(f"\n  Session: {collector.session_dir}")
        print(f"  Scene: {scene} ({room})  |  Model: {model}  |  YOLO conf={confidence}")
        print(f"  CLIP: {'ON' if use_clip else 'OFF'}  "
              f"|  Prior: {'ON' if use_prior else 'OFF'}  "
              f"|  Auto: {'ON' if auto_detect else 'OFF'}")
        print(f"  Commands: wasd=move  detect=perceive  clip/prior=toggle  quit=exit\n")

        if auto_detect:
            dets = pipeline.process(result.sensor_data.rgb, controller=ctrl)
            _print_detections(dets)

        while True:
            try:
                cmd = input("  > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Exiting.")
                break

            if not cmd:
                continue
            head = cmd.lower()

            if head in ("quit", "exit"):
                break

            if head == "info":
                p = result.agent_state.position
                print(f"  Scene: {ctrl.scene_name} ({room})  Steps: {ctrl.step_count}")
                print(f"  Model: {model}  CLIP: {use_clip}  Prior: {use_prior}")
                print(f"  Pos: ({p.x:.2f}, {p.y:.2f}, {p.z:.2f})  "
                      f"Head: {result.agent_state.heading_deg:.0f} deg")
                continue

            if head == "detect":
                dets = pipeline.process(result.sensor_data.rgb, controller=ctrl)
                _print_detections(dets)
                continue

            if head == "clip":
                if use_clip:
                    verifier = None
                    use_clip = False
                else:
                    try:
                        from src.perception.verifier import CLIPVerifier
                        verifier = CLIPVerifier()
                        use_clip = True
                    except ImportError as e:
                        print(f"  CLIP unavailable: {e}")
                        continue
                pipeline = PerceptionPipeline(detector, depth, verifier=verifier, prior=prior)
                print(f"  CLIP: {'ON' if use_clip else 'OFF'}")
                continue

            if head == "prior":
                use_prior = not use_prior
                prior = ScenePrior(ctrl.scene_name) if use_prior else None
                pipeline = PerceptionPipeline(detector, depth, verifier=verifier, prior=prior)
                room = prior.room_type if prior else "unknown"
                print(f"  Prior: {'ON' if use_prior else 'OFF'}  (room={room})")
                continue

            if head == "auto":
                auto_detect = not auto_detect
                print(f"  Auto-detect: {'ON' if auto_detect else 'OFF'}")
                continue

            if head in SHORTCUT:
                result = ctrl.step(SHORTCUT[head])
                collector.record(result)
                p = result.agent_state.position
                status = "BLOCKED" if result.agent_state.is_colliding else ("ok" if result.success else "FAIL")
                print(f"  [{ctrl.step_count:03d}] ({p.x:5.2f},{p.y:4.2f},{p.z:5.2f})  "
                      f"head={result.agent_state.heading_deg:.0f} deg  [{status}]")
                if auto_detect:
                    dets = pipeline.process(result.sensor_data.rgb, controller=ctrl)
                    _print_detections(dets)
                continue

            print(f"  Unknown: '{head}'.  Try: wasd / detect / clip / prior / auto / info / quit")

    if collector.frame_count > 0:
        try:
            path = collector.export_video(fps=5)
            print(f"  Video saved -> {path}")
        except Exception as exc:
            print(f"  Video export failed: {exc}")
    print(f"  Session dir: {collector.session_dir}")


def main():
    p = argparse.ArgumentParser(description="Perception demo — YOLO + CLIP + scene prior")
    p.add_argument("--scene", default="FloorPlan1")
    p.add_argument("--width", type=int, default=800)
    p.add_argument("--height", type=int, default=600)
    p.add_argument("--model", default="yolov8n.pt",
                   help="YOLO model: yolov8n/s/m/l/x.pt (default: yolov8n.pt)")
    p.add_argument("--confidence", type=float, default=0.3,
                   help="YOLO confidence threshold (default: 0.3)")
    p.add_argument("--clip", action="store_true",
                   help="Enable CLIP verification")
    p.add_argument("--no-prior", action="store_true",
                   help="Disable scene prior weighting")
    p.add_argument("--clip-threshold", type=float, default=0.0,
                   help="CLIP score filter threshold (default: 0.0 = no filter)")
    p.add_argument("--classes", default=None,
                   help="Path to classes.yaml (for --check-classes)")
    p.add_argument("--check-classes", action="store_true",
                   help="Diagnostic: list all objects in the scene and their class mapping status")
    p.add_argument("--verify-labels", action="store_true",
                   help="Collect a few frames with auto-labels and save annotated images for inspection")
    args = p.parse_args()

    if args.check_classes:
        _check_classes(args.scene, args.classes)
        return

    if args.verify_labels:
        _verify_labels(args.scene, args.classes)
        return

    run(scene=args.scene, width=args.width, height=args.height,
        model=args.model, confidence=args.confidence,
        use_clip=args.clip, use_prior=not args.no_prior,
        clip_threshold=args.clip_threshold)


if __name__ == "__main__":
    main()
