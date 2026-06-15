"""
Record a multi-task demo video on FloorPlan1.

Usage::

    python scripts/record_demo.py
    python scripts/record_demo.py --policy rule
    python scripts/record_demo.py --detector yolo-world
    python scripts/record_demo.py --detector yolo-world --confidence 0.15
    python scripts/record_demo.py --tasks "Pick up the book,Drop the book,Open the refrigerator,Go to the microwave"

Output::

    outputs/<timestamp>_demo/
      trajectory.mp4          ← annotated video
      frames/                 ← per-step annotated PNGs
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env early so OPENAI_API_KEY etc. are available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.controller.thor import ThorController
from src.perception.detector import (
    YOLODetector, YOLOWorldDetector, HybridDetector,
    create_hybrid, create_detector,
)
from src.perception.depth import HeuristicDepth
from src.perception.pipeline import PerceptionPipeline
from src.recording.collector import FrameCollector
from src.common.logger import setup_logger
from src.decision.parser import TaskParser
from src.decision.rule_policy import RulePolicy
from src.perception.spatial_memory import SpatialMemory

logger = setup_logger("record_demo")

# ---------------------------------------------------------------------------
# Import private helpers from decide.py (avoid code duplication)
# ---------------------------------------------------------------------------
from src.cli.decide import (
    _room_tour,
    _prescan_and_report,
    run_episode,
)


def _face_target(ctrl: ThorController, target_name: str,
                collector: FrameCollector) -> None:
    """Rotate the agent to face *target_name*, recording each turn step."""
    obj_map = ctrl.get_object_map()
    info = None
    tl = target_name.lower().replace(" ", "")
    for k, v in obj_map.items():
        if k.lower() == target_name.lower() or k.lower().replace(" ", "") == tl:
            info = v
            break
    if info is None:
        return

    target_pos = info["position"]
    for result in ctrl.look_at(target_pos):
        collector.record(result)


def _build_detector(detector_type: str, model: str, confidence: float,
                    controller=None):
    """Build a detector matching the --detector option."""
    if detector_type == "yolo-world":
        det = YOLOWorldDetector(model_name=model or "yolov8s-worldv2.pt",
                                confidence=confidence)
        if controller is not None:
            det.set_classes_from_scene(controller)
        return det

    if detector_type == "finetuned":
        return YOLODetector(model_name=model or "runs/detect/runs/train/weights/best.pt",
                            confidence=confidence)

    # hybrid (default)
    return create_hybrid(
        finetuned_model=model or "runs/detect/runs/train/weights/best.pt",
        finetuned_conf=confidence,
        world_model="yolov8s-worldv2.pt",
        world_conf=0.15,
        controller=controller,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a multi-task demo video")
    parser.add_argument("--scene", default="FloorPlan1")
    parser.add_argument("--policy", default="rule+vlm",
                        choices=["rule", "rule+vlm"],
                        help="rule = deterministic, rule+vlm = Rule + VLM recovery")
    parser.add_argument("--detector", default="yolo-world",
                        choices=["hybrid", "finetuned", "yolo-world"],
                        help="Detection mode (default: yolo-world)")
    parser.add_argument("--tasks", default="Go to the refrigerator,Open the refrigerator,Close the refrigerator,Pick up the mug,Drop the mug",
                        help="Comma-separated task instructions")
    parser.add_argument("--max-steps", type=int, default=200,
                        help="Max steps per task")
    parser.add_argument("--fps", type=int, default=5,
                        help="Output video FPS")
    parser.add_argument("--model", default=None,
                        help="YOLO model path (default depends on --detector)")
    parser.add_argument("--confidence", type=float, default=None,
                        help="YOLO confidence threshold (default: 0.3 finetuned, 0.15 yolo-world)")
    parser.add_argument("--list", action="store_true",
                        help="Just list all object types in the scene and exit")
    args = parser.parse_args()

    # ---- Set defaults based on detector type ----
    model = args.model
    confidence = args.confidence
    if model is None:
        model = ("yolov8s-worldv2.pt" if args.detector == "yolo-world"
                 else "runs/detect/runs/train/weights/best.pt")
    if confidence is None:
        confidence = 0.15 if args.detector == "yolo-world" else 0.3

    # ---- List-only mode ----
    if args.list:
        with ThorController(width=300, height=300) as ctrl:
            ctrl.load_scene(args.scene)
            obj_map = ctrl.get_object_map()
            print(f"\n  Objects in {args.scene} ({len(obj_map)} types):\n")
            for name in sorted(obj_map.keys()):
                p = obj_map[name]["position"]
                print(f"    {name:<25s}  ({p.x:6.1f}, {p.y:5.1f}, {p.z:6.1f})")
        return

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    use_vlm = (args.policy == "rule+vlm")

    print(f"{'='*60}")
    print(f"  Demo Recorder")
    print(f"  Scene:    {args.scene}")
    print(f"  Policy:   {args.policy}")
    print(f"  Detector: {args.detector}  |  Model: {model}  |  Conf: {confidence}")
    print(f"  Tasks:    {len(tasks)}")
    for i, t in enumerate(tasks, 1):
        print(f"    {i}. {t}")
    print(f"{'='*60}")

    # ---- Build recorder ----
    collector = FrameCollector(scene=f"{args.scene}_demo")

    # ---- Build policy ----
    policy = RulePolicy(use_vlm_recovery=use_vlm)
    use_llm_advisor = use_vlm

    # ---- Run ----
    with ThorController(width=800, height=600, render_depth=True) as ctrl:
        ctrl.load_scene(args.scene)
        view = ctrl.get_current_view()
        collector.record(view)

        # ---- Build detector (after scene load for vocab narrowing) ----
        detector = _build_detector(args.detector, model, confidence,
                                   controller=ctrl)
        depth = HeuristicDepth()
        pipeline = PerceptionPipeline(detector, depth)

        # ---- Room tour → spatial memory ----
        print(f"\n  [Room Tour] Exploring {args.scene}...")
        spatial_memory = _prescan_and_report(ctrl, pipeline, "", verbose=True)

        # Reload scene to reset agent to spawn position
        ctrl.load_scene(args.scene)
        view = ctrl.get_current_view()
        collector.record(view)

        print(f"\n  [Ready] {len(ctrl.get_object_map())} object types in scene")

        # ---- Execute each task ----
        frames_dir = os.path.join(collector.session_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        for i, task_text in enumerate(tasks):
            print(f"\n{'─'*60}")
            print(f"  [{i+1}/{len(tasks)}] {task_text}")
            print(f"{'─'*60}")

            # Per-task subdirectory so step numbers don't collide across tasks
            task_frames_dir = os.path.join(frames_dir, f"task_{i+1:02d}")
            os.makedirs(task_frames_dir, exist_ok=True)

            success, steps, reason = run_episode(
                task_text, policy, pipeline, ctrl, collector,
                max_steps=args.max_steps,
                verbose=True,
                save_frames=True,
                frames_dir=task_frames_dir,
                use_llm_advisor=use_llm_advisor,
                spatial_memory=spatial_memory,
            )

            status = "✓" if success else "✗"
            print(f"  [{i+1}/{len(tasks)}] {status} {steps} steps — {reason}")

            # Pause between tasks for a clean video cut
            if i < len(tasks) - 1:
                time.sleep(1.5)

            # After a navigation task, face the target for a clean camera shot
            if success:
                try:
                    spec = TaskParser.interpret(task_text)
                    if spec.is_navigation:
                        _face_target(ctrl, spec.target, collector)
                except Exception:
                    pass

        # ---- Export ----
        print(f"\n{'='*60}")
        print(f"  Exporting video...")
        video_path = collector.export_video(fps=args.fps)
        frames_path = collector.export_frames()
        print(f"  Video:  {video_path}")
        print(f"  Frames: {frames_path}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
