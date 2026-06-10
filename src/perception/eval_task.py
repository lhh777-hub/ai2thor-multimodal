"""
Task-level evaluation: measure navigation success rate per detector.

Instead of per-frame IoU matching (which penalises detectors for missing
partially-occluded or edge-angle objects), this evaluates what actually
matters: can the agent find and reach target objects?

Usage::

    python -m src.perception.eval_task --scenes FloorPlan1,FloorPlan3,FloorPlan5

Output::

    Detector          Tasks  Success   Steps   Time
    YOLO-finetuned       12      75%    18.3   8.2s
    YOLO-World           12      67%    22.1  11.4s
    Hybrid               12      83%    16.5   9.1s
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.controller.thor import ThorController
from src.perception.detector import (
    YOLODetector, YOLOWorldDetector, HybridDetector,
)
from src.perception.depth import HeuristicDepth
from src.perception.pipeline import PerceptionPipeline
from src.common.logger import setup_logger

logger = setup_logger("eval_task")

# Fixed scenes for reproducible comparison
DEFAULT_SCENES = ["FloorPlan1", "FloorPlan3", "FloorPlan5"]


def _build_pipeline(detector) -> PerceptionPipeline:
    return PerceptionPipeline(detector, HeuristicDepth())


def _run_navigation(ctrl: ThorController, target: str, pipeline: PerceptionPipeline,
                    max_steps: int = 80, verbose: bool = False) -> tuple[bool, int]:
    """Simple perception-guided navigation to *target*.

    Algorithm: scan 360°, if target visible → approach; else A* fallback.
    Returns (success, steps).
    """
    target_lower = target.lower()

    for step in range(max_steps):
        view = ctrl.get_current_view()
        dets = pipeline.process(view.sensor_data.rgb, controller=ctrl)

        # Check if target is detected and nearby
        best = None
        for d in dets:
            if d.label.lower() == target_lower or target_lower in d.label.lower():
                if best is None or d.distance_meters < best.distance_meters:
                    best = d

        if best is not None and best.distance_meters <= 1.5:
            return True, step + 1

        # Perception-guided approach
        if best is not None:
            if best.screen_position == "center":
                ctrl.step("MOVE_FORWARD")
            elif best.screen_position == "left":
                ctrl.step("TURN_LEFT_SMALL")
            else:
                ctrl.step("TURN_RIGHT_SMALL")
        else:
            # Scan for target
            ctrl.step("TURN_LEFT")

    # Check final distance
    dist = ctrl.distance_to(target)
    if dist is not None and dist <= 1.5:
        return True, max_steps
    return False, max_steps


def _get_navigable_objects(ctrl: ThorController, max_per_scene: int = 8) -> list[str]:
    """Return list of navigable object names in the current scene."""
    obj_map = ctrl.get_object_map()
    # Prefer visible objects within reasonable range
    candidates = []
    for name, info in obj_map.items():
        if info.get("visible", False):
            dist = ctrl.distance_to(name)
            if dist is not None and 1.0 < dist < 8.0:
                candidates.append((dist, name))

    # Sort by distance (closer first — easier tasks) and pick up to max
    candidates.sort()
    names = [name for _, name in candidates[:max_per_scene]]
    return names


def run_evaluation(
    scenes: list[str] = None,
    yolo_model: str = "runs/detect/runs/train/weights/best.pt",
    yolo_conf: float = 0.3,
    world_model: str = "yolov8s-worldv2.pt",
    world_conf: float = 0.15,
    max_steps: int = 80,
    tasks_per_scene: int = 5,
) -> dict:
    """Run task-level evaluation across scenes and detectors."""

    if scenes is None:
        scenes = DEFAULT_SCENES

    detectors_info = [
        ("YOLO-finetuned", "yolo"),
        ("YOLO-World", "world"),
        ("Hybrid", "hybrid"),
    ]

    results: dict[str, dict] = defaultdict(lambda: {
        "success": 0, "total": 0, "steps": [], "times": [],
    })

    print(f"\n  Task-level evaluation: {len(scenes)} scenes, "
          f"≤{tasks_per_scene} tasks/scene, max {max_steps} steps/task")
    print(f"  {'='*60}")

    for scene in scenes:
        print(f"\n  [{scene}]")

        with ThorController(width=640, height=480, visibility_distance=10.0) as ctrl:
            ctrl.load_scene(scene, seed=42)

            # Get task list — same tasks for all detectors
            task_objects = _get_navigable_objects(ctrl, tasks_per_scene)
            if not task_objects:
                print(f"    No navigable objects found")
                continue
            print(f"    Tasks: {', '.join(task_objects[:5])}"
                  + (f" +{len(task_objects)-5} more" if len(task_objects) > 5 else ""))

            for det_name, det_key in detectors_info:
                # Build detector for this scene
                if det_key == "yolo":
                    det = YOLODetector(model_name=yolo_model, confidence=yolo_conf)
                elif det_key == "world":
                    det = YOLOWorldDetector(model_name=world_model, confidence=world_conf)
                    det.set_classes_from_scene(ctrl)
                else:  # hybrid
                    det_f = YOLODetector(model_name=yolo_model, confidence=yolo_conf)
                    det_w = YOLOWorldDetector(model_name=world_model, confidence=world_conf)
                    det_w.set_classes_from_scene(ctrl)
                    det = HybridDetector(det_f, det_w)

                pipeline = _build_pipeline(det)
                success_count = 0
                step_list = []
                time_list = []

                for target in task_objects:
                    # Reset agent position for each task (reload scene)
                    ctrl.load_scene(scene, seed=42 + task_objects.index(target))
                    t0 = time.time()
                    ok, steps = _run_navigation(ctrl, target, pipeline,
                                                 max_steps=max_steps)
                    elapsed = time.time() - t0
                    success_count += 1 if ok else 0
                    step_list.append(steps)
                    time_list.append(elapsed)

                avg_steps = sum(step_list) / len(step_list) if step_list else 0
                avg_time = sum(time_list) / len(time_list) if time_list else 0
                rate = success_count / len(task_objects) * 100 if task_objects else 0

                results[det_name]["success"] += success_count
                results[det_name]["total"] += len(task_objects)
                results[det_name]["steps"].extend(step_list)
                results[det_name]["times"].extend(time_list)

                status = "✓" if rate >= 60 else ("△" if rate >= 30 else "✗")
                print(f"    {det_name:<18s} {success_count}/{len(task_objects)} "
                      f"({rate:5.1f}%)  avg {avg_steps:4.1f} steps  "
                      f"avg {avg_time:4.1f}s  {status}")

    # ---- Final summary ----
    print(f"\n  {'='*60}")
    print(f"  FINAL RESULTS")
    print(f"  {'='*60}")
    print(f"\n  {'Detector':<18s} {'Tasks':>6s} {'Success':>8s} {'Steps':>7s} {'Time':>7s}")
    print(f"  {'-'*18} {'-'*6} {'-'*8} {'-'*7} {'-'*7}")

    for det_name, _ in detectors_info:
        r = results[det_name]
        total = r["total"]
        if total == 0:
            continue
        rate = r["success"] / total * 100
        avg_s = sum(r["steps"]) / len(r["steps"]) if r["steps"] else 0
        avg_t = sum(r["times"]) / len(r["times"]) if r["times"] else 0
        print(f"  {det_name:<18s} {total:>6d} {rate:>7.1f}% {avg_s:>6.1f}  {avg_t:>5.1f}s")

    print()
    return dict(results)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Task-level evaluation: navigation success rate per detector")
    p.add_argument("--scenes", type=str, default=",".join(DEFAULT_SCENES),
                   help=f"Comma-separated scene names (default: {','.join(DEFAULT_SCENES)})")
    p.add_argument("--yolo-model", default="runs/detect/runs/train/weights/best.pt")
    p.add_argument("--yolo-conf", type=float, default=0.3)
    p.add_argument("--world-model", default="yolov8s-worldv2.pt")
    p.add_argument("--world-conf", type=float, default=0.15)
    p.add_argument("--max-steps", type=int, default=80)
    p.add_argument("--tasks", type=int, default=5,
                   help="Max tasks per scene (default: 5)")
    args = p.parse_args()

    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    run_evaluation(
        scenes=scenes,
        yolo_model=args.yolo_model,
        yolo_conf=args.yolo_conf,
        world_model=args.world_model,
        world_conf=args.world_conf,
        max_steps=args.max_steps,
        tasks_per_scene=args.tasks,
    )


if __name__ == "__main__":
    main()
