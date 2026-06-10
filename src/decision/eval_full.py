"""
Unified evaluation: Rule vs LLM(VLM) on navigation + interaction tasks.

Modes:
  quick — lightweight navigation-only eval, single policy (was eval_policy.py)
  demo  — curated episodes for project acceptance, trajectories (was eval_demo.py)
  full  — multi-policy comparison with spatial memory, traces (default)

Policies:
  rule       — RulePolicy (deterministic, no API)
  rule+vlm   — RulePolicy with VLM recovery on target loss
  llm        — LLMPolicy (multimodal VLM every step)

Usage::

    # Quick: 3 scenes, 10 nav tasks, Rule only
    python -m src.decision.eval_full --mode quick --scenes 3 --tasks 10

    # Demo: curated obvious-object tasks, mixed nav+interact, trajectories
    python -m src.decision.eval_full --mode demo --scenes 5 --tasks-per-scene 10

    # Full: multi-policy, spatial memory, detailed traces (default)
    python -m src.decision.eval_full --mode full --policies rule,rule+vlm --scenes 5 --tasks 20

    # Explicit scene list (all modes)
    python -m src.decision.eval_full --scene-list FloorPlan1,FloorPlan3,FloorPlan5 --policies rule,llm

Output::

    outputs/eval_{mode}_{timestamp}/
      summary.txt        ← results table
      details.csv        ← per-task results
      traces/            ← episode traces (demo/full mode)
      failure_report.md  ← failure analysis (demo/full mode)
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

# Load .env so OPENAI_API_KEY / OPENAI_BASE_URL / VLM_MODEL are available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.controller.thor import ThorController
from src.perception.detector import create_hybrid
from src.perception.depth import HeuristicDepth
from src.perception.pipeline import PerceptionPipeline
from src.perception.prior import ScenePrior
from src.decision.rule_policy import RulePolicy
from src.decision.llm_policy import LLMPolicy
from src.decision.parser import TaskParser
from src.decision.trace import EpisodeTrace
from src.decision.visualize import plot_episode_summary, generate_failure_report
from src.common.types import Vec3
from src.common.logger import setup_logger
from src.common.utils import img_to_b64, eval_distance, thor_to_yolo, safe_filename
from src.perception.spatial_memory import SpatialMemory

logger = setup_logger("eval_full")

SCENE_POOL = [f"FloorPlan{i}" for i in range(1, 31)]

# ===========================================================================
# Data types (shared across all modes)
# ===========================================================================

@dataclass
class TaskDef:
    text: str           # e.g. "Go to the refrigerator"
    target: str         # e.g. "refrigerator"
    task_type: str      # "navigation" | "interaction"
    interact_action: str  # "" | "OPEN" | "PICKUP" | "TOGGLE"
    in_scene: bool      # does the target exist in the scene?
    distance_m: float = 0.0  # distance from spawn (demo mode)


@dataclass
class TaskResult:
    scene: str
    policy: str          # "rule" | "rule+vlm" | "llm"
    task: str
    target: str
    task_type: str
    in_scene: bool
    success: bool
    steps: int
    elapsed: float
    reason: str
    optimal_steps: float = 0.0
    spl: float = 0.0
    distance_m: float = 0.0   # for distance-based analysis (demo mode)


# ===========================================================================
# Shared: spatial memory builder (room tour)
# ===========================================================================

def _build_spatial_memory(ctrl: ThorController, pipeline_tour: PerceptionPipeline,
                          num_stops: int = 4) -> SpatialMemory:
    """Walk to diverse vantage points, do 360° scans, build vision-based index."""
    mem = SpatialMemory()
    positions = ctrl._get_reachable_positions_raw()
    stops = []
    if positions:
        cx = sum(p.x for p in positions) / max(len(positions), 1)
        cz = sum(p.z for p in positions) / max(len(positions), 1)
        ranked = sorted(positions, key=lambda p: -(p.x - cx) ** 2 - (p.z - cz) ** 2)
        for p in ranked:
            if len(stops) >= num_stops: break
            if not any((p.x - s.x) ** 2 + (p.z - s.z) ** 2 < 1.5 for s in stops):
                stops.append(p)
        for pos in stops:
            try:
                for _ in ctrl.navigate_to(pos): pass
            except Exception: pass
    for _ in range(max(len(stops) + 1, 3)):
        for __ in range(6):
            ctrl.step("TURN_LEFT_SMALL")
            ctrl.step("TURN_LEFT_SMALL")
            view = ctrl.get_current_view()
            dets = pipeline_tour.process(view.sensor_data.rgb, controller=ctrl)
            s = view.agent_state
            mem.ingest_scan(dets, view.sensor_data.depth,
                            s.position.x, s.position.y, s.position.z,
                            s.heading_deg, s.horizon_deg, min_conf=0.2)
    ctrl.load_scene(ctrl.scene_name, seed=42)  # reset to spawn
    return mem


# ===========================================================================
# Shared: episode runner
# ===========================================================================

def _run_one_task(task: TaskDef, ctrl: ThorController,
                  pipeline: PerceptionPipeline, policy,
                  policy_type: str = "rule",
                  max_steps: int = 80,
                  spatial_memory: SpatialMemory | None = None,
                  traces_dir: str | None = None) -> TaskResult:
    """Run a single task. Returns TaskResult."""

    optimal_steps: float = 0.0

    def _result(success: bool, steps_: int, reason_: str) -> TaskResult:
        spl = 0.0
        if optimal_steps > 0:
            spl = (1.0 if success else 0.0) * optimal_steps / max(steps_, optimal_steps)
        elapsed = round(time.time() - t0, 1)
        trace.success = success
        trace.final_reason = reason_
        trace.total_steps = steps_
        trace.elapsed_s = elapsed
        trace.spl = round(spl, 4)
        if traces_dir:
            safe = safe_filename(task.text)
            fname = f"{ctrl.scene_name}_{policy_type}_{safe}"
            try:
                trace.save_json(os.path.join(traces_dir, fname + ".json"))
                plot_episode_summary(trace, os.path.join(traces_dir, fname + ".png"))
            except Exception:
                pass
        return TaskResult(
            scene=ctrl.scene_name, policy=policy_type,
            task=task.text, target=task.target,
            task_type=task.task_type, in_scene=task.in_scene,
            success=success, steps=steps_, elapsed=elapsed, reason=reason_,
            optimal_steps=optimal_steps, spl=round(spl, 4),
            distance_m=task.distance_m,
        )

    try:
        task_spec = TaskParser.interpret(task.text)
    except ValueError:
        return _result(False, 0, "Parse error")

    policy.reset()
    t0 = time.time()
    failed_a_star = 0

    # Compute optimal path for SPL
    if spatial_memory is not None:
        mem_pos = spatial_memory.lookup(task_spec.target)
        if mem_pos is not None:
            path = ctrl.plan_path(Vec3(x=mem_pos.x, y=0, z=mem_pos.z))
            if path:
                optimal_steps = len(path)
    if task_spec.is_interaction and optimal_steps > 0:
        optimal_steps += 1

    # Trace recording
    spawn_pos = ctrl.agent_state.position
    trace = EpisodeTrace(
        scene=ctrl.scene_name, policy=policy_type,
        task=task.text, target=task.target, task_type=task.task_type,
        success=False, final_reason="", total_steps=0,
        elapsed_s=0.0, optimal_steps=optimal_steps, spl=0.0,
        spawn_x=spawn_pos.x, spawn_z=spawn_pos.z,
    )
    target_pos: Vec3 | None = None
    if spatial_memory is not None:
        mem_pos = spatial_memory.lookup(task_spec.target)
        if mem_pos is not None:
            target_pos = mem_pos
            trace.target_x = mem_pos.x
            trace.target_z = mem_pos.z
    try:
        raw = ctrl._get_reachable_positions_raw()
        trace.reachable_positions = [(p.x, p.z) for p in raw[:2000]]
    except Exception:
        pass

    # VLM recovery setup (rule+vlm only)
    use_vlm_verify = (policy_type == "rule+vlm")
    vlm_attempts = 0
    vlm_max_attempts = 3
    api_key = os.environ.get("OPENAI_API_KEY", "") if use_vlm_verify else ""
    base_url = os.environ.get("OPENAI_BASE_URL", "") if use_vlm_verify else ""
    model = os.environ.get("VLM_MODEL", "qwen-vl-max") if use_vlm_verify else ""

    for step in range(max_steps):
        view = ctrl.get_current_view()
        detections = pipeline.process(view.sensor_data.rgb, controller=ctrl)

        if policy_type == "llm" and step % 5 == 0:
            elapsed = time.time() - t0
            print(f"      step {step}/{max_steps} ({elapsed:.0f}s)...", end="\r")

        decision = policy.decide(
            rgb=view.sensor_data.rgb, detections=detections,
            agent_state=view.agent_state, task=task_spec,
            step_count=step, depth_frame=view.sensor_data.depth,
            spatial_memory=spatial_memory,
        )

        # Record step in trace
        st = view.agent_state
        tl = task_spec.target.lower()
        target_detected = False
        target_dist = -1.0
        top_labels = []
        for d in detections[:5]:
            top_labels.append(f"{d.label}({d.distance_meters:.1f}m)")
            if d.label.lower() == tl or tl in d.label.lower():
                target_detected = True
                if d.distance_meters > 0:
                    target_dist = d.distance_meters
        trace.add_step(
            step=step, x=st.position.x, y=st.position.y, z=st.position.z,
            heading_deg=st.heading_deg, action=decision.action,
            reason=decision.reason, is_colliding=st.is_colliding,
            target_detected=target_detected, target_distance=target_dist,
            top_detections=", ".join(top_labels) if top_labels else "(none)",
        )

        # Terminal?
        if decision.done or decision.action == "STOP":
            if decision.action != "STOP" and not decision.action.startswith("STOP"):
                if decision.action.startswith("INTERACT_"):
                    ctrl._interact_target = task_spec.target
                ctrl.step(decision.action)
            dist = eval_distance(ctrl, task.target, target_pos)
            ok = dist is not None and dist <= 1.0
            if task.task_type == "navigation":
                reason = f"Stopped at {dist:.1f}m" if dist else "Target not in scene"
            else:
                reason = (f"Interaction executed (at {dist:.1f}m)" if ok
                          else f"Stopped {dist:.1f}m from target" if dist
                          else f"Target '{task.target}' not reachable")
            return _result(ok, step + 1, reason)

        # VLM_EXPLORE
        if decision.action == "VLM_EXPLORE":
            if vlm_attempts >= vlm_max_attempts:
                return _result(False, step + 1, f"VLM exhausted ({vlm_max_attempts} attempts)")
            if use_vlm_verify and api_key:
                vlm_attempts += 1
                print(f"      [VLM explore {vlm_attempts}/{vlm_max_attempts}] "
                      f"searching for '{task_spec.target}'...", end="\r")
                for v_step in range(15):
                    view2 = ctrl.get_current_view()
                    rgb2 = view2.sensor_data.rgb
                    dets2 = pipeline.process(rgb2, controller=ctrl)
                    _tl = task_spec.target.lower()
                    target_visible = None
                    _td = 999.0
                    for d in dets2:
                        if d.label.lower() == _tl or _tl in d.label.lower():
                            dd = d.distance_meters if d.distance_meters > 0 else 999
                            if dd < _td: _td = dd; target_visible = d
                    if target_visible is not None and _td <= 1.5:
                        print(f"      [VLM] target at {_td:.1f}m — handing back to Rule")
                        break
                    nearby = [d for d in dets2 if d.distance_meters > 0 and d.distance_meters < 5.0]
                    nearby_text = "\n".join(
                        f"  {d.label} ({d.screen_position}, {d.distance_meters:.1f}m, conf={d.confidence:.2f})"
                        for d in sorted(nearby, key=lambda d_: d_.distance_meters)[:8]
                    ) or "  (nothing nearby)"
                    all_labels = ", ".join(sorted(set(d.label for d in dets2[:15]))) or "nothing"
                    if target_visible is not None:
                        dir_hint = (f"TARGET VISIBLE! '{task_spec.target}' at {_td:.1f}m, "
                                    f"screen: {target_visible.screen_position}. Turn to face, "
                                    f"then APPROACH. FOUND when ≤1.5m.")
                    elif v_step < 5:
                        dir_hint = "Target not visible. Turn left to scan."
                    elif v_step < 10:
                        dir_hint = "Still searching. Try LOOK_UP (counters/shelves) or move."
                    else:
                        dir_hint = "Target not found. GIVE_UP if hopeless."
                    try:
                        from openai import OpenAI
                        client = OpenAI(api_key=api_key, base_url=base_url or None)
                        response = client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": (
                                    "Guide robot to FIND and APPROACH target. "
                                    "Actions: MOVE_FORWARD, MOVE_BACK, TURN_LEFT, TURN_RIGHT, "
                                    "TURN_LEFT_SMALL, TURN_RIGHT_SMALL, LOOK_UP, LOOK_DOWN, "
                                    "FOUND (≤1.5m), GIVE_UP. JSON: {\"action\":\"...\",\"reason\":\"...\"}"
                                )},
                                {"role": "user", "content": [
                                    {"type": "image_url", "image_url": {
                                        "url": f"data:image/jpeg;base64,{img_to_b64(rgb2)}", "detail": "auto"}},
                                    {"type": "text", "text": (
                                        f"Target: '{task_spec.target}'\nStep {v_step+1}/15\n"
                                        f"All visible: {all_labels}\nNearby (<5m):\n{nearby_text}\n{dir_hint}"
                                    )},
                                ]},
                            ], max_tokens=128, temperature=0.0)
                        text = (response.choices[0].message.content or "").strip()
                        if not text: continue
                        import re, json
                        m = re.search(r"\{.*\}", text, re.DOTALL)
                        if not m: continue
                        data = json.loads(m.group(0))
                        act = data.get("action", "TURN_LEFT")
                        if act in ("GIVE_UP", "FOUND"): break
                        if act not in ("STOP", "FOUND", "GIVE_UP"): ctrl.step(act)
                    except Exception: break
            else:
                ctrl.step("TURN_LEFT")
            continue

        # A* navigation
        if decision.action.startswith("NAVIGATE_TO:"):
            parts = decision.action.split(":")
            nav_ok = False
            if len(parts) >= 4:
                tx, tz = float(parts[2]), float(parts[3])
                for _ in ctrl.navigate_to(Vec3(x=tx, y=0, z=tz)): nav_ok = True
            else:
                for _ in ctrl.navigate_to_object(parts[1]): nav_ok = True
            if not nav_ok:
                failed_a_star += 1
                if failed_a_star >= 3:
                    return _result(False, step + 1, f"A* failed {failed_a_star}x")
            else:
                failed_a_star = 0
                if task_spec.is_navigation:
                    dist = eval_distance(ctrl, task_spec.target, target_pos)
                    if dist is not None and dist <= 1.0:
                        return _result(True, step + 1, f"A* arrived at {dist:.1f}m")
            continue

        if decision.action != "STOP":
            if decision.action.startswith("INTERACT_"):
                ctrl._interact_target = task_spec.target
            ctrl.step(decision.action)

    # Max steps
    dist = eval_distance(ctrl, task.target, target_pos)
    ok = dist is not None and dist <= 1.0
    return _result(ok, max_steps, f"Max steps ({max_steps})")


# ===========================================================================
# Mode: quick (was eval_policy.py)
# ===========================================================================

# Absent-object distractors
ABSENT_POOL = [
    "toilet", "bathtub", "piano", "bicycle", "motorcycle",
    "airplane", "train", "boat", "traffic light", "fire hydrant",
    "parking meter", "bench", "elephant", "giraffe", "zebra",
    "tennis racket", "skateboard", "surfboard", "baseball bat", "ski",
]


def _quick_pick_present(ctrl: ThorController, count: int,
                        pipeline: PerceptionPipeline | None = None) -> list[str]:
    """Pick *count* object names that exist AND are detectable by YOLO."""
    obj_map = ctrl.get_object_map()
    candidates = []
    for name, info in obj_map.items():
        if info.get("visible", False):
            dist = ctrl.distance_to(name)
            if dist is not None and 1.0 < dist < 8.0:
                candidates.append((dist, thor_to_yolo(name)))
    random.shuffle(candidates)
    picked = []
    if pipeline is not None and candidates:
        view = ctrl.get_current_view()
        dets = pipeline.process(view.sensor_data.rgb, controller=ctrl)
        detected_labels = {d.label.lower() for d in dets}
        for _, yolo_name in candidates:
            if len(picked) >= count: break
            nl = yolo_name.lower()
            if nl in detected_labels or any(nl in dl or dl in nl for dl in detected_labels):
                picked.append(yolo_name)
            else:
                logger.info("Skipping '%s' — not detected by YOLO from spawn", yolo_name)
    else:
        picked = [name for _, name in candidates[:count]]
    return picked


def _quick_pick_absent(ctrl: ThorController, count: int) -> list[str]:
    """Pick *count* objects NOT in the scene."""
    obj_map = ctrl.get_object_map()
    scene_names = {name.lower() for name in obj_map}
    candidates = [o for o in ABSENT_POOL if o.lower() not in scene_names]
    try:
        from src.perception.class_config import load_config
        cfg = load_config()
        for name in cfg.class_names:
            if name.lower() not in scene_names and name.lower() not in {c.lower() for c in candidates}:
                candidates.append(name)
    except Exception: pass
    random.shuffle(candidates)
    return candidates[:count]


def _quick_gen_tasks(ctrl: ThorController, pipeline: PerceptionPipeline,
                     total: int, present_ratio: float = 0.7) -> list[TaskDef]:
    """Generate nav-only tasks (present + absent)."""
    num_present = max(1, int(total * present_ratio))
    num_absent = total - num_present
    present = _quick_pick_present(ctrl, num_present, pipeline)
    absent = _quick_pick_absent(ctrl, num_absent)
    tasks = []
    for t in present: tasks.append(TaskDef(text=f"Go to the {t}", target=t, task_type="navigation", interact_action="", in_scene=True))
    for t in absent: tasks.append(TaskDef(text=f"Go to the {t}", target=t, task_type="navigation", interact_action="", in_scene=False))
    random.shuffle(tasks)
    return tasks


def _run_quick(scenes: list[str], tasks_per_scene: int, seed: int,
               max_steps: int = 80, present_ratio: float = 0.7,
               yolo_model: str = "runs/detect/runs/train/weights/best.pt",
               yolo_conf: float = 0.3) -> dict:
    """Quick nav-only evaluation, single policy (rule)."""
    random.seed(seed)
    results: list[TaskResult] = []
    started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("outputs", f"eval_quick_{ts}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n  Quick Evaluation — Navigation only, Rule policy")
    print(f"  {'='*60}")
    print(f"  Scenes:  {', '.join(scenes)}")
    print(f"  Tasks:   {tasks_per_scene}/scene")
    print(f"  Max steps: {max_steps}")

    for scene in scenes:
        print(f"\n  [{scene}]")
        with ThorController(width=640, height=480, visibility_distance=10.0,
                            render_depth=True) as ctrl:
            ctrl.load_scene(scene, seed=42)
            detector = create_hybrid(finetuned_model=yolo_model, finetuned_conf=yolo_conf,
                                     world_model="yolov8s-worldv2.pt", world_conf=0.15,
                                     controller=ctrl)
            depth = HeuristicDepth()
            prior = ScenePrior(scene)
            pipeline = PerceptionPipeline(detector, depth, prior=prior)
            policy = RulePolicy(max_steps=max_steps)
            tasks = _quick_gen_tasks(ctrl, pipeline, tasks_per_scene, present_ratio)
            for task in tasks:
                ctrl.load_scene(scene, seed=42)
                result = _run_one_task(task, ctrl, pipeline, policy, policy_type="rule",
                                       max_steps=max_steps)
                result.in_scene = task.in_scene  # override from task def
                results.append(result)
                status = "✓" if result.success else "✗"
                mark = " " if task.in_scene else "A"
                print(f"    {status} [{mark}] {task.target:<20s} {result.steps:>3d}s  {result.reason}")

    finished = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _print_quick_summary(results, scenes, started, finished)
    _save_results(results, scenes, started, finished, output_dir, ("rule",), mode="quick")
    print(f"\n  Report saved → {output_dir}/")
    return {"results": results, "output_dir": output_dir}


# ===========================================================================
# Mode: demo (was eval_demo.py)
# ===========================================================================

# Large, distinctive objects for curated demo tasks
OBVIOUS_OBJECTS: set[str] = {
    "refrigerator", "microwave", "oven", "sink", "cabinet", "countertop",
    "couch", "sofa", "chair", "dining table", "coffee table", "tv",
    "television", "bookshelf", "shelving unit",
    "bed", "dresser", "desk", "night stand",
    "toilet", "bathtub", "mirror",
    "potted plant", "garbage can", "stool", "bench",
}
SKIP_OBJECTS: set[str] = {
    "apple", "banana", "bottle", "bowl", "bread", "butterknife", "carrot",
    "cell phone", "credit card", "cup", "dishsponge", "egg", "fork",
    "knife", "lettuce", "mug", "pan", "pepper shaker", "plate", "potato",
    "remote", "salt shaker", "soap bottle", "spatula", "spoon",
    "tomato", "vase", "wine bottle", "wine glass",
}
DISTANCE_MIN = 2.0
DISTANCE_MAX = 5.0
PREFERRED_DISTANCE = (2.5, 4.5)
DEMO_SCENES = ["FloorPlan1", "FloorPlan3", "FloorPlan5", "FloorPlan7",
               "FloorPlan10", "FloorPlan12"]


def _is_obvious(yolo_name: str) -> bool:
    name = yolo_name.lower().strip()
    if name in SKIP_OBJECTS: return False
    if name in OBVIOUS_OBJECTS: return True
    return any(obv in name or name in obv for obv in OBVIOUS_OBJECTS)


def _demo_discover_tasks(ctrl: ThorController, mem: SpatialMemory,
                         total: int = 10, nav_ratio: float = 0.6) -> list[TaskDef]:
    """Discover curated tasks: obvious objects at moderate distance."""
    obj_map = ctrl.get_object_map()
    metadata_objects = ctrl._ctrl.last_event.metadata.get("objects", [])

    def _collect(d_min, d_max, require_obvious):
        result = []
        for thor_name, info in obj_map.items():
            yolo_name = thor_to_yolo(thor_name)
            if not mem.lookup(yolo_name): continue
            if require_obvious and not _is_obvious(yolo_name): continue
            dist = ctrl.distance_to(thor_name)
            if dist is None or dist < d_min or dist > d_max: continue
            props = set()
            for obj in metadata_objects:
                if obj.get("objectType") == thor_name:
                    if obj.get("openable"): props.add("openable")
                    if obj.get("pickupable"): props.add("pickupable")
                    if obj.get("toggleable"): props.add("toggleable")
                    break
            result.append({"yolo": yolo_name, "thor": thor_name, "dist": dist, "props": props})
        return result

    candidates = _collect(DISTANCE_MIN, DISTANCE_MAX, True)
    # Tier fallbacks
    for tier_dmin, tier_dmax, tier_obvious in [
        (DISTANCE_MIN, DISTANCE_MAX, False),
        (2.0, 6.5, True),
        (2.0, 7.0, False),
    ]:
        if len(candidates) >= total: break
        extra = _collect(tier_dmin, tier_dmax, tier_obvious)
        seen = {c["thor"] for c in candidates}
        for c in extra:
            if c["thor"] not in seen:
                candidates.append(c); seen.add(c["thor"])

    candidates.sort(key=lambda c: min(abs(c["dist"] - PREFERRED_DISTANCE[0]),
                                      abs(c["dist"] - PREFERRED_DISTANCE[1])))
    total = min(total, len(candidates))

    int_pool = []
    for c in candidates:
        action = None
        if "openable" in c["props"]: action = "OPEN"
        elif "toggleable" in c["props"]: action = "TOGGLE"
        elif "pickupable" in c["props"]: action = "PICKUP"
        if action: int_pool.append({**c, "interact_action": action})

    num_int = min(len(int_pool), max(2, int(total * (1 - nav_ratio))))
    num_nav = total - num_int
    tasks: list[TaskDef] = []
    used_thor = set()
    random.shuffle(int_pool)
    verbs = {"OPEN": "Open the", "PICKUP": "Pick up the", "TOGGLE": "Toggle on the"}
    for c in int_pool[:num_int]:
        tasks.append(TaskDef(text=f"{verbs[c['interact_action']]} {c['yolo']}",
                             target=c["yolo"], task_type="interaction",
                             interact_action=c["interact_action"], in_scene=True,
                             distance_m=c["dist"]))
        used_thor.add(c["thor"])
    nav_pool = [c for c in candidates if c["thor"] not in used_thor]
    if len(nav_pool) < num_nav:
        nav_pool.extend(c for c in candidates if c["thor"] in used_thor)
    random.shuffle(nav_pool)
    for c in nav_pool[:num_nav]:
        tasks.append(TaskDef(text=f"Go to the {c['yolo']}", target=c["yolo"],
                             task_type="navigation", interact_action="", in_scene=True,
                             distance_m=c["dist"]))
    random.shuffle(tasks)
    return tasks[:total]


def _run_demo(scenes: list[str], tasks_per_scene: int, seed: int,
              policies: tuple[str, ...] = ("rule",),
              max_steps: int = 80,
              yolo_model: str = "runs/detect/runs/train/weights/best.pt") -> dict:
    """Demo evaluation with curated tasks and trajectory traces."""
    random.seed(seed)
    started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("outputs", f"eval_demo_{ts}")
    traces_dir = os.path.join(output_dir, "traces")
    os.makedirs(traces_dir, exist_ok=True)

    policy_labels = {"rule": "Rule", "rule+vlm": "Rule+VLM"}
    all_policies = list(policies)

    print(f"\n{'='*65}")
    print(f"  DEMO EVALUATION — Curated episodes")
    print(f"  Policies:  {', '.join(p.upper() for p in all_policies)}")
    print(f"  Scenes:    {', '.join(scenes)}")
    print(f"  Tasks:     {tasks_per_scene}/scene")
    print(f"  Strategy:  OBVIOUS objects + moderate distance ({DISTANCE_MIN}–{DISTANCE_MAX}m)")
    print(f"  Output:    {output_dir}")
    print(f"{'='*65}")

    all_results: list[TaskResult] = []

    for scene_idx, scene in enumerate(scenes):
        print(f"\n{'─'*65}")
        print(f"  [{scene_idx+1}/{len(scenes)}] {scene}")

        with ThorController(width=640, height=480, visibility_distance=10.0,
                            render_depth=True) as ctrl:
            ctrl.load_scene(scene, seed=42)
            depth = HeuristicDepth()
            prior = ScenePrior(scene)

            detector_std = create_hybrid(finetuned_model=yolo_model, finetuned_conf=0.3,
                                         world_model="yolov8s-worldv2.pt", world_conf=0.15,
                                         controller=ctrl)
            pipeline_std = PerceptionPipeline(detector_std, depth, prior=prior)
            detector_tour = create_hybrid(finetuned_model=yolo_model, finetuned_conf=0.2,
                                          world_model="yolov8s-worldv2.pt", world_conf=0.10,
                                          controller=ctrl)
            pipeline_tour = PerceptionPipeline(detector_tour, depth, prior=prior)
            mem = _build_spatial_memory(ctrl, pipeline_tour)
            tasks = _demo_discover_tasks(ctrl, mem, tasks_per_scene)
            print(f"  Spatial memory: {len(mem)} objects  |  Tasks: {len(tasks)}")
            for t in tasks:
                tag = "[nav]" if t.task_type == "navigation" else "[int]"
                print(f"    {tag} {t.text:<45s}  dist={t.distance_m:.1f}m")
            if not tasks: continue

            for pt in all_policies:
                print(f"\n  --- {policy_labels.get(pt, pt)} ---")
                policy = RulePolicy(use_vlm_recovery=(pt == "rule+vlm"))
                for i, task in enumerate(tasks):
                    ctrl.load_scene(scene, seed=42)
                    result = _run_one_task(task, ctrl, pipeline_std, policy,
                                           policy_type=pt, max_steps=max_steps,
                                           spatial_memory=mem, traces_dir=traces_dir)
                    all_results.append(result)
                    status = "OK" if result.success else "FAIL"
                    print(f"    {i+1:>2}. [{status}] {result.steps:>3d}s | {result.reason[:55]}")

    finished = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _print_summary(all_results, scenes, started, finished, all_policies)
    _save_results(all_results, scenes, started, finished, output_dir, all_policies, mode="demo")

    # Failure report
    all_traces = []
    for fname in sorted(os.listdir(traces_dir)):
        if fname.endswith(".json"):
            try: all_traces.append(EpisodeTrace.load_json(os.path.join(traces_dir, fname)))
            except Exception: pass
    if all_traces:
        generate_failure_report(all_traces, os.path.join(output_dir, "failure_report.md"))
        print(f"\n  Failure report → {output_dir}/failure_report.md")
    print(f"  Traces ({len(all_traces)} episodes) → {traces_dir}/")
    print(f"  Report → {output_dir}/")
    return {"results": all_results, "output_dir": output_dir}


# ===========================================================================
# Mode: full (default)
# ===========================================================================

def _full_gen_tasks(ctrl: ThorController, pipeline: PerceptionPipeline,
                    total: int = 20, nav_ratio: float = 0.6,
                    spatial_memory: SpatialMemory | None = None) -> list[TaskDef]:
    """Generate nav+interact tasks filtered by spatial memory."""
    obj_map = ctrl.get_object_map()
    num_nav = int(total * nav_ratio)
    num_interact = total - num_nav
    all_objects = []
    for name, info in obj_map.items():
        yolo = thor_to_yolo(name)
        if spatial_memory is not None and not spatial_memory.lookup(yolo): continue
        props = set()
        for obj in ctrl._ctrl.last_event.metadata["objects"]:
            if obj["objectType"] == name:
                if obj.get("openable"): props.add("openable")
                if obj.get("pickupable"): props.add("pickupable")
                if obj.get("toggleable"): props.add("toggleable")
                break
        dist = ctrl.distance_to(name)
        if dist is not None and 2.0 < dist < 8.0:
            all_objects.append((yolo, dist, name, props))
    if len(all_objects) < total:
        print(f"  ⚠ Only {len(all_objects)} objects — reducing task count")
        total = max(len(all_objects), 4)
        num_nav = int(total * nav_ratio); num_interact = total - num_nav
    random.shuffle(all_objects)
    tasks: list[TaskDef] = []
    nav_cands = [(y, d) for y, d, _, _ in all_objects]
    random.shuffle(nav_cands)
    for yolo_name, _ in nav_cands[:num_nav]:
        tasks.append(TaskDef(text=f"Go to the {yolo_name}", target=yolo_name,
                             task_type="navigation", interact_action="", in_scene=True))
    int_cands = []
    verbs = {"OPEN": "Open the", "PICKUP": "Pick up the", "TOGGLE": "Toggle on the"}
    for yolo_name, dist, thor_name, props in all_objects:
        if "openable" in props: int_cands.append((yolo_name, "OPEN"))
        elif "pickupable" in props: int_cands.append((yolo_name, "PICKUP"))
        elif "toggleable" in props: int_cands.append((yolo_name, "TOGGLE"))
    random.shuffle(int_cands)
    for yolo_name, action in int_cands[:num_interact]:
        tasks.append(TaskDef(text=f"{verbs[action]} {yolo_name}", target=yolo_name,
                             task_type="interaction", interact_action=action, in_scene=True))
    random.shuffle(tasks)
    return tasks[:total]


def _run_full(scenes: list[str], tasks_per_scene: int, seed: int,
              policies: tuple[str, ...] = ("rule", "rule+vlm"),
              max_steps: int = 80,
              yolo_model: str = "runs/detect/runs/train/weights/best.pt") -> dict:
    """Full multi-policy evaluation with spatial memory and traces."""
    random.seed(seed)
    started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("outputs", f"eval_full_{ts}")
    traces_dir = os.path.join(output_dir, "traces")
    os.makedirs(traces_dir, exist_ok=True)

    policy_labels = {"rule": "RulePolicy", "rule+vlm": "Rule+VLM recovery", "llm": "LLMPolicy"}
    all_policies = list(policies)
    results: list[TaskResult] = []

    print(f"\n{'='*60}")
    print(f"  FULL EVALUATION: {' vs '.join(p.upper() for p in all_policies)}")
    print(f"  Scenes: {', '.join(scenes)}")
    print(f"  Tasks/scene: {tasks_per_scene}")
    print(f"  Started: {started}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")

    for scene in scenes:
        print(f"\n{'─'*60}")
        print(f"  [{scene}] Loading...")
        with ThorController(width=640, height=480, visibility_distance=10.0,
                            render_depth=True) as ctrl:
            ctrl.load_scene(scene, seed=42)
            depth = HeuristicDepth()
            prior = ScenePrior(scene)
            detector_std = create_hybrid(finetuned_model=yolo_model, finetuned_conf=0.3,
                                         world_model="yolov8s-worldv2.pt", world_conf=0.15,
                                         controller=ctrl)
            pipeline_std = PerceptionPipeline(detector_std, depth, prior=prior)
            detector_tour = create_hybrid(finetuned_model=yolo_model, finetuned_conf=0.2,
                                          world_model="yolov8s-worldv2.pt", world_conf=0.10,
                                          controller=ctrl)
            pipeline_tour = PerceptionPipeline(detector_tour, depth, prior=prior)
            mem = _build_spatial_memory(ctrl, pipeline_tour)
            print(f"  Spatial memory (vision): {len(mem)} objects")
            tasks = _full_gen_tasks(ctrl, pipeline_std, tasks_per_scene, spatial_memory=mem)
            nav_c = sum(1 for t in tasks if t.task_type == "navigation")
            int_c = sum(1 for t in tasks if t.task_type == "interaction")
            print(f"  Tasks: {len(tasks)} ({nav_c} nav + {int_c} interact)")

            for pt in all_policies:
                print(f"\n  --- {policy_labels.get(pt, pt)} ---")
                if pt == "llm":
                    model_name = os.environ.get("VLM_MODEL", "qwen-vl-max")
                    policy = LLMPolicy(model=model_name)
                else:
                    policy = RulePolicy(use_vlm_recovery=(pt == "rule+vlm"))
                for i, task in enumerate(tasks):
                    ctrl.load_scene(scene, seed=42)
                    result = _run_one_task(task, ctrl, pipeline_std, policy,
                                           policy_type=pt, max_steps=max_steps,
                                           spatial_memory=mem, traces_dir=traces_dir)
                    results.append(result)
                    status = "✓" if result.success else "✗"
                    print(f"    {i+1:>2}. {status} [{task.task_type[:4]}] {task.text:<40s} "
                          f"{result.steps:>3d}s {result.reason[:50]}")

    finished = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _print_summary(results, scenes, started, finished, all_policies)
    _save_results(results, scenes, started, finished, output_dir, all_policies, mode="full")
    all_traces = []
    for fname in sorted(os.listdir(traces_dir)):
        if fname.endswith(".json"):
            try: all_traces.append(EpisodeTrace.load_json(os.path.join(traces_dir, fname)))
            except Exception: pass
    if all_traces:
        generate_failure_report(all_traces, os.path.join(output_dir, "failure_report.md"))
        print(f"  Failure report → {output_dir}/failure_report.md")
    print(f"  Traces ({len(all_traces)} episodes) → {traces_dir}/")
    print(f"\n  Report saved → {output_dir}/")
    return {"results": results, "output_dir": output_dir, "traces": all_traces}


# ===========================================================================
# Summary + output (shared)
# ===========================================================================

def _print_summary(results, scenes, started, finished, policies):
    all_policies = list(policies)

    def _stats(rs, filt=None):
        if filt: rs = [r for r in rs if r.task_type == filt]
        if not rs: return 0, 0, 0, 0, 0.0
        ok = sum(1 for r in rs if r.success)
        avg_s = sum(r.steps for r in rs) / len(rs)
        avg_t = sum(r.elapsed for r in rs) / len(rs)
        avg_spl = sum(r.spl for r in rs) / len(rs)
        return ok, len(rs), avg_s, avg_t, avg_spl

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Scenes: {', '.join(scenes)}")
    print(f"  Started:  {started}  |  Finished: {finished}")
    print(f"\n  {'Policy':<14s} {'All':>8s} {'Nav':>8s} {'Int':>7s} "
          f"{'SPL':>7s} {'Steps':>7s} {'Time':>7s}")
    print(f"  {'─'*14} {'─'*8} {'─'*8} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")
    labels = {"rule": "Rule", "rule+vlm": "Rule+VLM", "llm": "LLM(VLM)"}
    for pt in all_policies:
        rs = [r for r in results if r.policy == pt]
        if not rs: continue
        a_ok, a_tot, a_s, a_t, a_spl = _stats(rs)
        n_ok, n_tot, _, _, _ = _stats(rs, "navigation")
        i_ok, i_tot, _, _, _ = _stats(rs, "interaction")
        print(f"  {labels.get(pt, pt):<14s} "
              f"{a_ok/a_tot*100 if a_tot else 0:>7.1f}% "
              f"{n_ok/n_tot*100 if n_tot else 0:>7.1f}% "
              f"{i_ok/i_tot*100 if i_tot else 0:>6.1f}% "
              f"{a_spl:>6.3f} {a_s:>6.1f}  {a_t:>5.1f}s")
    print(f"\n  Per-scene:")
    for scene in sorted(set(r.scene for r in results)):
        for pt in all_policies:
            srs = [r for r in results if r.scene == scene and r.policy == pt]
            if not srs: continue
            ok = sum(1 for r in srs if r.success)
            print(f"    {scene:<16s} {labels.get(pt, pt):<10s} {ok}/{len(srs)} ({ok/len(srs)*100:.0f}%)")


def _print_quick_summary(results, scenes, started, finished):
    """Simplified summary for quick mode (single policy, present/absent breakdown)."""
    by_scene = {}
    for r in results: by_scene.setdefault(r.scene, []).append(r)
    print(f"\n  {'='*60}")
    print(f"  RESULTS")
    print(f"  {'='*60}")
    print(f"\n  {'Scene':<16s} {'Present':>9s} {'Absent':>9s} {'Overall':>9s} "
          f"{'Steps':>7s} {'Time':>7s}")
    print(f"  {'─'*16} {'─'*9} {'─'*9} {'─'*9} {'─'*7} {'─'*7}")
    for scene, rs in sorted(by_scene.items()):
        present = [r for r in rs if r.in_scene]
        absent = [r for r in rs if not r.in_scene]
        p_r = sum(r.success for r in present) / len(present) * 100 if present else 0
        a_r = sum(r.success for r in absent) / len(absent) * 100 if absent else 0
        all_r = sum(r.success for r in rs) / len(rs) * 100 if rs else 0
        avg_s = sum(r.steps for r in rs) / len(rs) if rs else 0
        avg_t = sum(r.elapsed for r in rs) / len(rs) if rs else 0
        print(f"  {scene:<16s} {p_r:>8.1f}% {a_r:>8.1f}% {all_r:>8.1f}% {avg_s:>6.1f}  {avg_t:>5.1f}s")


def _save_results(results, scenes, started, finished, output_dir, policies, mode="full"):
    all_policies = list(policies)
    with open(os.path.join(output_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"Evaluation ({mode} mode): {' vs '.join(all_policies)}\n")
        f.write(f"{'='*60}\n")
        f.write(f"Started:  {started}\nFinished: {finished}\n")
        f.write(f"Scenes:   {', '.join(scenes)}\n\n")
        for pt in all_policies:
            rs = [r for r in results if r.policy == pt]
            ok = sum(1 for r in rs if r.success)
            tot = len(rs)
            rate = ok / tot * 100 if tot else 0
            avg_spl = sum(r.spl for r in rs) / len(rs) if rs else 0.0
            f.write(f"[{pt}] {ok}/{tot} ({rate:.1f}%), SPL={avg_spl:.3f}\n")
            for scene in sorted(set(r.scene for r in rs)):
                srs = [r for r in rs if r.scene == scene]
                f.write(f"  {scene}: {sum(1 for r in srs if r.success)}/{len(srs)}\n")
            nav_rs = [r for r in rs if r.task_type == "navigation"]
            int_rs = [r for r in rs if r.task_type == "interaction"]
            f.write(f"  Nav: {sum(1 for r in nav_rs if r.success)}/{len(nav_rs)}  "
                    f"Interact: {sum(1 for r in int_rs if r.success)}/{len(int_rs)}\n\n")
    with open(os.path.join(output_dir, "details.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scene", "policy", "task", "target", "task_type",
                     "in_scene", "success", "steps", "elapsed_s",
                     "optimal_steps", "spl", "distance_m", "reason"])
        for r in results:
            w.writerow([r.scene, r.policy, r.task, r.target, r.task_type,
                         r.in_scene, r.success, r.steps, r.elapsed,
                         r.optimal_steps, r.spl, getattr(r, 'distance_m', 0), r.reason])


# ===========================================================================
# CLI
# ===========================================================================

def main():
    p = argparse.ArgumentParser(
        description="Unified evaluation: multi-policy comparison on nav + interaction tasks")
    p.add_argument("--mode", default="full", choices=["quick", "demo", "full"],
                   help="Evaluation mode: quick (nav-only), demo (curated), full (default)")
    p.add_argument("--scenes", type=int, default=5,
                   help="Number of random scenes (quick: 3, demo: 5, full: 5)")
    p.add_argument("--tasks", type=int, default=None,
                   help="Tasks per scene per policy. Default: quick=10, demo=10, full=20")
    p.add_argument("--tasks-per-scene", type=int, default=None,
                   help="Alias for --tasks (demo mode compatibility)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--policies", type=str, default=None,
                   help="Comma-separated: rule, rule+vlm, llm. Default: "
                        "quick=rule, demo=rule, full=rule,rule+vlm")
    p.add_argument("--scene-list", type=str, default=None,
                   help="Comma-separated explicit scene list (overrides --scenes)")
    p.add_argument("--max-steps", type=int, default=80)
    p.add_argument("--model", default="runs/detect/runs/train/weights/best.pt",
                   help="Fine-tuned YOLO model path")
    p.add_argument("--confidence", type=float, default=0.3)
    p.add_argument("--present-ratio", type=float, default=0.7,
                   help="Quick mode: fraction of present-object tasks")
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    mode = args.mode
    seed = args.seed

    # Resolve scene list
    if args.scene_list:
        scenes = [s.strip() for s in args.scene_list.split(",") if s.strip()]
    elif mode == "demo":
        random.seed(seed)
        scenes = random.sample(DEMO_SCENES, min(args.scenes, len(DEMO_SCENES)))
    else:
        random.seed(seed)
        pool = list(SCENE_POOL)
        random.shuffle(pool)
        scenes = sorted(pool[:args.scenes])

    # Resolve tasks
    tasks_per_scene = args.tasks or args.tasks_per_scene
    if tasks_per_scene is None:
        tasks_per_scene = {"quick": 10, "demo": 10, "full": 20}[mode]

    # Resolve policies
    if args.policies:
        policies = tuple(p.strip() for p in args.policies.split(",") if p.strip())
    else:
        policies = {"quick": ("rule",), "demo": ("rule",), "full": ("rule", "rule+vlm")}[mode]

    valid = {"rule", "rule+vlm", "llm"}
    for pt in policies:
        if pt not in valid:
            print(f"ERROR: Unknown policy '{pt}'. Valid: {', '.join(sorted(valid))}")
            return

    # Route to mode
    if mode == "quick":
        _run_quick(scenes, tasks_per_scene, seed,
                   max_steps=args.max_steps, present_ratio=args.present_ratio,
                   yolo_model=args.model, yolo_conf=args.confidence)
    elif mode == "demo":
        _run_demo(scenes, tasks_per_scene, seed, policies=policies,
                  max_steps=args.max_steps, yolo_model=args.model)
    else:
        _run_full(scenes, tasks_per_scene, seed, policies=policies,
                  max_steps=args.max_steps, yolo_model=args.model)


if __name__ == "__main__":
    main()
