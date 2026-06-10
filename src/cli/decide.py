"""
Decision module demo — end-to-end NL instruction → action execution.

Usage::

    # Single-task mode
    python src/cli/decide.py --task "Go to the chair" --policy rule
    python src/cli/decide.py --task "Open the fridge" --policy rule
    python src/cli/decide.py --task "Pick up the mug" --policy llm

    # With fine-tuned model
    python src/cli/decide.py --task "Go to the chair" \\
        --model runs/detect/runs/train/weights/best.pt --policy rule

    # Interactive mode (no --task)
    python src/cli/decide.py --scene FloorPlan1
    > run Go to the chair
    > run Open the fridge
    > rule / llm / hybrid   (switch policy)
    > info / detect / list
    > quit

Commands (interactive mode)::

    run <instruction>    Execute a task
    rule                 Switch to RulePolicy
    llm                  Switch to LLMPolicy
    info                 Show agent state
    detect               Run perception on current view
    list                 List scene objects
    quit                 Exit
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.controller.thor import ThorController
from src.perception.detector import YOLODetector, create_detector, create_hybrid
from src.perception.depth import HeuristicDepth
from src.perception.pipeline import PerceptionPipeline
from src.recording.collector import FrameCollector
from src.common.logger import setup_logger
from src.common.types import Detection, BBox, Vec3
from src.common.utils import draw_annotations, img_to_b64
from src.perception.spatial_memory import SpatialMemory
from src.decision.types import TaskSpec, ActionDecision
from src.decision.parser import TaskParser
from src.decision.rule_policy import RulePolicy
from src.decision.llm_policy import LLMPolicy

logger = setup_logger("decide")

# Auto-load .env (idempotent — won't override already-set env vars)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ======================================================================
# Frame annotation helpers
# ======================================================================



def _save_annotated_frame(rgb: "np.ndarray", detections: "list[Detection]",
                          step: int, frames_dir: str,
                          collector: "FrameCollector | None" = None) -> str | None:
    """Save an annotated frame as PNG.  Returns the path or None."""
    os.makedirs(frames_dir, exist_ok=True)
    annotated = draw_annotations(rgb, detections)
    path = os.path.join(frames_dir, f"step_{step:04d}.png")
    cv2.imwrite(path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    return path


# ======================================================================
# LLM Advisor — called only when Rule gets stuck (A* failed, target lost)
# ======================================================================

_ADVISOR_PROMPT = """\
You are a navigation advisor for a robot in a 3D indoor room. You can SEE the
robot's current camera view (an image) and read a text summary of YOLO detections.

IMPORTANT: YOLO detection labels are UNRELIABLE — they often misidentify objects,
especially at close range (e.g. "microwave" → "oven" or "cabinet"). Use the IMAGE
as your primary source of truth, not the detection text. If you see the target
in the image but the text shows a different label, trust your eyes.

The robot is STUCK trying to reach a target. Suggest 1-3 actions to escape.

Available actions:
  MOVE_FORWARD / MOVE_BACK — step 0.25 m
  TURN_LEFT / TURN_RIGHT — rotate 90°
  TURN_LEFT_SMALL / TURN_RIGHT_SMALL — rotate 30°
  LOOK_UP / LOOK_DOWN — tilt camera
  NAVIGATE_TO:<object> — use A* pathfinding to a named nearby object
  GIVE_UP — truly unreachable, stop trying

Respond with ONLY a JSON array, e.g.:
["MOVE_BACK", "TURN_LEFT", "MOVE_FORWARD"]"""


def _build_stuck_report(task, detections, agent_state, failed_a_star):
    """Build a stuck-situation report for the VLM advisor.

    Shows ALL nearby detections (not just target matches) so the VLM can
    spot YOLO misidentifications — e.g. a "cabinet" at the position where
    the microwave should be.
    """
    nearby = [d for d in detections if d.distance_meters > 0
              and d.distance_meters < 3.0]
    nearby_lines = "\n".join(
        f"  - {d.label} ({d.screen_position}, {d.distance_meters:.1f}m, "
        f"conf={d.confidence:.2f})"
        for d in sorted(nearby, key=lambda d: d.distance_meters)[:8]
    ) if nearby else "  (nothing detected within 3 m)"

    return (
        f"Task: {task.raw_text}\n"
        f"Target to find: '{task.target}' (type: {task.task_type})\n"
        f"A* pathfinding has failed {failed_a_star} times.\n"
        f"Agent heading: {agent_state.heading_deg:.0f}°  "
        f"Colliding: {agent_state.is_colliding}\n"
        f"\nCurrent YOLO detections (labels MAY BE WRONG):\n"
        f"{nearby_lines}\n"
        f"\nIMAGE 1 = last view where '{task.target}' WAS detected.\n"
        f"IMAGE 2 = current view where it is NOT detected.\n"
        f"\nCompare the two images. YOU decide:\n"
        f"  A) Same object, YOLO changed the label → keep navigating to '{task.target}'\n"
        f"  B) Different object, YOLO is right → the target is gone, try somewhere else\n"
        f"  C) You can see '{task.target}' in IMAGE 2 even though YOLO missed it "
        f"→ guide the robot toward what you see\n"
        f"\nSuggest 1-3 actions. Use NAVIGATE_TO:<object> with whichever label "
        f"YOU think is correct based on the images."
    )




def _ask_llm_advisor(task, detections, agent_state, failed_a_star,
                     rgb: "np.ndarray | None" = None,
                     last_good_rgb: "np.ndarray | None" = None):
    """Call the VLM for stuck-situation advice. Returns list of action strings.

    If *last_good_rgb* is provided, sends it as a "before" image alongside the
    current *rgb* ("after") so the VLM can compare and spot label changes.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "")
    model = os.environ.get("VLM_MODEL", "qwen-vl-max")

    if not api_key:
        print("    [LLM advisor] No API key — skipping")
        return None

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url or None)

        report = _build_stuck_report(task, detections, agent_state,
                                      failed_a_star)
        messages = [
            {"role": "system", "content": _ADVISOR_PROMPT},
        ]

        # Build user content with 1-2 images + text
        user_content = []
        if last_good_rgb is not None and rgb is not None:
            # BEFORE/AFTER: show last frame where target was visible, then current
            user_content.append({
                "type": "text",
                "text": "IMAGE 1 (BEFORE) — last frame where the target WAS visible:",
            })
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_to_b64(last_good_rgb)}"},
            })
            user_content.append({
                "type": "text",
                "text": "IMAGE 2 (NOW) — current view where the target is NOT detected. "
                        "Compare with IMAGE 1: is it the same object with a different label?",
            })
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_to_b64(rgb)}"},
            })
            user_content.append({"type": "text", "text": report})
        elif rgb is not None:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_to_b64(rgb)}"},
            })
            user_content.append({"type": "text", "text": report})
        else:
            messages.append({"role": "user", "content": report})

        if user_content:
            messages.append({"role": "user", "content": user_content})

        response = client.chat.completions.create(
            model=model, messages=messages, max_tokens=256, temperature=0.0)

        text = (response.choices[0].message.content or "").strip()
        if not text:
            print(f"    [LLM advisor] Empty response from API")
            return None

        print(f"    [LLM advisor] Raw response: {text[:200]}")

        # Extract JSON array from response
        import re
        m = re.search(r"\[(.*?)\]", text, re.DOTALL)
        if not m:
            print(f"    [LLM advisor] No JSON array found in response")
            return None

        import json
        actions = json.loads(f"[{m.group(1)}]")
        if isinstance(actions, list) and all(isinstance(a, str) for a in actions):
            return actions
        print(f"    [LLM advisor] Response is not a string list: {actions}")
        return None

    except Exception as e:
        print(f"    [LLM advisor] Exception: {e}")
        logger.warning("LLM advisor call failed: %s", e)
        return None


# ======================================================================
# VLM Exploration — VLM takes direct control to re-acquire lost target
# ======================================================================

_VLM_EXPLORE_PROMPT = """\
You are controlling a robot in a 3D indoor room. You have LOST sight of a target
object you were tracking. Your job: explore to find it again.

You will receive 1-2 images:
  IMAGE 1 (BEFORE) = last view where the target WAS visible
  IMAGE 2 (NOW) = current view

Available actions:
  MOVE_FORWARD / MOVE_BACK — step 0.25 m
  TURN_LEFT / TURN_RIGHT — rotate 90°
  TURN_LEFT_SMALL / TURN_RIGHT_SMALL — rotate 30°
  LOOK_UP / LOOK_DOWN — tilt camera
  STOP — you can see the target again, task done
  GIVE_UP — cannot find it

Tips:
- If IMAGE 1 shows the target nearby, you probably just need a small turn or tilt to see it again.
- YOLO labels are unreliable — trust what you SEE in the image, not the text.
- If the target was close (IMAGE 1), don't walk far — look around from where you are.
- Try LOOK_UP if the target was on a counter/shelf.

Respond with ONLY a JSON object:
{"action": "<action>", "reason": "..."}"""


def _run_vlm_exploration(ctrl, pipeline, task, collector,
                          last_good_rgb, max_vlm_steps=8,
                          verbose=True, save_frames=False, frames_dir="") -> bool:
    """VLM-guided exploration to re-acquire a lost target.

    Sends images to Qwen-VL each step, VLM picks actions directly.
    Returns True if target is re-acquired, False if VLM gives up.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "")
    model = os.environ.get("VLM_MODEL", "qwen-vl-max")

    if not api_key:
        return False

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url or None)
    except Exception as e:
        print(f"    [VLM explore] Cannot init client: {e}")
        return False

    for v_step in range(max_vlm_steps):
        view = ctrl.get_current_view()
        rgb = view.sensor_data.rgb
        dets = pipeline.process(rgb, controller=ctrl)

        # Check if target is back
        target_lower = task.target.lower()
        for d in dets:
            if d.label.lower() == target_lower or target_lower in d.label.lower():
                if verbose:
                    print(f"    👁 VLM step {v_step}: target '{task.target}' "
                          f"re-acquired at {d.distance_meters:.1f}m")
                return True

        # Build nearby detection summary
        nearby = [d for d in dets if d.distance_meters > 0 and d.distance_meters < 3.0]
        nearby_text = "\n".join(
            f"  {d.label} ({d.screen_position}, {d.distance_meters:.1f}m)"
            for d in sorted(nearby, key=lambda d: d.distance_meters)[:6]
        ) if nearby else "  (nothing nearby)"

        user_text = (
            f"Target to find: '{task.target}'\n"
            f"Agent heading: {view.agent_state.heading_deg:.0f}°  "
            f"Colliding: {view.agent_state.is_colliding}\n"
            f"\nNearby detections (labels may be wrong):\n{nearby_text}\n"
            f"\nIMAGE 1 = last view where target WAS visible.\n"
            f"IMAGE 2 = current view. Choose next action."
        )

        # Build message with before/after images
        user_content = [
            {"type": "text", "text": "IMAGE 1 (BEFORE — target was visible here):"},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{img_to_b64(last_good_rgb)}"}},
            {"type": "text", "text": "IMAGE 2 (NOW — current view):"},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{img_to_b64(rgb)}"}},
            {"type": "text", "text": user_text},
        ]

        messages = [
            {"role": "system", "content": _VLM_EXPLORE_PROMPT},
            {"role": "user", "content": user_content},
        ]

        try:
            response = client.chat.completions.create(
                model=model, messages=messages, max_tokens=128, temperature=0.0)
            text = (response.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"    [VLM explore] API error: {e}")
            return False

        if not text:
            continue

        # Parse response
        import re, json
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            continue
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue

        action = str(data.get("action", "TURN_LEFT"))
        reason = str(data.get("reason", ""))[:80]

        if action == "GIVE_UP":
            if verbose:
                print(f"    ❌ VLM gives up: {reason}")
            return False
        if action == "STOP":
            if verbose:
                print(f"    ✅ VLM declares target found: {reason}")
            return True

        if verbose:
            print(f"    [{v_step}] {action:<22s} {reason}")

        if action.startswith("NAVIGATE_TO:"):
            nav_target = action.split(":", 1)[1]
            for _ in ctrl.navigate_to_object(nav_target):
                pass
        elif action != "STOP":
            ctrl.step(action)

        # Save frame
        if save_frames and frames_dir:
            _save_annotated_frame(rgb, dets, v_step, frames_dir, collector)

    return False


# ======================================================================
# VLM Detection Verification — correct YOLO labels every frame
# ======================================================================

_VERIFY_PROMPT = """\
You are a visual inspector for a robot. You see the robot's camera image and a
list of YOLO detections. YOLO is fast but often wrong about labels.

For each YOLO detection, decide if the label is correct. If you think it's wrong,
propose a correction WITH a confidence score (0.0-1.0) reflecting how sure you are.

Also flag objects you see that YOLO missed entirely.

Respond with a JSON object:
{
  "corrections": [
    {"old": "cabinet", "new": "microwave", "vlm_conf": 0.90}
  ],
  "additions": [
    {"label": "apple", "position": "left", "dist": "near", "vlm_conf": 0.80}
  ],
  "target_visible": true,
  "target_label": "microwave",
  "target_position": "center",
  "target_distance": 1.2,
  "target_vlm_conf": 0.95
}

Rules:
- vlm_conf MUST be a number 0.0-1.0. High confidence (>0.8) = you're very sure.
  Low confidence (<0.5) = you're guessing. Be honest.
- Only include corrections where you are REASONABLY confident (vlm_conf >= 0.5).
- If YOLO's label looks correct, don't include it in corrections.
- additions: objects YOLO missed. Use "near"/"mid"/"far" for dist.
- target_visible/target_label/target_position/target_distance: your assessment of
  whether the TARGET object is visible, regardless of YOLO's label."""


def _vlm_verify_detections(rgb, detections, task_target,
                            api_key, base_url, model):
    """Ask VLM to verify/correct YOLO detections on a single frame.

    Returns (corrected_detections, vlm_target_info) or (None, None) on failure.
    vlm_target_info is a dict with keys: visible, label, position, distance.
    """
    if not api_key:
        return None, None, [], []

    # Build detection summary
    det_lines = []
    for i, d in enumerate(detections[:8]):
        det_lines.append(
            f"  [{i}] {d.label} ({d.screen_position}, {d.distance_meters:.1f}m, "
            f"conf={d.confidence:.2f})"
        )
    det_text = "\n".join(det_lines) if det_lines else "  (no detections)"

    user_text = (
        f"Target: '{task_target}'\n"
        f"YOLO detections:\n{det_text}\n"
        f"\nLook at the image. Are the YOLO labels correct? "
        f"Is '{task_target}' visible? Respond with JSON."
    )

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url or None)

        messages = [
            {"role": "system", "content": _VERIFY_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{img_to_b64(rgb)}"}},
                {"type": "text", "text": user_text},
            ]},
        ]

        response = client.chat.completions.create(
            model=model, messages=messages, max_tokens=256, temperature=0.0)
        text = (response.choices[0].message.content or "").strip()
        if not text:
            return None, None, []

        import re, json
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None, None, []
        data = json.loads(m.group(0))

        # Build a log of YOLO vs VLM for each detection
        verify_log: list[str] = []

        # Apply corrections — only if VLM is MORE confident than YOLO
        corrections = data.get("corrections", [])
        for corr in corrections:
            old_label = corr.get("old", "").lower()
            new_label = corr.get("new", "")
            vlm_conf = float(corr.get("vlm_conf", 0.5))
            for d in detections:
                if d.label.lower() == old_label:
                    if vlm_conf > d.confidence:
                        verify_log.append(
                            f"  {old_label}: yolo={d.confidence:.2f} vlm={vlm_conf:.2f} "
                            f"→ override '{new_label}' (vlm wins)")
                        d.label = new_label
                        d.confidence = vlm_conf
                    else:
                        verify_log.append(
                            f"  {old_label}: yolo={d.confidence:.2f} vlm={vlm_conf:.2f} "
                            f"→ keep '{old_label}' (yolo wins)")
                    break

        # Apply additions (VLM spotted objects YOLO missed)
        additions = data.get("additions", [])
        for add in additions:
            label = add.get("label", "")
            pos = add.get("position", "center")
            dist_str = add.get("dist", "mid")
            vlm_conf = float(add.get("vlm_conf", 0.5))
            dist_map = {"near": 0.75, "mid": 2.25, "far": 5.0}
            detections.append(Detection(
                label=label,
                bbox=BBox(x1=0, y1=0, x2=10, y2=10),
                confidence=vlm_conf,
                screen_position=pos,
                distance_level=dist_str.upper(),
                distance_meters=dist_map.get(dist_str, 2.0),
            ))

        # Log uncorrected detections too
        corrected_labels = {c.get("old", "").lower() for c in corrections}
        for d in detections[:6]:
            if d.label.lower() not in corrected_labels:
                verify_log.append(
                    f"  {d.label}: yolo={d.confidence:.2f}  (VLM no objection)")

        # Extract VLM's target assessment
        target_info = {
            "visible": data.get("target_visible", False),
            "label": data.get("target_label", ""),
            "position": data.get("target_position", ""),
            "distance": data.get("target_distance", 0.0),
        }
        target_vlm_conf = data.get("target_vlm_conf", 0.0)
        if target_info["visible"]:
            verify_log.append(
                f"  🎯 TARGET '{task_target}': VLM sees '{target_info['label']}' "
                f"({target_info['position']}, {target_info['distance']:.1f}m, "
                f"vlm_conf={target_vlm_conf:.2f})")

        return detections, target_info, verify_log

    except Exception as e:
        logger.warning("VLM verify failed: %s", e)
        return None, None, []


# ======================================================================
# Episode runner
# ======================================================================

def run_episode(
    task_text: str,
    policy,
    pipeline: PerceptionPipeline,
    ctrl: ThorController,
    collector: FrameCollector | None = None,
    *,
    max_steps: int = 200,
    verbose: bool = True,
    save_frames: bool = True,
    frames_dir: str = "",
    use_llm_advisor: bool = False,
    spatial_memory: SpatialMemory | None = None,
) -> tuple[bool, int, str]:
    """Run a single task episode.

    Returns ``(success, steps_taken, reason)``.
    """
    # Parse instruction
    try:
        task_spec = TaskParser.interpret(task_text)
    except ValueError as e:
        return False, 0, str(e)

    if verbose:
        extra = f"  Interact: {task_spec.interact_action}" if task_spec.is_interaction else ""
        print(f"\n{'='*60}")
        print(f"  Task:   {task_spec.raw_text}")
        print(f"  Type:   {task_spec.task_type}  |  Target: {task_spec.target}" + extra)
        print(f"  Policy: {policy.__class__.__name__}")
        print(f"{'='*60}")

    # Reset policy state
    if hasattr(policy, 'reset'):
        policy.reset()

    start_time = time.time()
    failed_a_star = 0      # track consecutive A* failures to detect impossible targets
    llm_advisor_used = False  # only call LLM advisor once per episode
    explore_count = 0        # try different exploration spots
    last_target_view: "np.ndarray | None" = None  # last frame where target was visible

    was_near_before_lost = False  # agent was near target before it disappeared

    for step in range(max_steps):
        # --- 1. Observe ---
        view = ctrl.get_current_view()
        if collector is not None:
            collector.record(view)

        # --- 2. Perceive ---
        detections = pipeline.process(view.sensor_data.rgb, controller=ctrl)

        # --- 2b. VLM verification ---
        vlm_target = None
        if use_llm_advisor:
            api_key = os.environ.get("OPENAI_API_KEY", "")
            base_url = os.environ.get("OPENAI_BASE_URL", "")
            model = os.environ.get("VLM_MODEL", "qwen-vl-max")
            corrected, vlm_target, vlog = _vlm_verify_detections(
                view.sensor_data.rgb, detections, task_spec.target,
                api_key, base_url, model)
            if corrected is not None:
                detections = corrected
                if verbose and step == 0 and vlog:
                    print(f"  VLM vs YOLO:")
                    for line in vlog:
                        print(line)

        # Coverage stats on first step
        if step == 0 and verbose:
            summary = pipeline.coverage_summary(detections, ctrl)
            if summary:
                print(f"  {summary}")

        # Save annotated frame
        if save_frames and frames_dir:
            _save_annotated_frame(view.sensor_data.rgb, detections,
                                  step, frames_dir, collector)

        # --- 3. Decide ---
        decision = policy.decide(
            rgb=view.sensor_data.rgb,
            detections=detections,
            agent_state=view.agent_state,
            task=task_spec,
            step_count=step,
            depth_frame=view.sensor_data.depth,
            spatial_memory=spatial_memory,
        )

        # --- 4. Print ---
        if verbose:
            target_det = _find_best_det(task_spec.target, detections)
            vis = f"vis=yes({target_det.distance_meters:.1f}m)" if target_det else "vis=no"
            print(f"  [{step:03d}] {decision.action:<22s} {vis:<16s} {decision.reason}")
            # Remember last frame where target was visible (for LLM before/after comparison)
            if target_det is not None:
                last_target_view = view.sensor_data.rgb.copy()
                if target_det.distance_meters <= 1.5:
                    was_near_before_lost = True

        # --- 5a. VLM Exploration — once per episode, then give up ---
        if decision.action == "VLM_EXPLORE":
            if llm_advisor_used:
                return False, step + 1, \
                    f"VLM already tried for '{task_spec.target}' — giving up."
            if use_llm_advisor:
                llm_advisor_used = True
                if verbose:
                    print(f"  🧠 VLM exploring for '{task_spec.target}'...")
                vlm_ok = _run_vlm_exploration(
                    ctrl, pipeline, task_spec, collector,
                    last_target_view, max_vlm_steps=8,
                    verbose=verbose, save_frames=save_frames,
                    frames_dir=frames_dir)
                if vlm_ok:
                    failed_a_star = 0
                    was_near_before_lost = False
                    continue
            return False, step + 1, \
                f"VLM recovery failed for '{task_spec.target}'"

        # --- 5b. A* navigation — vision-based or metadata-based ---
        if decision.action.startswith("NAVIGATE_TO:"):
            parts = decision.action.split(":")
            if len(parts) >= 4:
                # Vision-based: NAVIGATE_TO:microwave:1.52:2.14
                # Navigate to the remembered XZ position (y=0 for floor projection)
                obj_name = parts[1]
                tx, tz = float(parts[2]), float(parts[3])
                if verbose:
                    print(f"    A* (vision): navigating to ({tx:.1f}, {tz:.1f})")
                nav_ok = False
                for _ in ctrl.navigate_to(Vec3(x=tx, y=0, z=tz)):
                    nav_ok = True
                    if collector is not None:
                        collector.record(_)
                # After reaching, look around for the target
                if nav_ok:
                    for _ in range(2):
                        ctrl.step("TURN_LEFT")
                        view = ctrl.get_current_view()
                        if collector is not None:
                            collector.record(view)
            else:
                # Metadata-based: NAVIGATE_TO:microwave (fallback)
                obj_name = parts[1]
                nav_ok = _run_navigation(ctrl, obj_name, collector,
                                          verbose=verbose,
                                          save_frames=save_frames,
                                          frames_dir=frames_dir,
                                          pipeline=pipeline)
            if not nav_ok:
                failed_a_star += 1
                if verbose:
                    print(f"  ⚠ A* could not reach '{obj_name}' "
                          f"(fail {failed_a_star}/3)")

                # --- Multi-angle approach: try to reach the target from
                # different directions.  A single approach point may be
                # blocked (e.g. counter), but left/right/opposite sides
                # may have open access.
                if failed_a_star <= 2:
                    try:
                        obj_map = ctrl.get_object_map()
                        info = obj_map.get(obj_name)
                        if info is None:
                            for k, v in obj_map.items():
                                if k.lower() == obj_name.lower():
                                    info = v
                                    break
                        if info:
                            obj_pos = info["position"]  # Vec3
                            ax = ctrl.agent_state.position.x
                            az = ctrl.agent_state.position.z
                            dx = ax - obj_pos.x
                            dz = az - obj_pos.z
                            # Multi-angle: 0° (front), ±90° (sides), 180° (back)
                            offsets = [0, 1.57, -1.57, 3.14]
                            idx = (failed_a_star - 1) % len(offsets)
                            angle = math.atan2(dx, dz) + offsets[idx]
                            ax_x = obj_pos.x + math.sin(angle) * 1.2
                            ax_z = obj_pos.z + math.cos(angle) * 1.2
                            if verbose:
                                print(f"    🔍 Angle #{failed_a_star}: "
                                      f"({ax_x:.1f}, {ax_z:.1f})")
                            for _ in ctrl.navigate_to(
                                Vec3(x=ax_x, y=obj_pos.y, z=ax_z)):
                                pass
                            failed_a_star = 0
                            continue
                    except Exception:
                        pass

                # --- LLM Advisor: called at most once per episode when stuck ---
                if use_llm_advisor and failed_a_star >= 2 and not llm_advisor_used:
                    llm_advisor_used = True
                    if verbose:
                        print(f"  🤖 Asking LLM advisor (with before/after)...")
                    advice = _ask_llm_advisor(
                        task_spec, detections, view.agent_state, failed_a_star,
                        rgb=view.sensor_data.rgb,
                        last_good_rgb=last_target_view)
                    if verbose:
                        print(f"  🤖 LLM advisor returned: {advice}")
                    if advice:
                        if verbose:
                            print(f"  💡 LLM advisor suggests: {advice}")
                        for act in advice:
                            if act == "GIVE_UP":
                                break
                            if act.startswith("NAVIGATE_TO:"):
                                nav_target = act.split(":", 1)[1]
                                for _ in ctrl.navigate_to_object(nav_target):
                                    pass
                            elif act != "STOP":
                                result = ctrl.step(act)
                                if collector is not None:
                                    collector.record(result)
                        failed_a_star = 0  # reset — advisor gave us a new plan
                        continue  # re-observe

                # --- LLM fallback: when A* fails, hand to VLM for visual exploration ---
                if use_llm_advisor and failed_a_star >= 2 and not llm_advisor_used:
                    llm_advisor_used = True
                    if verbose:
                        print(f"  🧠 A* failed {failed_a_star}x — handing to VLM explorer...")
                    vlm_ok = _run_vlm_exploration(
                        ctrl, pipeline, task_spec, collector,
                        last_target_view, max_vlm_steps=8,
                        verbose=verbose, save_frames=save_frames,
                        frames_dir=frames_dir)
                    if vlm_ok:
                        failed_a_star = 0
                        continue
                    else:
                        return False, step + 1, \
                            f"A* + VLM both failed for '{task_spec.target}'"

                # If LLM already tried and failed, give up
                give_up_after = 2 if llm_advisor_used else 3
                if failed_a_star >= give_up_after:
                    if verbose:
                        print(f"  ✗ A* failed {failed_a_star} times — "
                              f"'{obj_name}' may not exist in this scene")
                    return False, step + 1, \
                        f"'{obj_name}' not found (A* failed {failed_a_star}x)"
            else:
                failed_a_star = 0
            continue  # re-observe after navigation

        # --- 6. Execute action (including terminal actions like INTERACT_*) ---
        if decision.action == "STOP":
            pass  # pure stop, no controller action needed
        else:
            if decision.action.startswith("INTERACT_"):
                ctrl._interact_target = task_spec.target
            result = ctrl.step(decision.action)
            if collector is not None:
                collector.record(result)
            if result.agent_state.is_colliding and verbose:
                print(f"    ⚠ blocked")

        # Capture post-action frame for video (especially important for INTERACT_*)
        if save_frames and frames_dir:
            view = ctrl.get_current_view()
            dets = pipeline.process(view.sensor_data.rgb, controller=ctrl)
            _save_annotated_frame(view.sensor_data.rgb, dets,
                                  step + 1, frames_dir, collector)

        # --- 7. Terminal? ---
        if decision.done:
            elapsed = time.time() - start_time
            if verbose:
                print(f"  ✓ SUCCESS  ({step + 1} steps, {elapsed:.1f} s): {decision.reason}")
            return True, step + 1, decision.reason

        if decision.action == "STOP":
            elapsed = time.time() - start_time
            success, reason = _evaluate_stop(ctrl, task_spec)
            if verbose:
                tag = "✓ SUCCESS" if success else "✗ FAIL"
                print(f"  {tag}  ({step + 1} steps, {elapsed:.1f} s): {reason}")
            return success, step + 1, reason

    # Out of steps
    elapsed = time.time() - start_time
    success, reason = _evaluate_stop(ctrl, task_spec)
    reason = f"{reason} (max {max_steps} steps, {elapsed:.1f} s)"
    if verbose:
        tag = "✓ SUCCESS" if success else "✗ FAIL"
        print(f"  {tag}: {reason}")
    return success, max_steps, reason


# ======================================================================
# Room tour — walk to multiple spots and scan for full coverage
# ======================================================================

def _scan360(ctrl, pipeline) -> set[str]:
    """Rotate 360° in 90° steps, return union of all detected labels."""
    labels: set[str] = set()
    for i in range(4):
        view = ctrl.get_current_view()
        dets = pipeline.process(view.sensor_data.rgb, controller=ctrl)
        for d in dets:
            labels.add(d.label.lower())
        if i < 3:
            ctrl.step("TURN_LEFT")
    return labels


def _room_tour(ctrl, pipeline, *, num_stops: int = 3, verbose: bool = True
               ) -> tuple[set[str], bool, str]:
    """Walk to *num_stops* positions spread across the room, 360° scan at each.

    Returns ``(all_labels, target_seen, target_label)``.
    """
    from src.common.types import Vec3

    positions = ctrl._get_reachable_positions_raw()
    if len(positions) < num_stops + 1:
        # Tiny room — just scan from spawn
        return _scan360(ctrl, pipeline), False, ""

    # Pick positions spread across the room: farthest points from centroid
    cx = sum(p.x for p in positions) / len(positions)
    cz = sum(p.z for p in positions) / len(positions)
    # Sort by distance from centroid, pick evenly spaced
    ranked = sorted(positions, key=lambda p: -(p.x - cx)**2 - (p.z - cz)**2)

    # Pick num_stops from the far extremes (skip the first if it's spawn)
    stops = []
    for p in ranked:
        if len(stops) >= num_stops:
            break
        # Don't pick positions too close to each other
        too_close = False
        for sp in stops:
            if ((p.x - sp.x)**2 + (p.z - sp.z)**2) < 1.0:
                too_close = True
                break
        if not too_close:
            stops.append(p)

    if verbose:
        stop_str = ", ".join(f"({s.x:.1f},{s.z:.1f})" for s in stops)
        print(f"  Room tour: {len(stops)} stops → {stop_str}")

    all_labels: set[str] = set()
    for i, pos in enumerate(stops):
        # Walk to position via A* (silent)
        for _ in ctrl.navigate_to(pos):
            pass
        # Scan
        labels = _scan360(ctrl, pipeline)
        all_labels |= labels
        if verbose:
            print(f"    stop {i+1}/{len(stops)}: +{len(labels)} labels → "
                  f"running total {len(all_labels)}")

    # Walk back toward center (first stop was farthest, last is nearest to center)
    if stops:
        for _ in ctrl.navigate_to(stops[0]):
            pass

    # Check target visibility
    target_seen = False
    target_label = ""
    tl = ""
    return all_labels, target_seen, target_label


def _prescan_and_report(ctrl, pipeline, target: str, verbose: bool = True
                         ) -> SpatialMemory:
    """Vision-based room tour: walk + YOLO + depth → spatial memory.

    Walks to 3 spread-out positions, does a 360° scan at each, and
    computes 3D world positions for every YOLO detection using the
    depth frame + camera pose.  No AI2-THOR metadata used.
    """
    mem = SpatialMemory()
    tl = target.lower()

    # Room tour: walk to 3 spots, 360° scan at each
    labels_seen, _, _ = _room_tour(ctrl, pipeline, num_stops=3, verbose=verbose)

    # Ingest each scan's detections with depth → 3D world positions
    # (re-scan because _room_tour only collects labels, not positions)
    for _ in range(4):
        ctrl.step("TURN_LEFT")
        view = ctrl.get_current_view()
        dets = pipeline.process(view.sensor_data.rgb, controller=ctrl)
        s = view.agent_state
        mem.ingest_scan(
            dets, view.sensor_data.depth,
            s.position.x, s.position.y, s.position.z,
            s.heading_deg, s.horizon_deg)

    # Coverage stats
    obj_map = ctrl.get_object_map()
    total = len(obj_map)
    target_in_mem = mem.lookup(tl) is not None

    if verbose:
        print(f"  Spatial memory (vision): {len(mem)} objects indexed"
              + (f", target '{target}' {'✓' if target_in_mem else '✗ NOT found'}")
              )

    return mem


# ======================================================================
# Helpers
# ======================================================================

def _find_best_det(target: str, detections: list[Detection]) -> Detection | None:
    """Find best detection matching *target* (for display only)."""
    t = target.lower()
    candidates = [d for d in detections
                  if d.label.lower() == t
                  or t in d.label.lower()
                  or d.label.lower() in t]
    return min(candidates, key=lambda d: d.distance_meters) if candidates else None


def _run_navigation(ctrl: ThorController, obj_name: str,
                    collector: FrameCollector | None = None,
                    verbose: bool = True,
                    save_frames: bool = False,
                    frames_dir: str = "",
                    pipeline: PerceptionPipeline | None = None) -> bool:
    """Run A* navigation to *obj_name*.  Returns True if any steps executed."""
    nav_steps = 0
    try:
        for result in ctrl.navigate_to_object(obj_name):
            nav_steps += 1
            if collector is not None:
                collector.record(result)
            if save_frames and frames_dir and pipeline is not None:
                dets = pipeline.process(result.sensor_data.rgb, controller=ctrl)
                _save_annotated_frame(result.sensor_data.rgb, dets,
                                      nav_steps, frames_dir, collector)
            if verbose and nav_steps % 4 == 0:
                p = result.agent_state.position
                print(f"    nav[{nav_steps:03d}] ({p.x:5.2f},{p.z:5.2f})")
        if nav_steps > 0:
            d = ctrl.distance_to(obj_name)
            if verbose:
                info = f"{d:.2f} m" if d is not None else "?"
                print(f"    A* complete: {nav_steps} steps, final distance: {info}")
        return nav_steps > 0
    except Exception as e:
        if verbose:
            print(f"    Navigation error: {e}")
        return False


def _evaluate_stop(ctrl: ThorController, task: TaskSpec) -> tuple[bool, str]:
    """After STOP, check if the task was actually accomplished."""
    d = ctrl.distance_to(task.target)
    if task.is_navigation:
        if d is not None and d <= 1.0:
            return True, f"Arrived within {d:.2f} m of '{task.target}'"
        elif d is not None:
            return False, f"Stopped {d:.2f} m from '{task.target}'"
        else:
            return False, f"'{task.target}' not found in scene"
    else:
        # Interaction: can't easily verify success; trust the policy
        return True, f"Interaction with '{task.target}' executed"


# ======================================================================
# Interactive loop
# ======================================================================

def _interactive_loop(ctrl: ThorController, pipeline: PerceptionPipeline,
                      collector: FrameCollector | None, args,
                      save_frames: bool = True, frames_dir: str = "",
                      spatial_memory: SpatialMemory | None = None):
    """Interactive command loop (when --task is not specified)."""
    policy = RulePolicy()
    policy_name = "rule"
    use_llm_advisor = False
    if getattr(args, 'policy', 'rule') == 'llm':
        policy = RulePolicy(use_vlm_recovery=True)
        policy_name = "rule+vlm"
        use_llm_advisor = True
    use_clip = getattr(args, 'clip', False)
    use_prior = getattr(args, 'no_prior', False) is False

    print(f"\n  Scene: {ctrl.scene_name}  |  Policy: {policy_name}")
    print(f"  Commands: run <task>  |  rule / llm  |  info / detect / list  |  quit\n")

    while True:
        try:
            cmd = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Exiting.")
            break
        if not cmd:
            continue

        head, _, tail = cmd.partition(" ")
        head_l = head.lower()

        # --- run <instruction> ---
        if head_l == "run" and tail:
            run_episode(
                tail, policy, pipeline, ctrl, collector,
                max_steps=args.max_steps,
                save_frames=save_frames,
                frames_dir=frames_dir,
                use_llm_advisor=use_llm_advisor,
                spatial_memory=spatial_memory,
            )
            continue

        # --- switch policy ---
        if head_l == "rule":
            policy = RulePolicy()
            policy_name = "rule"
            use_llm_advisor = False
            print(f"  Policy → rule (deterministic, no API)")
            continue

        if head_l == "llm":
            policy = RulePolicy(use_vlm_recovery=True)
            policy_name = "rule+vlm"
            use_llm_advisor = True
            print(f"  Policy → rule+vlm (Rule control + VLM visual recovery)")
            continue

        # --- info ---
        if head_l == "info":
            s = ctrl.agent_state
            p = s.position
            print(f"  Scene: {ctrl.scene_name}  Steps: {ctrl.step_count}")
            print(f"  Pos: ({p.x:.2f}, {p.y:.2f}, {p.z:.2f})  "
                  f"Head: {s.heading_deg:.0f} deg  Colliding: {s.is_colliding}")
            print(f"  Policy: {policy_name}  Max steps: {args.max_steps}")
            continue

        # --- detect ---
        if head_l == "detect":
            view = ctrl.get_current_view()
            dets = pipeline.process(view.sensor_data.rgb, controller=ctrl)
            _print_detections(dets)
            if save_frames and frames_dir:
                _save_annotated_frame(view.sensor_data.rgb, dets,
                                      ctrl.step_count, frames_dir, collector)
            continue

        # --- list ---
        if head_l == "list":
            obj_map = ctrl.get_object_map()
            if not obj_map:
                print("  (no objects)")
            else:
                print(f"  Scene objects ({len(obj_map)} types):")
                for name, info in sorted(obj_map.items()):
                    p = info["position"]
                    vis = "visible" if info["visible"] else "hidden"
                    print(f"    {name:<25s} ({p.x:5.1f},{p.y:4.1f},{p.z:5.1f})  {vis}")
            continue

        # --- quit ---
        if head_l in ("quit", "exit", "q"):
            break

        print(f"  Unknown: '{cmd}'.  Try: run <task> / rule / llm / hybrid / info / detect / list / quit")


def _print_detections(detections: list[Detection]) -> None:
    """Print detection table (compact version of perceive.py's output)."""
    if not detections:
        print("  (nothing detected)")
        return
    header = f"  {'Label':<18s} {'Conf':>6s}  {'Pos':<8s}  {'Dist':>8s}"
    print(header)
    print(f"  {'-'*18} {'-'*6}  {'-'*8}  {'-'*8}")
    for d in detections[:15]:
        dist_str = f"{d.distance_meters:.2f} m" if d.distance_meters else d.distance_level
        print(f"  {d.label:<18s} {d.confidence:5.2f}   {d.screen_position:<8s}  {dist_str:>8s}")


# ======================================================================
# CLI entry
# ======================================================================

def main():
    p = argparse.ArgumentParser(
        description="Decision module demo — NL instruction → action execution")
    p.add_argument("--scene", default="FloorPlan1")
    p.add_argument("--task", default=None,
                   help="NL instruction (omit for interactive mode)")
    p.add_argument("--policy", default="rule", choices=["rule", "llm"],
                   help="Decision policy: rule (no API, fast) | "
                        "llm (Rule+VLM — VLM helps with visual recovery)")
    p.add_argument("--detector", default="hybrid", choices=["hybrid", "finetuned", "yolo-world"],
                   help="Detection mode: hybrid (finetuned+YOLO-World), finetuned, yolo-world")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--model", default="runs/detect/runs/train/weights/best.pt",
                   help="YOLO model path")
    p.add_argument("--confidence", type=float, default=0.3)
    p.add_argument("--vlm-model", default=None,
                   help="VLM model for LLM policy (default: VLM_MODEL env var)")
    p.add_argument("--clip", action="store_true",
                   help="Enable CLIP verification")
    p.add_argument("--no-prior", action="store_true",
                   help="Disable scene prior")
    p.add_argument("--no-record", action="store_true",
                   help="Disable frame recording")
    p.add_argument("--no-export", action="store_true",
                   help="Disable auto video export at end of session")
    p.add_argument("--no-frames", action="store_true",
                   help="Disable saving annotated per-frame screenshots")
    args = p.parse_args()

    # ---- Build perception pipeline ----
    if args.detector == "hybrid":
        detector = None  # Created after scene load (needs controller for YOLO-W vocab)
        logger.info("Detector: Hybrid (finetuned + YOLO-World) — will init after scene load")
    elif args.detector == "yolo-world":
        detector = create_detector(model_name=args.model, confidence=args.confidence,
                                   classes_path="config/classes.yaml")
        logger.info("Detector: YOLO-World with %d classes",
                     len(detector.classes) if hasattr(detector, 'classes') else 0)
    else:
        detector = YOLODetector(model_name=args.model, confidence=args.confidence)
        logger.info("Detector: YOLO-finetuned")
    depth = HeuristicDepth()

    verifier = None
    if args.clip:
        try:
            from src.perception.verifier import CLIPVerifier
            verifier = CLIPVerifier()
        except ImportError as e:
            print(f"  CLIP not available: {e}")

    prior = None
    if not args.no_prior:
        from src.perception.prior import ScenePrior
        prior = ScenePrior(args.scene)

    pipeline = PerceptionPipeline(detector, depth, verifier=verifier, prior=prior)

    # ---- Build policy ----
    # "rule" : RulePolicy — deterministic rules, no API, fast.
    # "llm"  : RulePolicy + VLM recovery — Rule handles control, VLM helps
    #          with label correction and exploration when target is lost.
    #          This hybrid is the BEST performing option (51.4% vs 38.9% pure VLM).
    use_llm_advisor = False
    if args.policy == "llm":
        policy = RulePolicy(use_vlm_recovery=True)
        use_llm_advisor = True
        print(f"  Policy: Rule+VLM (Rule for control + VLM for visual recovery)")
    else:
        policy = RulePolicy()
        print(f"  Policy: RulePolicy (deterministic, no API)")

    collector = None if args.no_record else FrameCollector(scene=args.scene)

    # ---- Run ----
    with ThorController(width=800, height=600, render_depth=True) as ctrl:
        ctrl.load_scene(args.scene)
        view = ctrl.get_current_view()
        if collector is not None:
            collector.record(view)

        # Narrow YOLO-World vocabulary to objects actually in this scene
        if args.detector == "yolo-world" and hasattr(detector, 'set_classes_from_scene'):
            detector.set_classes_from_scene(ctrl)

        # Create hybrid detector after scene load (needs controller for YOLO-W vocab)
        if args.detector == "hybrid":
            world_model = args.vlm_model or "yolov8s-worldv2.pt"
            detector = create_hybrid(
                finetuned_model=args.model,
                finetuned_conf=args.confidence,
                world_model=world_model,
                world_conf=0.15,
                controller=ctrl,
            )
            logger.info("Hybrid detector ready: %s + %s", args.model, world_model)
            pipeline.detector = detector  # wire into the pipeline created earlier with detector=None

        print(f"\n  Session: {collector.session_dir if collector is not None else '(no recording)'}")

        # --- Room tour: build spatial memory once for this scene ---
        spatial_memory = _prescan_and_report(ctrl, pipeline, "", verbose=True)

        objects = len(ctrl.get_object_map())
        tag = f" ({args.detector})" if args.detector != "finetuned" else ""
        print(f"  Scene: {args.scene} ({objects} object types)  |  "
              f"Policy: {args.policy}  |  Model: {args.model}{tag}")

        # Setup annotated frames directory
        save_frames = not args.no_frames
        frames_dir = ""
        if collector is not None and save_frames:
            frames_dir = os.path.join(collector.session_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            # Save initial view with detection annotations
            dets = pipeline.process(view.sensor_data.rgb, controller=ctrl)
            _save_annotated_frame(view.sensor_data.rgb, dets, 0, frames_dir, collector)

        if args.task:
            # ---- Single-episode mode ----
            success, steps, reason = run_episode(
                args.task, policy, pipeline, ctrl, collector,
                max_steps=args.max_steps,
                save_frames=save_frames,
                frames_dir=frames_dir,
                use_llm_advisor=use_llm_advisor,
                spatial_memory=spatial_memory,
            )
            print(f"\n  Result: {'SUCCESS' if success else 'FAIL'}  |  "
                  f"Steps: {steps}  |  {reason}")
        else:
            _interactive_loop(ctrl, pipeline, collector, args,
                              save_frames=save_frames, frames_dir=frames_dir,
                              spatial_memory=spatial_memory)

    # ---- Auto-export (same behaviour as manual.py) ----
    if collector is not None and collector.frame_count > 0:
        if not args.no_export:
            try:
                path = collector.export_video(fps=5)
                print(f"  Trajectory video saved → {path}")
            except Exception as exc:
                print(f"  Video export failed: {exc}")
            try:
                d = collector.export_frames()
                count = len([f for f in os.listdir(d) if f.endswith(".png")])
                print(f"  Annotated frames: {count} → {d}/")
            except Exception as exc:
                print(f"  Frame export failed: {exc}")
        print(f"  Session dir: {collector.session_dir}")
        if not args.no_frames:
            frames_dir = os.path.join(collector.session_dir, "frames")
            if os.path.isdir(frames_dir):
                count = len([f for f in os.listdir(frames_dir)
                           if f.endswith(".png")])
                if count > 0:
                    print(f"  Detection-annotated frames: {count} → {frames_dir}")


if __name__ == "__main__":
    main()
