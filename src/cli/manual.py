"""
Manual control demo with A* pathfinding.

Drive the agent through an AI2-THOR scene from the terminal.
Screenshots are auto-saved to outputs/ after every action.

Usage::

    python src/cli/manual.py [--scene FloorPlan1] [--width 800] [--height 600]

Commands::

    w / s         Move forward / back
    a / d         Turn left / right (90 deg)
    q / e         Fine turn (30 deg)
    r / f         Look up / down
    go  <name>    A*-navigate to an object (e.g. "go Chair")
    look <name>   Turn to face an object
    info          Agent pose & scene
    list          Objects in scene
    where <name>  Distance to object
    desc          Describe current view with VLM
    detect        Run YOLO detection + distance on current view
    shot          Save single screenshot
    video         Export recorded frames as MP4 video
    frames        Export recorded frames as PNG sequence
    auto          Toggle auto-save screenshot on every action
    scene <n>     Switch to FloorPlan<n>
    help          This help
    quit          Exit (auto-exports video on quit)
"""

import argparse
import base64
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.controller.thor import ThorController
from src.recording.collector import FrameCollector
from src.common.logger import setup_logger
from src.common.utils import draw_annotations

logger = setup_logger("manual")

# ---------------------------------------------------------------------------
# Shortcut → action
# ---------------------------------------------------------------------------

SHORTCUT: dict[str, str] = {
    "w": "MOVE_FORWARD",
    "s": "MOVE_BACK",
    "a": "TURN_LEFT",
    "d": "TURN_RIGHT",
    "q": "TURN_LEFT_SMALL",
    "e": "TURN_RIGHT_SMALL",
    "r": "LOOK_UP",
    "f": "LOOK_DOWN",
}

SHORTCUT_DESC: dict[str, str] = {
    "w": "forward", "s": "back",
    "a": "left 90", "d": "right 90",
    "q": "left 30", "e": "right 30",
    "r": "look up", "f": "look down",
}

# Track previous position for delta display
_prev_pos: tuple[float, float, float] | None = None


def _print_state(result, ctrl: ThorController) -> None:
    global _prev_pos
    p = result.agent_state.position
    h = result.agent_state.heading_deg
    ok = result.success
    coll = result.agent_state.is_colliding

    # Compute delta from previous step
    if _prev_pos is not None:
        dx = p.x - _prev_pos[0]
        dy = p.y - _prev_pos[1]
        dz = p.z - _prev_pos[2]
        if abs(dx) > 0.001 or abs(dy) > 0.001 or abs(dz) > 0.001:
            delta = f"d=({dx:+.2f},{dy:+.2f},{dz:+.2f})"
        else:
            delta = "d=(no movement)"
    else:
        delta = ""

    # Status
    if coll:
        status = "BLOCKED (wall/obstacle)"
    elif not ok:
        status = "FAILED"
    else:
        status = "ok"

    print(f"  [{ctrl.step_count:03d}] pos=({p.x:6.2f},{p.y:5.2f},{p.z:6.2f})  "
          f"heading={h:6.1f} deg  [{status}]  {delta}")

    _prev_pos = (p.x, p.y, p.z)


def _describe_view(result) -> str | None:
    """Call VLM (DeepSeek / GPT-4V compatible) to describe the current frame."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "")
    if not api_key:
        print("  No OPENAI_API_KEY set — cannot call VLM.")
        return None

    # Encode frame as base64 JPEG
    img = result.sensor_data.rgb
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    b64 = base64.b64encode(buf).decode()

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url or None)
        response = client.chat.completions.create(
            model=os.environ.get("VLM_MODEL", "deepseek-v4-pro"),
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this indoor scene in one sentence as if you are a robot. Mention the key objects and their rough positions (left/center/right)."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }],
            max_tokens=120,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  VLM error: {e}")
        return None


# Lazy-loaded perception pipeline (loaded on first use)
_pipeline = None

def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        try:
            from src.perception.detector import YOLODetector
        except ImportError as e:
            print(f"  Cannot load perception: {e}")
            return None
        from src.perception.depth import HeuristicDepth
        from src.perception.pipeline import PerceptionPipeline
        _pipeline = PerceptionPipeline(
            YOLODetector(confidence=0.3),
            HeuristicDepth(),
        )
    return _pipeline




def _print_detections(detections, img_w: int) -> None:
    if not detections:
        print("  (nothing detected)")
        return
    print(f"  {'Label':<18s} {'Conf':>6s}  {'Pos':<8s}  {'Dist':>8s}  {'Dist(m)':>8s}  BBox")
    print(f"  {'-'*18} {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*20}")
    for d in detections:
        b = d.bbox
        print(f"  {d.label:<18s} {d.confidence:5.2f}   {d.screen_position:<8s}  "
              f"{d.distance_level:<8s}  {d.distance_meters:>6.2f}   "
              f"({b.x1:4.0f},{b.y1:4.0f})-({b.x2:4.0f},{b.y2:4.0f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(scene: str = "FloorPlan1", width: int = 800, height: int = 600) -> None:
    auto_save = True
    collector = FrameCollector(scene=scene)

    with ThorController(width=width, height=height, render_depth=False) as ctrl:
        # --- load ---
        try:
            ctrl.load_scene(scene)
            result = ctrl.get_current_view()
            collector.record(result)
            if auto_save:
                collector.save_screenshot(result, "init")
            print(f"\n  Session: {collector.session_dir}")
            print(f"  Scene loaded: {scene}")
            _print_state(result, ctrl)
        except Exception as exc:
            print(f"  Failed: {exc}")
            return

        print(f"  Frames recorded: {collector.frame_count}")
        print(f"  Auto-save: {'ON' if auto_save else 'OFF'} | 'help' for commands\n")

        # --- loop ---
        while True:
            try:
                cmd = input("  > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Exiting.")
                break

            if not cmd:
                continue

            parts = cmd.split(maxsplit=1)
            head = parts[0].lower()
            tail = parts[1] if len(parts) > 1 else ""

            # ---- quit ----
            if head in ("quit", "exit", "q!"):
                break

            # ---- help ----
            if head == "help":
                print("""
  ╔══════════════════════════════════════════════╗
  ║  Manual Control                             ║
  ╠══════════════════════════════════════════════╣
  ║  w/s         Move forward / back            ║
  ║  a/d         Turn left / right  (90 deg)    ║
  ║  q/e         Fine turn  (30 deg)            ║
  ║  r/f         Look up / down                 ║
  ║  go  <obj>   A*-navigate to object          ║
  ║  look <obj>  Face an object                 ║
  ║  info        Agent pose & scene info        ║
  ║  list        All objects in scene           ║
  ║  where <obj> Distance to object             ║
  ║  desc        Describe view with VLM         ║
  ║  detect      YOLO detection + distance      ║
  ║  shot        Save single screenshot         ║
  ║  video       Export frames as MP4 video     ║
  ║  frames      Export frames as PNG sequence  ║
  ║  auto        Toggle auto-save               ║
  ║  scene <n>   Switch FloorPlan<n>            ║
  ║  quit        Exit (auto-exports video)      ║
  ╚══════════════════════════════════════════════╝
  Frames are recorded automatically.
""")
                continue

            # ---- info ----
            if head == "info":
                p = result.agent_state.position
                print(f"  Scene: {ctrl.scene_name} | Steps: {ctrl.step_count}")
                print(f"  Position:  ({p.x:.2f}, {p.y:.2f}, {p.z:.2f})")
                print(f"  Heading:   {result.agent_state.heading_deg:.1f} deg")
                print(f"  Horizon:   {result.agent_state.horizon_deg:.1f} deg")
                print(f"  Frames recorded: {collector.frame_count}")
                print(f"  Colliding: {result.agent_state.is_colliding}")
                continue

            # ---- shot ----
            if head == "shot":
                path = collector.save_screenshot(result)
                print(f"  Screenshot -> {path}")
                continue

            # ---- video ----
            if head == "video":
                if collector.frame_count == 0:
                    print("  No frames recorded.")
                    continue
                print(f"  Exporting {collector.frame_count} frames to video ...")
                path = collector.export_video(fps=5)
                print(f"  Video saved -> {path}")
                continue

            # ---- frames ----
            if head == "frames":
                if collector.frame_count == 0:
                    print("  No frames recorded.")
                    continue
                print(f"  Exporting {collector.frame_count} frames ...")
                d = collector.export_frames()
                print(f"  {collector.frame_count} PNGs saved -> {d}/")
                continue

            # ---- auto ----
            if head == "auto":
                auto_save = not auto_save
                print(f"  Auto-save: {'ON' if auto_save else 'OFF'}")
                continue

            # ---- list ----
            if head == "list":
                obj_map = ctrl.get_object_map()
                print(f"  Objects in {ctrl.scene_name} ({len(obj_map)} types):")
                for name, info in sorted(obj_map.items()):
                    p = info["position"]
                    vis = "visible" if info["visible"] else "hidden"
                    print(f"    {name:<25s} ({p.x:5.1f}, {p.y:5.1f}, {p.z:5.1f})  {vis}")
                continue

            # ---- where <obj> ----
            if head == "where":
                if not tail:
                    print("  Usage: where <object name>")
                    continue
                d = ctrl.distance_to(tail)
                if d is None:
                    print(f"  No object matching '{tail}'.")
                else:
                    print(f"  Distance to '{tail}': {d:.2f} m")
                continue

            # ---- desc ----
            if head == "desc":
                print("  Calling VLM to describe the scene ...")
                caption = _describe_view(result)
                if caption:
                    print(f"  VLM: {caption}")
                continue

            # ---- detect ----
            if head == "detect":
                pipeline = _get_pipeline()
                if pipeline is None:
                    continue
                print("  Running YOLO detection ...")
                try:
                    dets = pipeline.process(result.sensor_data.rgb, controller=ctrl)
                    _print_detections(dets, width)
                    # Save annotated frame with detection bboxes
                    annotated = draw_annotations(result.sensor_data.rgb, dets)
                    os.makedirs(os.path.join(collector.session_dir, "frames"), exist_ok=True)
                    fpath = os.path.join(collector.session_dir, "frames",
                                         f"detect_{collector.frame_count:04d}.png")
                    cv2.imwrite(fpath, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                    print(f"  Annotated frame saved -> {fpath}")
                except Exception as e:
                    print(f"  Detection failed: {e}")
                continue

            # ---- look <obj> ----
            if head == "look":
                if not tail:
                    print("  Usage: look <object name>")
                    continue
                obj_map = ctrl.get_object_map()
                info = obj_map.get(tail)
                if info is None:
                    for k, v in obj_map.items():
                        if k.lower() == tail.lower():
                            info = v
                            break
                if info is None:
                    print(f"  '{tail}' not found. Use 'list'.")
                    continue
                print(f"  Turning to face '{tail}' ...")
                for result in ctrl.look_at(info["position"]):
                    collector.record(result)
                    _print_state(result, ctrl)
                    if auto_save:
                        collector.save_screenshot(result, f"look_{tail}")
                continue

            # ---- go <obj> ----
            if head == "go":
                if not tail:
                    print("  Usage: go <object name>")
                    continue
                print(f"  Planning A* path to '{tail}' ...")
                count = 0
                try:
                    for result in ctrl.navigate_to_object(tail):
                        count += 1
                        collector.record(result)
                        _print_state(result, ctrl)
                        if auto_save:
                            collector.save_screenshot(result, f"goto_{tail}")
                except Exception as exc:
                    print(f"  Navigation error: {exc}")
                if count == 0:
                    print(f"  Could not navigate to '{tail}'. Check 'list' for valid names.")
                else:
                    d = ctrl.distance_to(tail)
                    dst_str = f"{d:.2f}m" if d else "?"
                    print(f"  Arrived near '{tail}' ({count} steps, distance: {dst_str})")
                continue

            # ---- scene <n> ----
            if head == "scene":
                if not tail:
                    print(f"  Current scene: {ctrl.scene_name}")
                    continue
                try:
                    idx = int(tail)
                    name = f"FloorPlan{idx}"
                except ValueError:
                    name = tail
                try:
                    ctrl.load_scene(name)
                    result = ctrl.get_current_view()
                    collector.record(result)
                    print(f"  Loaded: {name}")
                    _print_state(result, ctrl)
                    if auto_save:
                        collector.save_screenshot(result, "loaded")
                except Exception as exc:
                    print(f"  Failed: {exc}")
                continue

            # ---- movement shortcuts ----
            if head in SHORTCUT:
                action = SHORTCUT[head]
                result = ctrl.step(action)
                collector.record(result)
                _print_state(result, ctrl)
                if auto_save:
                    collector.save_screenshot(result, SHORTCUT_DESC[head])
                continue

            print(f"  Unknown: '{head}'. Type 'help'.")

    # --- session end: auto-export ---
    if collector.frame_count > 0:
        try:
            path = collector.export_video(fps=5)
            print(f"  Trajectory video saved -> {path}")
        except Exception as exc:
            print(f"  Video export failed: {exc}")
        try:
            d = collector.export_frames()
            count = len([f for f in os.listdir(d) if f.endswith(".png")])
            print(f"  Annotated frames: {count} -> {d}/")
        except Exception as exc:
            print(f"  Frame export failed: {exc}")
    print(f"  Session dir: {collector.session_dir}")
    print("  Controller closed.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Terminal-based manual control with A* pathfinding")
    p.add_argument("--scene", default="FloorPlan1")
    p.add_argument("--width", type=int, default=800)
    p.add_argument("--height", type=int, default=600)
    args = p.parse_args()
    run(scene=args.scene, width=args.width, height=args.height)


if __name__ == "__main__":
    main()
