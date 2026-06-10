"""
Compare YOLO-World vs fine-tuned YOLO against AI2-THOR ground truth.

Randomly walks through N scenes, runs both detectors on every frame,
and compares each against instance-segmentation ground truth.

Usage::

    python -m src.perception.eval_detectors --scenes 3 --steps 100

Output::

    outputs/eval_20260526_171500/
      per_frame.csv       ← per-frame metrics for every step
      summary.txt         ← overall comparison table
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.controller.thor import ThorController
from src.perception.class_config import load_config
from src.perception.detector import YOLODetector, YOLOWorldDetector, HybridDetector
from src.common.logger import setup_logger
from src.common.utils import box_iou

logger = setup_logger("eval_detectors")


# Label synonym groups — labels in the same group are considered equivalent
# for matching purposes.  This fixes the "couch vs sofa" / "fridge vs
# refrigerator" class of unfair penalisation against YOLO-World.
_LABEL_SYNONYMS: dict[str, str] = {
    "couch": "sofa", "sofa": "couch",
    "fridge": "refrigerator",
    "television": "tv", "tvset": "tv",
    "diningtable": "dining table", "dinner table": "dining table",
    "coffeetable": "coffee table", "end table": "coffee table",
    "pottedplant": "potted plant", "houseplant": "potted plant",
    "cellphone": "cell phone", "mobile phone": "cell phone",
    "wineglass": "wine glass",
    "microwaveoven": "microwave",
    "trashcan": "trash can", "garbage bin": "trash can",
    "lightswitch": "light switch",
    "ceilingfan": "ceiling fan",
    "bathtowel": "towel", "bath towel": "towel",
    "nightstand": "bedside table", "bedstand": "bedside table",
    "dresser": "wardrobe", "chest": "wardrobe",
    "cookingpot": "pot", "saucepan": "pot",
    "teddybear": "teddy bear",
    "tennisracket": "tennis racket",
    "baseballbat": "baseball bat",
    "hairdrier": "hair drier",
    "tissuebox": "tissue box",
    "creditcard": "credit card",
    "hotdog": "hot dog",
    "sportsball": "sports ball",
    "potted plant": "potted plant",  # self-mapping for canonical forms
}

# Structural / decorative objects that appear visually in AI2-THOR scenes
# but are NEVER in the instance-segmentation GT (they're wall textures or
# non-interactive geometry).  Detections of these are "practical TP" rather
# than false positives.
_STRUCTURAL_NO_GT: set[str] = {
    "door", "window", "wall", "floor", "ceiling", "ceiling fan",
    "light", "light switch", "curtain", "blinds", "rug", "carpet",
    "painting", "picture", "poster", "mirror", "counter", "countertop",
}


def _canonical_label(label: str) -> str:
    """Normalize a label so synonyms match."""
    key = label.lower().replace(" ", "").replace("-", "").replace("_", "")
    return _LABEL_SYNONYMS.get(key, label.lower())


# ---------------------------------------------------------------------------
# Ground-truth extraction
# ---------------------------------------------------------------------------

def _extract_ground_truth(event, cfg,
                          min_box_size: int = 20) -> list[dict]:
    """Extract ground-truth labels + boxes from AI2-THOR instance segmentation.

    Filters out boxes smaller than *min_box_size* in either dimension —
    these are too small / far to be reasonably detectable.

    Returns list of dicts: {label, bbox=(x1,y1,x2,y2)}.
    """
    seg = getattr(event, "instance_segmentation_frame", None)
    if seg is None:
        return []

    colour_to_id = getattr(event, "color_to_object_id", {})
    if not colour_to_id:
        return []

    gt: list[dict] = []
    seen = set()
    skipped_small = 0

    for colour, obj_id in colour_to_id.items():
        if obj_id in seen:
            continue
        seen.add(obj_id)

        obj = event.get_object(obj_id)
        if obj is None:
            continue
        thor_type = obj.get("objectType", "")
        cls_id = cfg.thor_to_class_id(thor_type)
        if cls_id is None:
            continue

        colour_np = np.array(colour, dtype=np.uint8)
        mask = np.all(seg == colour_np, axis=-1)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue

        x1, y1 = float(xs.min()), float(ys.min())
        x2, y2 = float(xs.max()), float(ys.max())
        bw, bh = x2 - x1, y2 - y1

        if bw < min_box_size or bh < min_box_size:
            skipped_small += 1
            continue

        gt.append({
            "label": cfg.class_names[cls_id],
            "bbox": (x1, y1, x2, y2),
            "size": (bw, bh),
        })

    return gt


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def _match_detections(detections: list, ground_truth: list[dict],
                      iou_threshold: float = 0.5) -> tuple[int, int, int, int, list]:
    """Match detections to ground truth via greedy IoU assignment.

    Uses canonical labels to handle synonyms (couch↔sofa, fridge↔refrigerator).
    Structural detections (door, window, etc.) with no GT are tracked as
    *structural_fp* rather than regular FP.

    Returns (tp, fp, fn, structural_fp, details).
    """
    # Canonicalize GT boxes grouped by canonical label
    gt_by_canon: dict[str, list[tuple[str, tuple]]] = defaultdict(list)
    for gt in ground_truth:
        canon = _canonical_label(gt["label"])
        gt_by_canon[canon].append((gt["label"], gt["bbox"]))

    # Canonicalize detection boxes
    det_by_canon: dict[str, list[tuple[str, tuple, bool]]] = defaultdict(list)
    for d in detections:
        canon = _canonical_label(d.label)
        is_structural = d.label.lower() in _STRUCTURAL_NO_GT
        det_by_canon[canon].append(
            (d.label, (d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2), is_structural))

    tp_total = 0
    fp_total = 0
    fn_total = 0
    structural_fp_total = 0
    class_details: list[dict] = []

    all_canon = set(gt_by_canon.keys()) | set(det_by_canon.keys())

    for canon in sorted(all_canon):
        gt_entries = gt_by_canon.get(canon, [])  # [(orig_label, bbox), ...]
        det_entries = det_by_canon.get(canon, [])  # [(orig_label, bbox, is_structural), ...]

        if not gt_entries and not det_entries:
            continue

        gt_boxes = [b for _, b in gt_entries]
        det_boxes = [b for _, b, _ in det_entries]
        det_structural = [s for _, _, s in det_entries]

        matched_gt = set()
        matched_det = set()

        for di, dbox in enumerate(det_boxes):
            best_iou = 0.0
            best_gi = -1
            for gi, gbox in enumerate(gt_boxes):
                if gi in matched_gt:
                    continue
                iou = box_iou(dbox, gbox)
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi
            if best_iou >= iou_threshold:
                matched_gt.add(best_gi)
                matched_det.add(di)

        tp = len(matched_det)

        # Count unmatched detections — structural ones are NOT penalised
        # because AI2-THOR doesn't provide GT for doors/windows/walls.
        regular_fp = 0
        structural_fp = 0
        for di in range(len(det_boxes)):
            if di not in matched_det:
                if det_structural[di]:
                    structural_fp += 1
                else:
                    regular_fp += 1

        fn = len(gt_boxes) - len(matched_gt)

        tp_total += tp
        fp_total += regular_fp
        fn_total += fn
        structural_fp_total += structural_fp

        prec = tp / (tp + regular_fp) if (tp + regular_fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        class_details.append({
            "class": canon, "tp": tp, "fp": regular_fp, "fn": fn,
            "structural_fp": structural_fp,
            "precision": prec, "recall": rec, "f1": f1,
            "gt_count": len(gt_boxes), "det_count": len(det_boxes),
        })

    return tp_total, fp_total, fn_total, structural_fp_total, class_details


# ---------------------------------------------------------------------------
# Scene random walk
# ---------------------------------------------------------------------------

def _walk_scene(ctrl: ThorController, scene: str, steps: int,
                actions_per_step: int = 2) -> list[dict]:
    """Walk randomly through *scene* and collect frame data.

    Returns list of dicts: {rgb, gt, step}.
    """
    actions = ["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT",
               "TURN_LEFT_SMALL", "TURN_RIGHT_SMALL",
               "LOOK_UP", "LOOK_DOWN"]

    cfg = load_config()
    frames: list[dict] = []
    consecutive_fails = 0

    for _ in range(steps):
        # 1–3 random moves per "step" to spread out
        for _ in range(random.randint(1, actions_per_step)):
            action = random.choice(actions)
            result = ctrl.step(action)
            if not result.success:
                consecutive_fails += 1
                if consecutive_fails > 5:
                    # Stuck — turn around
                    ctrl.step("TURN_LEFT")
                    ctrl.step("TURN_LEFT")
                    consecutive_fails = 0
            else:
                consecutive_fails = 0

        # Collect ground truth after the movement
        result = ctrl.get_current_view()
        gt = _extract_ground_truth(result.raw_event, cfg)

        if gt:  # Only keep frames with at least one GT label
            frames.append({
                "rgb": result.sensor_data.rgb.copy(),
                "gt": gt,
                "step": ctrl.step_count,
            })

    return frames


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_evaluation(
    yolo_model: str = "runs/detect/runs/train/weights/best.pt",
    yolo_conf: float = 0.3,
    world_model: str = "yolov8s-worldv2.pt",
    world_conf: float = 0.15,
    num_scenes: int = 3,
    start_scene: int = 1,
    steps_per_scene: int = 100,
    output_dir: str = "outputs",
    min_box_size: int = 20,
    quiet: bool = False,
) -> dict | None:

    # Pick random scenes from FloorPlan1–30
    all_scenes = [f"FloorPlan{i}" for i in range(1, 31)]
    scenes = random.sample(all_scenes, min(num_scenes, len(all_scenes)))
    print(f"\n  Scenes selected: {', '.join(scenes)}")
    print(f"  Steps per scene: {steps_per_scene}")
    print(f"  Min GT box size: {min_box_size}x{min_box_size} px")
    print(f"  YOLO-World: {world_model} (conf={world_conf})")
    print(f"  Fine-tuned YOLO: {yolo_model} (conf={yolo_conf})\n")

    # ---- Load detectors ----
    if not quiet:
        print("  Loading detectors ...")
    try:
        det_world = YOLOWorldDetector(model_name=world_model, confidence=world_conf)
        # Set vocabulary to ONLY AI2-THOR-mapped classes (not all 114 COCO classes).
        # This is the crucial difference: YOLO-World should only look for objects
        # that can actually appear in AI2-THOR indoor scenes.
        cfg = load_config()
        thor_relevant = sorted(set(cfg.thor_to_names.values()))
        det_world.set_classes(thor_relevant)
        if not quiet:
            print(f"  YOLO-World base vocabulary: {len(thor_relevant)} AI2-THOR classes")
    except ImportError as e:
        print(f"  YOLO-World not available: {e}")
        print("  Install: pip install -U ultralytics")
        return {} if quiet else None
    det_yolo = YOLODetector(model_name=yolo_model, confidence=yolo_conf)
    det_hybrid = None  # Created per-scene after scene load (needs controller)

    # ---- Setup output ----
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(output_dir, f"eval_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "per_frame.csv")
    csv_f = open(csv_path, "w", encoding="utf-8")
    csv_f.write("scene,step,detector,tp,fp,fn,structural_fp,precision,recall,f1\n")

    # Accumulators
    world_stats = {"tp": 0, "fp": 0, "fn": 0, "structural_fp": 0, "frames": 0,
                   "total_gt": 0, "synonym_matches": 0}
    yolo_stats = {"tp": 0, "fp": 0, "fn": 0, "structural_fp": 0, "frames": 0,
                  "total_gt": 0}
    world_class_accum: dict[str, dict] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    yolo_class_accum: dict[str, dict] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    hybrid_class_accum: dict[str, dict] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    hybrid_stats = {"tp": 0, "fp": 0, "fn": 0, "structural_fp": 0, "frames": 0,
                    "total_gt": 0}
    gt_classes_all: set[str] = set()      # All classes in GT
    world_det_classes: set[str] = set()   # All classes YOLO-World outputs
    yolo_det_classes: set[str] = set()    # All classes YOLO-finetuned outputs
    hybrid_det_classes: set[str] = set()  # All classes Hybrid outputs

    # ---- Evaluate each scene ----
    for si, scene in enumerate(scenes):
        print(f"\n  [{si+1}/{len(scenes)}] {scene} — walking ...")

        with ThorController(width=640, height=480, visibility_distance=10.0,
                            render_instance_seg=True) as ctrl:
            ctrl.load_scene(scene, seed=42 + si)

            # Narrow YOLO-World vocabulary to this scene
            det_world.set_classes_from_scene(ctrl)
            # Hybrid: fine-tuned (primary) + YOLO-World (gap-filler)
            # Create fresh per-scene so YOLO-World vocab is scene-specific
            det_hybrid = HybridDetector(det_yolo, det_world, merge_iou=0.5)
            n_classes = len(det_world.classes)
            if not quiet:
                print(f"  Vocabulary: {n_classes} classes  |  Hybrid: YOLO-F + YOLO-W")

            frames = _walk_scene(ctrl, scene, steps_per_scene)
            print(f"  Collected {len(frames)} frames with GT labels")

            for fi, frame in enumerate(frames):
                rgb = frame["rgb"]
                gt = frame["gt"]
                step = frame["step"]

                # --- YOLO-World ---
                dets_w = det_world.detect(rgb)
                tp_w, fp_w, fn_w, sfp_w, cls_w = _match_detections(dets_w, gt)
                world_stats["tp"] += tp_w
                world_stats["fp"] += fp_w
                world_stats["fn"] += fn_w
                world_stats["structural_fp"] += sfp_w
                world_stats["frames"] += 1
                world_stats["total_gt"] += len(gt)
                for g in gt:
                    gt_classes_all.add(g["label"])
                for d in dets_w:
                    world_det_classes.add(d.label)
                for cd in cls_w:
                    acc = world_class_accum[cd["class"]]
                    acc["tp"] += cd["tp"]
                    acc["fp"] += cd["fp"]
                    acc["fn"] += cd["fn"]

                # --- Fine-tuned YOLO ---
                dets_y = det_yolo.detect(rgb)
                tp_y, fp_y, fn_y, sfp_y, cls_y = _match_detections(dets_y, gt)
                yolo_stats["tp"] += tp_y
                yolo_stats["fp"] += fp_y
                yolo_stats["fn"] += fn_y
                yolo_stats["structural_fp"] += sfp_y
                yolo_stats["frames"] += 1
                yolo_stats["total_gt"] += len(gt)
                for d in dets_y:
                    yolo_det_classes.add(d.label)
                for cd in cls_y:
                    acc = yolo_class_accum[cd["class"]]
                    acc["tp"] += cd["tp"]
                    acc["fp"] += cd["fp"]
                    acc["fn"] += cd["fn"]

                # --- Hybrid (YOLO-F + YOLO-W) ---
                if det_hybrid is not None:
                    dets_h = det_hybrid.detect(rgb)
                    tp_h, fp_h, fn_h, sfp_h, cls_h = _match_detections(dets_h, gt)
                    hybrid_stats["tp"] += tp_h
                    hybrid_stats["fp"] += fp_h
                    hybrid_stats["fn"] += fn_h
                    hybrid_stats["structural_fp"] += sfp_h
                    hybrid_stats["frames"] += 1
                    hybrid_stats["total_gt"] += len(gt)
                    for d in dets_h:
                        hybrid_det_classes.add(d.label)
                    for cd in cls_h:
                        acc = hybrid_class_accum[cd["class"]]
                        acc["tp"] += cd["tp"]
                        acc["fp"] += cd["fp"]
                        acc["fn"] += cd["fn"]

                # Write per-frame CSV
                pw = _safe_div(tp_w, tp_w + fp_w), _safe_div(tp_w, tp_w + fn_w)
                py = _safe_div(tp_y, tp_y + fp_y), _safe_div(tp_y, tp_y + fn_y)
                csv_f.write(f"{scene},{step},YOLO-World,{tp_w},{fp_w},{fn_w},{sfp_w},"
                            f"{pw[0]:.4f},{pw[1]:.4f},"
                            f"{_f1(pw[0], pw[1]):.4f}\n")
                csv_f.write(f"{scene},{step},YOLO-finetuned,{tp_y},{fp_y},{fn_y},{sfp_y},"
                            f"{py[0]:.4f},{py[1]:.4f},"
                            f"{_f1(py[0], py[1]):.4f}\n")

                if (fi + 1) % 20 == 0 and not quiet:
                    wp = _safe_div(world_stats["tp"], world_stats["tp"] + world_stats["fp"])
                    wr = _safe_div(world_stats["tp"], world_stats["tp"] + world_stats["fn"])
                    yp = _safe_div(yolo_stats["tp"], yolo_stats["tp"] + yolo_stats["fp"])
                    yr = _safe_div(yolo_stats["tp"], yolo_stats["tp"] + yolo_stats["fn"])
                    hp = _safe_div(hybrid_stats["tp"], hybrid_stats["tp"] + hybrid_stats["fp"])
                    hr = _safe_div(hybrid_stats["tp"], hybrid_stats["tp"] + hybrid_stats["fn"])
                    print(f"  [{fi+1:3d}] Hybrid: P={hp:.3f} R={hr:.3f} F1={_f1(hp,hr):.3f}  |  "
                          f"YOLO-F: P={yp:.3f} R={yr:.3f} F1={_f1(yp,yr):.3f}  |  "
                          f"YOLO-W: P={wp:.3f} R={wr:.3f} F1={_f1(wp,wr):.3f}")

    csv_f.close()

    # ---- Diagnostic ----
    if not quiet:
        print(f"\n{'='*70}")
        print(f"  DIAGNOSTIC")
        print(f"{'='*70}")
        avg_gt = world_stats["total_gt"] / max(world_stats["frames"], 1)
        print(f"  Avg GT boxes/frame: {avg_gt:.1f}  |  "
              f"Total GT instances: {world_stats['total_gt']}")
        print(f"  GT classes: {len(gt_classes_all)}  |  "
              f"YOLO-W detected classes: {len(world_det_classes)}  |  "
              f"YOLO-F detected classes: {len(yolo_det_classes)}")
        gt_only_w = sorted(gt_classes_all - world_det_classes)
        gt_only_y = sorted(gt_classes_all - yolo_det_classes)
        print(f"  GT classes YOLO-W NEVER detected: {len(gt_only_w)}")
        if gt_only_w:
            print(f"    {', '.join(gt_only_w[:15])}")
            if len(gt_only_w) > 15:
                print(f"    ... +{len(gt_only_w)-15} more")
        print(f"  GT classes YOLO-F NEVER detected: {len(gt_only_y)}")
        if gt_only_y:
            print(f"    {', '.join(gt_only_y[:15])}")
            if len(gt_only_y) > 15:
                print(f"    ... +{len(gt_only_y)-15} more")
        ghost_w = sorted(world_det_classes - gt_classes_all)
        ghost_y = sorted(yolo_det_classes - gt_classes_all)
        if ghost_w:
            print(f"  YOLO-W detects but NOT in GT: {', '.join(ghost_w[:10])}")
        if ghost_y:
            print(f"  YOLO-F detects but NOT in GT: {', '.join(ghost_y[:10])}")

    # ---- Summary ----
    if not quiet:
        print(f"\n{'='*70}")
        print(f"  OVERALL RESULTS  ({world_stats['frames']} frames, {len(scenes)} scenes)")
        print(f"  IoU threshold: 0.5  |  Min GT box: {min_box_size}×{min_box_size} px")
        print(f"{'='*70}")

    wp = _safe_div(world_stats["tp"], world_stats["tp"] + world_stats["fp"])
    wr = _safe_div(world_stats["tp"], world_stats["tp"] + world_stats["fn"])
    yp = _safe_div(yolo_stats["tp"], yolo_stats["tp"] + yolo_stats["fp"])
    yr = _safe_div(yolo_stats["tp"], yolo_stats["tp"] + yolo_stats["fn"])
    hp = _safe_div(hybrid_stats["tp"], hybrid_stats["tp"] + hybrid_stats["fp"])
    hr = _safe_div(hybrid_stats["tp"], hybrid_stats["tp"] + hybrid_stats["fn"])

    results = [
        ("Hybrid (YOLO-F+W)", hybrid_stats["tp"], hybrid_stats["fp"], hybrid_stats["fn"],
         hp, hr, _f1(hp, hr), hybrid_stats["structural_fp"]),
        ("YOLO-finetuned", yolo_stats["tp"], yolo_stats["fp"], yolo_stats["fn"],
         yp, yr, _f1(yp, yr), yolo_stats["structural_fp"]),
        ("YOLO-World", world_stats["tp"], world_stats["fp"], world_stats["fn"],
         wp, wr, _f1(wp, wr), world_stats["structural_fp"]),
    ]

    if not quiet:
        print(f"\n  {'Detector':<18s} {'TP':>6s} {'FP':>6s} {'FN':>6s} {'StrFP':>6s}  "
              f"{'Prec':>7s} {'Recall':>7s} {'F1':>7s}")
        print(f"  {'-'*18} {'-'*6} {'-'*6} {'-'*6} {'-'*6}  {'-'*7} {'-'*7} {'-'*7}")
        for name, tp, fp, fn, prec, rec, f1, sfp in results:
            print(f"  {name:<18s} {tp:>6d} {fp:>6d} {fn:>6d} {sfp:>6d}  "
                  f"{prec:>6.3f}  {rec:>6.3f}  {f1:>6.3f}")
        print(f"  StrFP = structural detections (door/window/wall) with no GT available")

    # Save summary
    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Scenes: {', '.join(scenes)}\n")
        f.write(f"Frames: {world_stats['frames']}\n")
        f.write(f"YOLO-World: {world_model} conf={world_conf}\n")
        f.write(f"YOLO-finetuned: {yolo_model} conf={yolo_conf}\n\n")
        f.write(f"{'Detector':<18s} {'TP':>6s} {'FP':>6s} {'FN':>6s} "
                f"{'StrFP':>6s}  {'Prec':>7s} {'Recall':>7s} {'F1':>7s}\n")
        for name, tp, fp, fn, prec, rec, f1, sfp in results:
            f.write(f"{name:<18s} {tp:>6d} {fp:>6d} {fn:>6d} {sfp:>6d}  "
                    f"{prec:>6.3f}  {rec:>6.3f}  {f1:>6.3f}\n")

    # Per-class comparison (only classes with ≥5 GT instances)
    if not quiet:
        print(f"\n  Per-class F1 (≥5 GT instances):")
        print(f"  {'Class':<20s} {'YOLO-W':>8s} {'YOLO-F':>8s} {'GT':>6s}  Win")
        print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*6}  {'-'*4}")
    all_classes = set(world_class_accum.keys()) | set(yolo_class_accum.keys())
    rows = []
    for cls in all_classes:
        wa = world_class_accum.get(cls, {"tp": 0, "fp": 0, "fn": 0})
        ya = yolo_class_accum.get(cls, {"tp": 0, "fp": 0, "fn": 0})
        gt_total = wa["tp"] + wa["fn"]
        if gt_total < 5:
            continue
        wp_c = _safe_div(wa["tp"], wa["tp"] + wa["fp"])
        wr_c = _safe_div(wa["tp"], wa["tp"] + wa["fn"])
        yp_c = _safe_div(ya["tp"], ya["tp"] + ya["fp"])
        yr_c = _safe_div(ya["tp"], ya["tp"] + ya["fn"])
        wf1 = _f1(wp_c, wr_c)
        yf1 = _f1(yp_c, yr_c)
        diff = wf1 - yf1
        better = "W" if diff > 0.05 else ("F" if diff < -0.05 else "—")
        rows.append((cls, wf1, yf1, gt_total, diff, better))
    # Sort by absolute F1 difference (biggest gap first)
    rows.sort(key=lambda r: abs(r[4]), reverse=True)
    if not quiet:
        for cls, wf1, yf1, gt_total, diff, better in rows:
            print(f"  {cls:<20s} {wf1:>8.3f} {yf1:>8.3f} {gt_total:>6d}  {better}")
        if not rows:
            print("  (no class had ≥5 GT instances)")

    # Quick win summary
    w_wins = sum(1 for r in rows if r[5] == "W")
    f_wins = sum(1 for r in rows if r[5] == "F")
    if not quiet:
        print(f"\n  YOLO-World wins: {w_wins} classes  |  "
              f"YOLO-finetuned wins: {f_wins} classes  |  "
              f"Ties: {len(rows) - w_wins - f_wins}")

    if not quiet:
        print(f"\n  Results saved → {out_dir}/")
        print(f"    per_frame.csv  — per-frame metrics")
        print(f"    summary.txt    — overall comparison\n")

    return {
        "world_conf": world_conf, "world_f1": _f1(wp, wr),
        "world_prec": wp, "world_rec": wr,
        "yolo_f1": _f1(yp, yr), "yolo_prec": yp, "yolo_rec": yr,
        "hybrid_f1": _f1(hp, hr), "hybrid_prec": hp, "hybrid_rec": hr,
        "scenes": scenes, "frames": world_stats["frames"],
    }


def _safe_div(a, b) -> float:
    return float(a) / float(b) if b > 0 else 0.0


def _f1(prec: float, rec: float) -> float:
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Compare YOLO-World vs fine-tuned YOLO against AI2-THOR ground truth")
    p.add_argument("--scenes", type=int, default=3,
                   help="Number of random scenes to evaluate (default: 3)")
    p.add_argument("--start-scene", type=int, default=1,
                   help="Min scene number (default: 1)")
    p.add_argument("--steps", type=int, default=100,
                   help="Random walk steps per scene (default: 100)")
    p.add_argument("--yolo-model", default="runs/detect/runs/train/weights/best.pt",
                   help="Fine-tuned YOLO model path")
    p.add_argument("--yolo-conf", type=float, default=0.3,
                   help="Fine-tuned YOLO confidence threshold (default: 0.3)")
    p.add_argument("--world-model", default="yolov8s-worldv2.pt",
                   help="YOLO-World model (default: yolov8s-worldv2.pt)")
    p.add_argument("--world-conf", type=float, default=0.15,
                   help="YOLO-World confidence threshold (default: 0.15)")
    p.add_argument("--output", default="outputs",
                   help="Output directory (default: outputs/)")
    p.add_argument("--min-box-size", type=int, default=20,
                   help="Minimum GT box width/height in pixels (default: 20)")
    p.add_argument("--world-confs", type=str, default=None,
                   help="Comma-separated YOLO-World confidence thresholds to sweep "
                        "(e.g. '0.10,0.15,0.20,0.25'). If set, overrides --world-conf.")
    args = p.parse_args()

    # Single confidence mode
    if args.world_confs is None:
        run_evaluation(
            yolo_model=args.yolo_model,
            yolo_conf=args.yolo_conf,
            world_model=args.world_model,
            world_conf=args.world_conf,
            num_scenes=args.scenes,
            start_scene=args.start_scene,
            steps_per_scene=args.steps,
            output_dir=args.output,
            min_box_size=args.min_box_size,
        )
    else:
        # Multi-confidence sweep mode
        confs = [float(c.strip()) for c in args.world_confs.split(",")]
        print(f"\n  YOLO-World confidence sweep: {confs}")
        print(f"  Fine-tuned YOLO baseline: conf={args.yolo_conf}")
        print(f"\n  {'Conf':>6s}  {'W-Prec':>7s} {'W-Rec':>7s} {'W-F1':>7s}  "
              f"{'Y-Prec':>7s} {'Y-Rec':>7s} {'Y-F1':>7s}")
        print(f"  {'-'*6}  {'-'*7} {'-'*7} {'-'*7}  {'-'*7} {'-'*7} {'-'*7}")
        best_result = None
        for wc in confs:
            result = run_evaluation(
                yolo_model=args.yolo_model,
                yolo_conf=args.yolo_conf,
                world_model=args.world_model,
                world_conf=wc,
                num_scenes=args.scenes,
                start_scene=args.start_scene,
                steps_per_scene=args.steps,
                output_dir=args.output,
                min_box_size=args.min_box_size,
                quiet=True,
            )
            if result is None:
                continue
            print(f"  {wc:>5.2f}   {result['world_prec']:>6.3f}  {result['world_rec']:>6.3f}  "
                  f"{result['world_f1']:>6.3f}  {result['yolo_prec']:>6.3f}  "
                  f"{result['yolo_rec']:>6.3f}  {result['yolo_f1']:>6.3f}")
            if best_result is None or result["world_f1"] > best_result["world_f1"]:
                best_result = result

        if best_result:
            print(f"\n  Best YOLO-World threshold: conf={best_result['world_conf']:.2f} "
                  f"(F1={best_result['world_f1']:.3f} vs YOLO-F F1={best_result['yolo_f1']:.3f})")


if __name__ == "__main__":
    main()
