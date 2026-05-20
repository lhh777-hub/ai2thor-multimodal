"""
Evaluate YOLO detection quality against AI2-THOR ground truth.

Generates a validation set using instance segmentation labels, then
runs ``model.val()`` to compute mAP, precision, and recall per class.

Usage::

    # Evaluate base YOLO (before fine-tuning) on held-out scenes 11-15
    python src/perception/finetune/eval.py --model yolov8n.pt --num-scenes 5

    # Evaluate fine-tuned model (after training)
    python src/perception/finetune/eval.py --model runs/train/weights/best.pt --num-scenes 5

    # Compare: run both and diff the per-class AP columns
    # Scenes 11-15 are never seen during training (which uses scenes 1-10)
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.controller.thor import ThorController
from src.perception.class_config import load_config
from src.common.logger import setup_logger

logger = setup_logger("eval")


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def collect_validation_set(output_dir: str, start_scene: int = 11,
                           num_scenes: int = 5,
                           steps_per_scene: int = 100,
                           width: int = 640, height: int = 480,
                           classes_path: str | None = None,
                           valid_class_ids: set[int] | None = None):
    """Walk through AI2-THOR scenes and auto-label a validation set.

    By default uses FloorPlan11-15 (held-out from training scenes 1-10).
    If *valid_class_ids* is given, labels outside this set are skipped.
    """
    import cv2
    import shutil

    cfg = load_config(classes_path)
    img_dir = os.path.join(output_dir, "images", "val")
    lbl_dir = os.path.join(output_dir, "labels", "val")
    # Clear old data to avoid mixing with stale labels
    if os.path.exists(img_dir):
        shutil.rmtree(img_dir)
    if os.path.exists(lbl_dir):
        shutil.rmtree(lbl_dir)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    actions = ["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT",
               "TURN_LEFT_SMALL", "TURN_RIGHT_SMALL",
               "LOOK_UP", "LOOK_DOWN"]
    frame_idx = 0
    total_labels = 0
    skipped_by_model = 0
    valid_ids = valid_class_ids  # model-supported class IDs

    with ThorController(width=width, height=height, visibility_distance=10.0,
                        render_instance_seg=True) as ctrl:
        for s in range(start_scene, start_scene + num_scenes):
            scene = f"FloorPlan{s}"
            try:
                ctrl.load_scene(scene, seed=1000 + s)
            except Exception:
                logger.warning("Scene %s unavailable, skipping", scene)
                continue

            logger.info("[%s] collecting validation frames ...", scene)

            for _ in range(steps_per_scene):
                action = random.choice(actions)
                result = ctrl.step(action)

                if not result.success and action in ("MOVE_FORWARD", "MOVE_BACK"):
                    continue

                labels = _extract_labels(result.raw_event, cfg)
                if not labels:
                    continue

                # Filter to model-supported classes
                if valid_ids is not None:
                    original = len(labels)
                    labels = [l for l in labels if l[0] in valid_ids]
                    skipped_by_model += original - len(labels)

                if not labels:
                    continue

                img_name = f"{scene}_{frame_idx:04d}"
                cv2.imwrite(
                    os.path.join(img_dir, f"{img_name}.png"),
                    cv2.cvtColor(result.sensor_data.rgb, cv2.COLOR_RGB2BGR),
                )
                with open(os.path.join(lbl_dir, f"{img_name}.txt"), "w") as f:
                    for cls_id, cx, cy, bw, bh in labels:
                        f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

                frame_idx += 1
                total_labels += len(labels)

    # Write val data.yaml (YOLO requires 'train' + 'val' keys)
    # Only include classes the model knows about
    if valid_ids is not None:
        used_names = [cfg.class_names[i] for i in sorted(valid_ids) if i < len(cfg.class_names)]
        nc = len(used_names)
    else:
        used_names = cfg.class_names
        nc = cfg.num_classes

    yaml_path = os.path.join(output_dir, "data_val.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"# Auto-generated validation set\n")
        f.write(f"path: {os.path.abspath(output_dir)}\n")
        f.write(f"train: images/val\n")
        f.write(f"val: images/val\n")
        f.write(f"nc: {nc}\n")
        f.write(f"names: {used_names}\n")

    print(f"\n  Validation set: {frame_idx} images, {total_labels} labels")
    if skipped_by_model:
        print(f"  Labels skipped (class not in model): {skipped_by_model}")
    print(f"  Model-compatible classes: {nc}")
    return yaml_path


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model_path: str, data_yaml: str, device: str = "cpu"):
    """Run YOLO validation and print a formatted table."""
    import numpy as np
    from ultralytics import YOLO

    print(f"\n  Model: {model_path}")
    print(f"  Data:  {data_yaml}")
    print(f"  Device: {device}\n")

    model = YOLO(model_path)
    metrics = model.val(data=data_yaml, device=device, verbose=False)

    # --- Overall summary ---
    map50  = getattr(metrics.box, "map50", 0.0)
    map95  = getattr(metrics.box, "map", 0.0)
    prec   = getattr(metrics.box, "p", [0.0])
    recall = getattr(metrics.box, "r", [0.0])

    print(f"  ╔{'═'*54}╗")
    print(f"  ║  {'Overall Metrics':^52s} ║")
    print(f"  ╠{'═'*25}╤{'═'*28}╣")
    print(f"  ║  {'mAP@0.5':>22s} │ {map50:>25.4f} ║")
    print(f"  ║  {'mAP@0.5:0.95':>22s} │ {map95:>25.4f} ║")
    print(f"  ║  {'Precision':>22s} │ {float(np.mean(prec)):>25.4f} ║")
    print(f"  ║  {'Recall':>22s} │ {float(np.mean(recall)):>25.4f} ║")
    print(f"  ╚{'═'*25}╧{'═'*28}╝")

    # --- Per-class table ---
    # metrics.box.ap gives AP values, metrics.box.ap_class_index gives the
    # corresponding class IDs.  Together they map correctly to class names.
    ap_per_class = getattr(metrics.box, "ap", None)
    ap_class_idx = getattr(metrics.box, "ap_class_index", None)
    if ap_per_class is not None and len(ap_per_class) > 0:
        with open(data_yaml) as f:
            names = yaml.safe_load(f).get("names", [])
        # Pair AP with the correct class name using ap_class_index
        if ap_class_idx is not None and len(ap_class_idx) == len(ap_per_class):
            indexed = [(ap_per_class[i], names[ap_class_idx[i]] if ap_class_idx[i] < len(names) else f"cls_{ap_class_idx[i]}")
                       for i in range(len(ap_per_class))]
        else:
            # Fallback: assume contiguous from 0
            indexed = [(ap_per_class[i], names[i] if i < len(names) else f"cls_{i}")
                       for i in range(len(ap_per_class))]
        indexed.sort(key=lambda x: x[0], reverse=True)

        # Count GT instances per class from the label files
        # data.yaml has "path: <abs>/data_val", "val: images/val"
        with open(data_yaml) as f:
            ydata = yaml.safe_load(f)
        data_root = ydata.get("path", os.path.dirname(data_yaml))
        lbl_dir = os.path.join(data_root, "labels", "val")
        gt_counts: dict[str, int] = {}
        n_label_files = 0
        if os.path.isdir(lbl_dir):
            for fname in os.listdir(lbl_dir):
                if fname.endswith(".txt"):
                    n_label_files += 1
                    with open(os.path.join(lbl_dir, fname)) as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                cls_id = int(parts[0])
                                if cls_id < len(names):
                                    gt_counts[names[cls_id]] = gt_counts.get(names[cls_id], 0) + 1
        print(f"  Label files found: {n_label_files}, unique classes with GT: {len(gt_counts)}")
        print(f"  GT classes: {sorted(gt_counts.keys())[:10]}...")

        # Build a combined list: for each class, find GT count and AP
        # Use class index as the common key to avoid name mismatch
        print(f"\n  ┌{'─'*22}┬{'─'*8}┬{'─'*10}┬{'─'*10}┐")
        print(f"  │ {'Class':<20s} │ {'GT':>6s} │ {'AP@0.5':>8s} │ {'Status':>8s} │")
        print(f"  ├{'─'*22}┼{'─'*8}┼{'─'*10}┼{'─'*10}┤")

        printed = 0
        for i, (ap, name) in enumerate(indexed):
            gt_n = gt_counts.get(name.strip(), 0)
            if gt_n == 0:
                continue  # skip classes without GT instances
            if ap >= 0.6:
                status = "GOOD"
            elif ap >= 0.3:
                status = "OK"
            elif ap > 0:
                status = "LOW"
            else:
                status = "NONE"
            print(f"  │ {name:<20s} │ {gt_n:>6d} │ {ap:>8.4f} │ {status:>8s} │")
            printed += 1

        print(f"  └{'─'*22}┴{'─'*8}┴{'─'*10}┴{'─'*10}┘")

        classes_with_gt = sorted(gt_counts.keys())
        classes_detected = [name for ap, name in indexed if ap > 0]
        print(f"  {len(classes_with_gt)} classes with GT labels")
        print(f"  {printed} rows printed")

    # --- Guidance ---
    print(f"\n  Status legend:")
    print(f"    GOOD  AP>=0.6  — reliable detection")
    print(f"    OK    AP>=0.3  — usable, fine-tuning will improve")
    print(f"    LOW   AP>0     — poor, needs fine-tuning or more data")
    print(f"    NONE  AP=0     — never detected (not in COCO / no samples)")
    print(f"\n  Re-run after fine-tuning to compare.\n")

    return metrics


# ---------------------------------------------------------------------------
# Label extraction
# ---------------------------------------------------------------------------

def _extract_labels(event, cfg) -> list[tuple[int, float, float, float, float]]:
    import numpy as np

    if not hasattr(event, "instance_segmentation_frame"):
        return []
    seg = event.instance_segmentation_frame
    if seg is None:
        return []

    img_h, img_w = seg.shape[:2]
    colour_to_id = getattr(event, "color_to_object_id", {})
    if not colour_to_id:
        return []

    labels = []
    seen = set()

    for colour, obj_id in colour_to_id.items():
        if obj_id in seen:
            continue
        seen.add(obj_id)

        obj = event.get_object(obj_id)
        if obj is None:
            continue
        cls_id = cfg.thor_to_class_id(obj.get("objectType", ""))
        if cls_id is None:
            continue

        colour_np = np.array(colour, dtype=np.uint8)
        mask = np.all(seg == colour_np, axis=-1)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue

        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()
        bw, bh = float(xmax - xmin), float(ymax - ymin)
        if bw <= 0 or bh <= 0:
            continue

        cx = (xmin + xmax) / 2.0 / img_w
        cy = (ymin + ymax) / 2.0 / img_h
        bw /= img_w
        bh /= img_h
        labels.append((cls_id, cx, cy, bw, bh))

    return labels


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Evaluate YOLO detection with mAP/Precision/Recall")
    p.add_argument("--model", default="yolov8n.pt",
                   help="YOLO model to evaluate (default: yolov8n.pt)")
    p.add_argument("--start-scene", type=int, default=11,
                   help="First validation scene (default: 11)")
    p.add_argument("--num-scenes", type=int, default=5,
                   help="Number of validation scenes (default: 5)")
    p.add_argument("--steps", type=int, default=100,
                   help="Random walk steps per scene (default: 100)")
    p.add_argument("--output", default="data_val",
                   help="Output directory (default: data_val/)")
    p.add_argument("--device", default="cpu",
                   help="Device: cpu, cuda:0, etc. (default: cpu)")
    p.add_argument("--classes", default=None,
                   help="Path to classes.yaml")
    p.add_argument("--data", default=None,
                   help="Use existing data.yaml (skip collection)")
    args = p.parse_args()

    # Detect the model's class count
    from ultralytics import YOLO
    model = YOLO(args.model)
    model_nc = getattr(model.model, "nc", 80)
    valid_ids = set(range(model_nc))
    print(f"  Model class count: {model_nc} (will only evaluate these classes)")

    if args.data and os.path.exists(args.data):
        data_yaml = args.data
        print(f"  Using existing dataset: {data_yaml}")
    else:
        print(f"  Collecting validation set (scenes {args.start_scene}-{args.start_scene + args.num_scenes - 1}) ...")
        data_yaml = collect_validation_set(
            output_dir=args.output,
            start_scene=args.start_scene,
            num_scenes=args.num_scenes,
            steps_per_scene=args.steps,
            classes_path=args.classes,
            valid_class_ids=valid_ids,
        )

    evaluate(model_path=args.model, data_yaml=data_yaml, device=args.device)


if __name__ == "__main__":
    main()
