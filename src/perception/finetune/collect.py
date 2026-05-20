"""
Auto-labeling data collector for YOLO fine-tuning.

Uses AI2-THOR's instance segmentation to generate bounding-box labels
automatically.  No manual annotation needed.

Usage::

    python src/perception/finetune/collect.py --scenes 10 --steps 200

Output::

    data/
      data.yaml              ← YOLO dataset config
      images/train/*.png     ← captured frames
      labels/train/*.txt     ← YOLO-format labels

Next step::

    yolo train data=data/data.yaml model=yolov8n.pt epochs=50 imgsz=640
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.controller.thor import ThorController
from src.perception.class_config import load_config, ClassConfig
from src.common.logger import setup_logger

logger = setup_logger("collect")

# Load class config (can be overridden via --classes argument)
_CONFIG: ClassConfig | None = None


def _get_config(config_path: str | None = None) -> ClassConfig:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = load_config(config_path)
    return _CONFIG


def _thor_to_coco(thor_type: str, cfg: ClassConfig) -> int | None:
    return cfg.thor_to_class_id(thor_type)


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def collect(scenes: int = 10, steps_per_scene: int = 200,
            output_dir: str = "data", width: int = 640, height: int = 480,
            config_path: str | None = None):
    """Walk through AI2-THOR scenes and auto-label every frame."""
    cfg = _get_config(config_path)

    img_dir = os.path.join(output_dir, "images", "train")
    lbl_dir = os.path.join(output_dir, "labels", "train")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    actions = ["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT",
               "TURN_LEFT_SMALL", "TURN_RIGHT_SMALL",
               "LOOK_UP", "LOOK_DOWN"]
    frame_idx = 0
    total_labels = 0

    with ThorController(width=width, height=height, visibility_distance=10.0,
                        render_instance_seg=True) as ctrl:
        for s in range(1, scenes + 1):
            scene = f"FloorPlan{s}"
            try:
                ctrl.load_scene(scene, seed=random.randint(0, 9999))
            except Exception:
                logger.warning("Scene %s unavailable, skipping", scene)
                continue

            logger.info("[%s] collecting ...", scene)

            for _ in range(steps_per_scene):
                action = random.choice(actions)
                result = ctrl.step(action)

                if not result.success and action in ("MOVE_FORWARD", "MOVE_BACK"):
                    continue

                labels = _extract_labels(result.raw_event, cfg)
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

                if frame_idx % 100 == 0:
                    print(f"  {frame_idx} frames, {total_labels} labels")

    # Write dataset config (use the same class names from the YAML config)
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"# Auto-generated from AI2-THOR\n")
        f.write(f"# Classes source: {cfg.source_path}\n")
        f.write(f"path: {os.path.abspath(output_dir)}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/train\n")    # required by YOLO; external eval uses separate val set
        f.write(f"nc: {cfg.num_classes}\n")
        f.write(f"names: {cfg.class_names}\n")

    print(f"\nDone. {frame_idx} images, {total_labels} labels")
    print(f"Dataset: {output_dir}/")
    print(f"Classes: {cfg.num_classes} (from {cfg.source_path})")
    print(f"\nTrain command:")
    print(f"  yolo train data={yaml_path} model=yolov8n.pt epochs=50 imgsz={width} device=cpu")
    print(f"  (remove device=cpu if you have a CUDA GPU)")


# ---------------------------------------------------------------------------
# Instance segmentation → YOLO labels
# ---------------------------------------------------------------------------

def _extract_labels(event, cfg: ClassConfig) -> list[tuple[int, float, float, float, float]]:
    """Convert AI2-THOR instance seg → YOLO-format labels.

    AI2-THOR's instance_segmentation_frame gives each visible object a
    unique colour.  ``event.color_to_object_id`` maps colour → objectId,
    and ``event.get_object(objectId)`` gives us the object metadata.
    """
    if not hasattr(event, "instance_segmentation_frame"):
        return []
    seg = event.instance_segmentation_frame
    if seg is None:
        return []

    img_h, img_w = seg.shape[:2]

    # Build colour → label lookup (inverse of the colour→id mapping)
    colour_to_id = getattr(event, "color_to_object_id", {})
    if not colour_to_id:
        return []

    labels: list[tuple[int, float, float, float, float]] = []
    seen = set()

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

        # Find all pixels with this colour
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
    p = argparse.ArgumentParser(description="Auto-label YOLO training data from AI2-THOR")
    p.add_argument("--scenes", type=int, default=10,
                   help="Number of FloorPlan scenes (default: 10)")
    p.add_argument("--steps", type=int, default=200,
                   help="Random walk steps per scene (default: 200)")
    p.add_argument("--output", default="data",
                   help="Output directory (default: data/)")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--classes", default=None,
                   help="Path to classes.yaml (default: config/classes.yaml)")
    args = p.parse_args()
    collect(scenes=args.scenes, steps_per_scene=args.steps,
            output_dir=args.output, width=args.width, height=args.height,
            config_path=args.classes)


if __name__ == "__main__":
    main()
