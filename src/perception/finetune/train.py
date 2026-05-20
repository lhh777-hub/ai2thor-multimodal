"""
Fine-tune YOLOv8 on AI2-THOR auto-labeled data.

Usage::

    # Step 1: collect data
    python src/perception/finetune/collect.py --scenes 10 --steps 200

    # Step 2: train
    python src/perception/finetune/train.py --data data/data.yaml --epochs 50

    # Step 3: use the fine-tuned model
    python src/cli/perceive.py --model runs/train/weights/best.pt

Hardware notes:
- With CUDA GPU (recommended):         ~1-2 hours for 50 epochs on 10 scenes
- Without GPU (device=cpu):            much slower, reduce --epochs to 10-20
"""

from __future__ import annotations

import argparse
import os
import sys
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.common.logger import setup_logger

logger = setup_logger("train")


def train(data_yaml: str, base_model: str = "yolov8n.pt",
          epochs: int = 50, imgsz: int = 640, device: str = "cpu",
          batch: int = 8, lr0: float = 0.01, project: str = "runs",
          name: str = "train", resume: bool = False) -> str:
    """Fine-tune YOLOv8 and return the path to the best checkpoint.

    Set *resume=True* to continue from the last saved checkpoint.
    """
    from ultralytics import YOLO

    if resume:
        checkpoint = os.path.join(project, name, "weights", "last.pt")
        logger.info("Resuming from: %s", checkpoint)
        model = YOLO(checkpoint)
    else:
        logger.info("Loading base model: %s", base_model)
        model = YOLO(base_model)

    logger.info("Starting training (%d epochs, device=%s, batch=%d)", epochs, device, batch)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        batch=batch,
        lr0=lr0,
        project=project,
        name=name,
        exist_ok=True,
        resume=resume,
        verbose=True,
    )

    best = os.path.join(project, name, "weights", "best.pt")
    logger.info("Training done.  Best weights: %s", best)
    print(f"\n  Best model: {best}")
    print(f"  Use it:  python src/cli/perceive.py --model {best}")
    return best


def main():
    p = argparse.ArgumentParser(description="Fine-tune YOLOv8 on AI2-THOR data")
    p.add_argument("--data", default="data/data.yaml",
                   help="Path to data.yaml (default: data/data.yaml)")
    p.add_argument("--base-model", default="yolov8n.pt",
                   help="Pretrained YOLO checkpoint (default: yolov8n.pt)")
    p.add_argument("--epochs", type=int, default=50,
                   help="Training epochs (default: 50, use 10-20 if no GPU)")
    p.add_argument("--imgsz", type=int, default=640,
                   help="Image size (default: 640)")
    p.add_argument("--device", default="cpu",
                   help="Device: cpu, cuda:0, etc. (default: cpu)")
    p.add_argument("--batch", type=int, default=8,
                   help="Batch size (default: 8, reduce if OOM)")
    p.add_argument("--lr0", type=float, default=0.01,
                   help="Initial learning rate (default: 0.01)")
    p.add_argument("--project", default="runs",
                   help="Output project dir (default: runs/)")
    p.add_argument("--name", default="train",
                   help="Run name (default: train)")
    p.add_argument("--resume", action="store_true",
                   help="Resume from last checkpoint")
    args = p.parse_args()

    if not os.path.exists(args.data):
        print(f"  Error: {args.data} not found.")
        print(f"  Run 'python src/perception/finetune/collect.py' first.")
        sys.exit(1)

    train(data_yaml=args.data, base_model=args.base_model,
          epochs=args.epochs, imgsz=args.imgsz, device=args.device,
          batch=args.batch, lr0=args.lr0, project=args.project,
          name=args.name, resume=args.resume)


if __name__ == "__main__":
    main()
