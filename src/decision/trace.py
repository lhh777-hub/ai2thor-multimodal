"""Episode trace recording — per-step data for trajectory visualization.

Records agent state, decisions, and detections at each step so that
trajectories can be visualised after evaluation completes.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np


@dataclass
class StepRecord:
    """One step of an episode."""
    step: int
    x: float               # agent world X
    y: float               # agent world Y (height)
    z: float               # agent world Z
    heading_deg: float
    action: str            # action dispatched this step
    reason: str            # decision reason (truncated to 120 chars)
    is_colliding: bool
    target_detected: bool
    target_distance: float  # -1 if unknown
    top_detections: str    # top 5 labels + distances, comma-separated


@dataclass
class EpisodeTrace:
    """Full trace of one evaluation episode."""
    scene: str
    policy: str
    task: str
    target: str
    task_type: str           # "navigation" | "interaction"
    success: bool
    final_reason: str
    total_steps: int
    elapsed_s: float
    optimal_steps: float
    spl: float
    steps: list[StepRecord] = field(default_factory=list)
    spawn_x: float = 0.0
    spawn_z: float = 0.0
    target_x: float | None = None   # from spatial memory / AI2-THOR metadata
    target_z: float | None = None
    reachable_positions: list[tuple[float, float]] = field(default_factory=list)

    def add_step(self, step: int, x: float, y: float, z: float,
                 heading_deg: float, action: str, reason: str,
                 is_colliding: bool, target_detected: bool,
                 target_distance: float, top_detections: str) -> None:
        self.steps.append(StepRecord(
            step=step, x=x, y=y, z=z, heading_deg=heading_deg,
            action=action, reason=reason[:120],
            is_colliding=is_colliding,
            target_detected=target_detected,
            target_distance=round(target_distance, 2) if target_distance > 0 else -1.0,
            top_detections=top_detections,
        ))

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict (numpy arrays excluded)."""
        return {
            "scene": self.scene,
            "policy": self.policy,
            "task": self.task,
            "target": self.target,
            "task_type": self.task_type,
            "success": self.success,
            "final_reason": self.final_reason,
            "total_steps": self.total_steps,
            "elapsed_s": self.elapsed_s,
            "optimal_steps": self.optimal_steps,
            "spl": self.spl,
            "spawn_x": self.spawn_x,
            "spawn_z": self.spawn_z,
            "target_x": self.target_x,
            "target_z": self.target_z,
            "reachable_positions": self.reachable_positions,
            "steps": [
                {
                    "step": s.step,
                    "x": s.x, "y": s.y, "z": s.z,
                    "heading_deg": s.heading_deg,
                    "action": s.action,
                    "reason": s.reason,
                    "is_colliding": s.is_colliding,
                    "target_detected": s.target_detected,
                    "target_distance": s.target_distance,
                    "top_detections": s.top_detections,
                }
                for s in self.steps
            ],
        }

    def save_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @staticmethod
    def load_json(path: str) -> "EpisodeTrace":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        trace = EpisodeTrace(
            scene=d["scene"], policy=d["policy"],
            task=d["task"], target=d["target"],
            task_type=d["task_type"],
            success=d["success"], final_reason=d["final_reason"],
            total_steps=d["total_steps"], elapsed_s=d["elapsed_s"],
            optimal_steps=d.get("optimal_steps", 0),
            spl=d.get("spl", 0),
            spawn_x=d.get("spawn_x", 0), spawn_z=d.get("spawn_z", 0),
            target_x=d.get("target_x"), target_z=d.get("target_z"),
            reachable_positions=d.get("reachable_positions", []),
        )
        for s in d["steps"]:
            trace.add_step(
                step=s["step"], x=s["x"], y=s["y"], z=s["z"],
                heading_deg=s["heading_deg"], action=s["action"],
                reason=s["reason"], is_colliding=s["is_colliding"],
                target_detected=s["target_detected"],
                target_distance=s["target_distance"],
                top_detections=s["top_detections"],
            )
        return trace
