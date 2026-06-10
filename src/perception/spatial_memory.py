"""Spatial memory — build object position index from YOLO + depth + pose.

During a room tour, the agent walks to multiple vantage points.  At each
position it does a 360° scan, runs YOLO, and computes the 3D world
position of each detected object using the depth frame and camera pose.

This is genuine vision-based spatial memory — no AI2-THOR metadata used.
Objects that YOLO can't detect simply won't be in the index, and tasks
targeting them will fail.  This is the right behaviour for a "视觉+语言"
course project.
"""

from __future__ import annotations

import numpy as np

from src.common.types import Vec3, Detection
from src.common.logger import setup_logger
from src.perception.geo_anchor import pixel_to_world

logger = setup_logger("spatial_memory")


class SpatialMemory:
    """YOLO label → 3D world position, built from vision only."""

    def __init__(self):
        self._index: dict[str, list[Vec3]] = {}  # label → [positions]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_scan(self, detections: list[Detection],
                    depth_frame: np.ndarray,
                    agent_x: float, agent_y: float, agent_z: float,
                    heading_deg: float, horizon_deg: float = 0.0,
                    min_conf: float = 0.3) -> int:
        """Ingest one 360° scan: compute 3D positions for each detection
        using depth + pose, store in the index.

        Returns number of new objects added.
        """
        if depth_frame is None:
            return 0

        h, w = depth_frame.shape[:2]
        added = 0

        for d in detections:
            if d.confidence < min_conf:
                continue
            cx = int(d.bbox.center_x)
            cy = int(d.bbox.center_y)
            cx = max(0, min(w - 1, cx))
            cy = max(0, min(h - 1, cy))
            depth_val = float(depth_frame[cy, cx])
            if depth_val <= 0 or depth_val > 20:
                continue

            wx, wy, wz = pixel_to_world(
                cx, cy, depth_val,
                width=w, height=h,
                agent_x=agent_x, agent_y=agent_y, agent_z=agent_z,
                heading_deg=heading_deg, horizon_deg=horizon_deg,
            )
            label = d.label.lower()
            if label not in self._index:
                self._index[label] = []
                added += 1
            self._index[label].append(Vec3(x=wx, y=wy, z=wz))

        return added

    def build_from_metadata(self, thor_object_map: dict) -> int:
        """Fallback: build from AI2-THOR metadata (for debugging only)."""
        self._index.clear()
        try:
            from src.perception.class_config import load_config
            cfg = load_config()
        except Exception:
            return 0
        for thor_type, info in thor_object_map.items():
            yolo = cfg.thor_to_names.get(thor_type, thor_type.lower())
            self._index[yolo.lower()] = [info["position"]]
        return len(self._index)

    def lookup(self, yolo_label: str) -> Vec3 | None:
        """Return the best remembered position, or None."""
        positions = self._index.get(yolo_label.lower())
        if positions:
            # Return the one closest to the centroid (most reliable)
            if len(positions) == 1:
                return positions[0]
            cx = sum(p.x for p in positions) / len(positions)
            cz = sum(p.z for p in positions) / len(positions)
            best = min(positions,
                       key=lambda p: (p.x - cx)**2 + (p.z - cz)**2)
            return best
        # Substring fallback
        tl = yolo_label.lower()
        for key, vals in self._index.items():
            if tl in key or key in tl:
                return vals[0]
        return None

    def __len__(self) -> int:
        return len(self._index)

    def __contains__(self, label: str) -> bool:
        return self.lookup(label) is not None

    def reset(self) -> None:
        self._index.clear()
