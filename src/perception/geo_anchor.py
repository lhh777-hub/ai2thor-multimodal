"""Geometric anchor — locks target identity by 3D world position.

Problem: YOLO labels change at close range (microwave→oven→cabinet),
breaking the tracking loop.

Solution: When a target is first detected with high confidence at a distance,
compute its 3D world position from the depth frame + camera pose.  During
approach, any detection whose 3D position is within a threshold of the anchor
is remapped to the anchor label — regardless of what YOLO says.

This exploits the fact that physical objects don't move; only YOLO's
perception of them changes.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.common.logger import setup_logger

logger = setup_logger("geo_anchor")

# ---------------------------------------------------------------------------
# Camera intrinsics (AI2-THOR default: FOV=90°, same as ThorController)
# ---------------------------------------------------------------------------

# Default resolution used by the eval / decide demos
_DEFAULT_WIDTH = 640
_DEFAULT_HEIGHT = 480
_DEFAULT_FOV_DEG = 90.0


def _focal_px(width: int = _DEFAULT_WIDTH,
              fov_deg: float = _DEFAULT_FOV_DEG) -> float:
    """Focal length in pixels for a pinhole camera."""
    return (width / 2.0) / math.tan(math.radians(fov_deg / 2.0))


def pixel_to_world(
    px: float,
    py: float,
    depth_m: float,
    *,
    width: int = _DEFAULT_WIDTH,
    height: int = _DEFAULT_HEIGHT,
    agent_x: float = 0.0,
    agent_y: float = 0.0,
    agent_z: float = 0.0,
    heading_deg: float = 0.0,
    horizon_deg: float = 0.0,
) -> tuple[float, float, float]:
    """Convert a pixel (px, py) at *depth_m* metres to a 3D world coordinate.

    Parameters
    ----------
    px, py :
        Pixel coordinates in the RGB image (origin top-left).
    depth_m :
        Depth value at that pixel, in metres (from AI2-THOR depth frame).
    agent_x, agent_y, agent_z :
        Agent position in world coordinates.
    heading_deg :
        Agent yaw (rotation around Y axis, 0 = +Z, 90 = +X).
    horizon_deg :
        Agent pitch (0 = horizontal, positive = looking up).
    """
    fp = _focal_px(width)

    # Camera-space coordinates (X right, Y down, Z forward)
    cx = (px - width / 2.0) * depth_m / fp
    cy = -(py - height / 2.0) * depth_m / fp   # flip Y
    cz = depth_m

    # Apply pitch (horizon) rotation around camera X axis
    h_rad = math.radians(horizon_deg)
    cy2 = cy * math.cos(h_rad) - cz * math.sin(h_rad)
    cz2 = cy * math.sin(h_rad) + cz * math.cos(h_rad)

    # Apply yaw (heading) rotation around world Y axis
    y_rad = math.radians(heading_deg)
    wx = cx * math.cos(y_rad) + cz2 * math.sin(y_rad)
    wz = -cx * math.sin(y_rad) + cz2 * math.cos(y_rad)
    wy = cy2

    return (agent_x + wx, agent_y + wy, agent_z + wz)


# ---------------------------------------------------------------------------
# GeometricAnchor
# ---------------------------------------------------------------------------

class GeometricAnchor:
    """Tracks a target object by its 3D world position during approach.

    Usage (inside a policy's ``decide()`` or ``_find_target()``)::

        anchor = GeometricAnchor(distance_threshold=0.3)

        for each step:
            for det in detections:
                world_xyz = pixel_to_world(...)
                det._world_xyz = world_xyz

            # Try to lock on
            if not anchor.is_locked:
                anchor.try_lock(target_label, detections, min_conf=0.7)
            else:
                anchor.remap_by_proximity(target_label, detections)

            # Auto-unlock when very close
            if anchor.is_locked and anchor.closest_distance < 0.5:
                anchor.unlock()
    """

    def __init__(self, distance_threshold: float = 0.3):
        self._label: str = ""
        self._world_pos: tuple[float, float, float] | None = None
        self._threshold: float = distance_threshold
        self._locked_at_distance: float = 999.0
        self.closest_distance: float = 999.0
        self._unlock_cooldown: int = 0      # frames to wait before re-lock

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_locked(self) -> bool:
        return self._world_pos is not None

    @property
    def label(self) -> str:
        return self._label

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def try_lock(
        self,
        target_label: str,
        detections: list[Any],
        *,
        min_conf: float = 0.7,
        min_distance: float = 1.0,
    ) -> bool:
        """Attempt to lock onto *target_label* using a high-confidence detection.

        Only locks when the detection is far enough away (> *min_distance*)
        for reliable depth, confidence exceeds *min_conf*, and the computed
        3D position passes sanity checks.
        """
        if self._unlock_cooldown > 0:
            self._unlock_cooldown -= 1
            return False

        best = None
        best_dist = 999.0
        tl = target_label.lower()

        for d in detections:
            if not hasattr(d, '_world_xyz') or d._world_xyz is None:
                continue
            if d.label.lower() != tl and tl not in d.label.lower():
                continue
            if d.confidence < min_conf:
                continue
            dist = getattr(d, 'distance_meters', 0) or 0
            if dist < min_distance:
                continue

            # Sanity check: 3D position should be roughly consistent with
            # the estimated distance.  If depth says 20m but YOLO says 1.3m,
            # the depth is looking through a window or hitting a wall.
            wx, wy, wz = d._world_xyz
            world_dist = math.sqrt(wx**2 + wy**2 + wz**2) if hasattr(d, '_world_xyz') else 999
            _ = world_dist  # reserved for future sanity check

            if dist < best_dist:
                best = d
                best_dist = dist

        if best is not None and hasattr(best, '_world_xyz') and best._world_xyz is not None:
            # Final sanity: reject anchors where 3D position is extreme
            wx, wy, wz = best._world_xyz
            if abs(wy) > 20 or abs(wx) > 50 or abs(wz) > 50:
                logger.debug("GeoAnchor: rejected extreme 3D=(%.1f, %.1f, %.1f) for '%s'",
                             wx, wy, wz, target_label)
                return False

            self._label = target_label
            self._world_pos = best._world_xyz
            self._locked_at_distance = best_dist
            self.closest_distance = best_dist
            self._unlock_cooldown = 0
            logger.info("GeoAnchor LOCKED: '%s' at 3D=(%.2f, %.2f, %.2f)  dist=%.2f m",
                        target_label, *self._world_pos, best_dist)
            return True
        return False

    def remap_by_proximity(
        self,
        target_label: str,
        detections: list[Any],
    ) -> int:
        """For each detection, if its 3D position is within threshold of the
        anchor, remap its label to *target_label*.  Returns count of remapped
        detections.

        Also tracks the closest remapped detection distance for auto-unlock.
        """
        if not self.is_locked:
            return 0

        ax, ay, az = self._world_pos
        remapped = 0
        self.closest_distance = 999.0

        for d in detections:
            if not hasattr(d, '_world_xyz') or d._world_xyz is None:
                continue
            dx, dy, dz = d._world_xyz
            dist = math.sqrt((dx - ax) ** 2 + (dy - ay) ** 2 + (dz - az) ** 2)

            if dist <= self._threshold:
                old_label = d.label
                d.label = target_label
                d.confidence = max(d.confidence, 0.85)
                remapped += 1
                # Track distance of the REMAPPED detection for auto-unlock
                det_dist = getattr(d, 'distance_meters', 0) or 0
                if det_dist > 0 and det_dist < self.closest_distance:
                    self.closest_distance = det_dist
                logger.debug("GeoAnchor remap: '%s' → '%s' (3D dist=%.2f m)",
                             old_label, target_label, dist)

        return remapped

    def unlock(self) -> None:
        """Release the anchor and set a cooldown to prevent immediate re-lock."""
        if self.is_locked:
            logger.info("GeoAnchor UNLOCKED: '%s' (cooldown=3 frames)", self._label)
        self._label = ""
        self._world_pos = None
        self._locked_at_distance = 999.0
        self.closest_distance = 999.0
        self._unlock_cooldown = 3       # wait 3 frames before re-locking

    def reset(self) -> None:
        """Full reset between episodes."""
        self._label = ""
        self._world_pos = None
        self._locked_at_distance = 999.0
        self.closest_distance = 999.0
        self._unlock_cooldown = 0
