"""
Frame collector — records RGB frames and manages per-session output directories.

Usage::

    collector = FrameCollector(scene="FloorPlan1")
    collector.record(result)          # one call per step
    collector.save_screenshot(result) # save a single screenshot
    collector.export_video()          # → <session_dir>/trajectory.mp4
    collector.export_frames()         # → <session_dir>/frames/

Each session gets its own timestamped directory::

    outputs/20260519_143022_FloorPlan1/
      frames/
        frame_0000.png
        frame_0001.png
        ...
      screenshots/
        shot_forward.png
        shot_turn_left.png
        ...
      trajectory.mp4
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import cv2
import numpy as np

from src.common.types import ActionResult


@dataclass
class FrameRecord:
    """One recorded frame with its context."""
    rgb: np.ndarray                          # H x W x 3, uint8, RGB order
    step: int
    action: str
    success: bool
    position: tuple[float, float, float]     # (x, y, z)
    heading_deg: float
    timestamp: float                         # time.time() when recorded

    def to_bgr(self) -> np.ndarray:
        return cv2.cvtColor(self.rgb, cv2.COLOR_RGB2BGR)


class FrameCollector:
    """Collects frames during a session.  Creates a per-session output directory.

    All frames are held in memory.  For long sessions call ``export_*``
    periodically and ``clear()``.
    """

    def __init__(self, scene: str = "unknown", root: str = "outputs"):
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(root, f"{ts}_{scene}")
        self._frames: list[FrameRecord] = []
        self._start_time: float | None = None
        self._shot_counter = 0
        os.makedirs(self.session_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, result: ActionResult) -> None:
        """Append the current frame to the in-memory buffer."""
        if self._start_time is None:
            self._start_time = time.time()
        p = result.agent_state.position
        self._frames.append(FrameRecord(
            rgb=result.sensor_data.rgb.copy(),
            step=result.step_count,
            action=result.action_name,
            success=result.success,
            position=(p.x, p.y, p.z),
            heading_deg=result.agent_state.heading_deg,
            timestamp=time.time(),
        ))

    def save_screenshot(self, result: ActionResult, tag: str = "") -> str:
        """Save a single screenshot (with overlay) into the session directory.

        Returns the file path.
        """
        self._shot_counter += 1
        os.makedirs(os.path.join(self.session_dir, "screenshots"), exist_ok=True)

        img = self._annotate(result.sensor_data.rgb.copy(), result)
        if tag:
            name = f"shot_{self._shot_counter:03d}_{tag}.png"
        else:
            name = f"shot_{self._shot_counter:03d}.png"
        path = os.path.join(self.session_dir, "screenshots", name)
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return path

    def clear(self) -> None:
        """Discard all buffered frames (does not delete exported files)."""
        self._frames.clear()
        self._start_time = None

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    @property
    def duration_seconds(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    def __len__(self) -> int:
        return len(self._frames)

    def __iter__(self):
        return iter(self._frames)

    def __getitem__(self, index: int) -> FrameRecord:
        return self._frames[index]

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_frames(self) -> str:
        """Save all buffered frames as numbered PNGs in ``<session_dir>/frames/``.

        Returns the output directory path.
        """
        out = os.path.join(self.session_dir, "frames")
        os.makedirs(out, exist_ok=True)
        for i, fr in enumerate(self._frames):
            img = self._annotate(fr.rgb.copy(), fr)
            cv2.imwrite(os.path.join(out, f"frame_{i:04d}.png"),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return out

    def export_video(self, fps: int = 5) -> str:
        """Export buffered frames as MP4 to ``<session_dir>/trajectory.mp4``.

        Returns the output file path.
        """
        if not self._frames:
            raise ValueError("No frames to export")

        path = os.path.join(self.session_dir, "trajectory.mp4")
        h, w = self._frames[0].rgb.shape[:2]
        writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        for fr in self._frames:
            img = self._annotate(fr.rgb.copy(), fr)
            writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        writer.release()
        return path

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _annotate(rgb: np.ndarray, src) -> np.ndarray:
        """Draw position / action overlay on a frame.

        *src* can be a ``FrameRecord`` or ``ActionResult``.
        """
        if isinstance(src, FrameRecord):
            px, py, pz = src.position
            step = src.step
            action = src.action
            ok = src.success
            heading = src.heading_deg
        else:
            p = src.agent_state.position
            px, py, pz = p.x, p.y, p.z
            step = src.step_count
            action = src.action_name
            ok = src.success
            heading = src.agent_state.heading_deg

        lines = [
            f"Step: {step}  Action: {action}  {'OK' if ok else 'FAIL'}",
            f"Pos: ({px:.2f}, {py:.2f}, {pz:.2f})  Head: {heading:.0f} deg",
        ]
        for i, line in enumerate(lines):
            y = 18 + i * 22
            cv2.putText(rgb, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.50, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(rgb, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.50, (0, 255, 0), 1, cv2.LINE_AA)
        return rgb
