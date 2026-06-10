"""RulePolicy — deterministic, rule-based decision making.

Does NOT require any API key or GPU beyond what YOLO already uses.

Algorithm (per ``decide()`` call):

1. Fuzzy-match *task.target* in *detections* (exact → substring → no match).
2. **Target found** → perception-guided approach:
   a. If near (≤ 1.5 m):
      - Navigation → **STOP** (done)
      - Interaction → **INTERACT_OPEN/PICKUP/TOGGLE** → **STOP** (done)
   b. If far: turn toward / move forward based on ``screen_position``.
3. **Target not found** → scan in place (up to 4 turns), then emit
   ``NAVIGATE_TO:<target>`` to delegate A* navigation.
4. **Stuck detection**: 3+ consecutive forward moves with collision → rotate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.common.types import Detection, AgentState
from src.common.logger import setup_logger
from src.decision.base import DecisionPolicy
from src.decision.types import TaskSpec, ActionDecision
from src.perception.geo_anchor import GeometricAnchor, pixel_to_world

logger = setup_logger("rule_policy")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DISTANCE_NEAR = 1.0          # metres — agent must be this close to declare success
_DISTANCE_STOP = 1.0          # metres — AI2-THOR interaction requires close proximity
_SCAN_ATTEMPTS = 4            # full turns (90 deg) before delegating to A*
_MAX_STEPS = 200              # built-in safety timeout

# Common user→YOLO label aliases (user may say "fridge", YOLO detects "refrigerator")
_ALIASES: dict[str, str] = {
    "fridge":       "refrigerator",
    "tv":           "tv",
    "television":   "tv",
    "couch":        "couch",
    "sofa":         "couch",
    "plant":        "potted plant",
    "table":        "dining table",
    "desk":         "dining table",
    "cup":          "cup",
    "mug":          "cup",
    "lamp":         "tv",           # AI2-THOR lamps are often detected as tv by YOLO
    "light":        "tv",
    "counter":      "countertop",
}


# ---------------------------------------------------------------------------
# Internal state (reset per episode)
# ---------------------------------------------------------------------------

@dataclass
class _RulePolicyState:
    scan_counter: int = 0
    scan_direction: str = ""          # "left" | "right"
    consecutive_moves: int = 0        # for stuck detection
    consecutive_blocks: int = 0       # consecutive blocked steps (far target)
    sidestep_phase: int = 0           # 0=idle, 1..N=active sidestep
    sidestep_direction: str = ""      # "right" | "left" — alternates on retry
    sidestep_retries: int = 0         # how many times sidestep has been retried
    was_near_target: str = ""         # target name we were recently following
    was_near_distance: float = 0.0    # last known distance when following
    consecutive_turns: int = 0        # for turn-loop detection
    last_turn_dir: str = ""           # "left" | "right"
    stuck_unblock_step: int = 0       # close-range unblock sequence step
    label_history: dict = None        # label → [(conf, distance), ...] sliding window
    geo_anchor: object = None         # GeometricAnchor instance

    def __post_init__(self):
        if self.label_history is None:
            self.label_history = {}
        if self.geo_anchor is None:
            self.geo_anchor = GeometricAnchor(distance_threshold=0.3)


# ---------------------------------------------------------------------------
# RulePolicy
# ---------------------------------------------------------------------------

class RulePolicy(DecisionPolicy):
    """Deterministic, rule-based decision policy.

    Usage::

        policy = RulePolicy()
        policy.reset()                     # between episodes
        decision = policy.decide(
            rgb=frame,
            detections=pipeline_output,
            agent_state=state,
            task=task_spec,
            step_count=step,
        )
    """

    def __init__(self, *, near_distance: float = _DISTANCE_NEAR,
                 max_steps: int = _MAX_STEPS, scan_attempts: int = _SCAN_ATTEMPTS,
                 use_vlm_recovery: bool = False):
        self._near = near_distance
        self._max_steps = max_steps
        self._scan_attempts = scan_attempts
        self._use_vlm = use_vlm_recovery
        self._state = _RulePolicyState()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decide(
        self,
        *,
        rgb: np.ndarray,
        detections: list[Detection],
        agent_state: AgentState,
        task: TaskSpec,
        step_count: int = 0,
        depth_frame: np.ndarray | None = None,
        spatial_memory: object | None = None,
    ) -> ActionDecision:
        target = task.target

        # --- Geometric anchoring: inject 3D world positions into detections ---
        if depth_frame is not None:
            self._inject_world_positions(detections, depth_frame, agent_state)

        # --- Geometric anchor: lock target by 3D position ---
        anchor = self._state.geo_anchor
        if not anchor.is_locked:
            anchor.try_lock(target, detections, min_conf=0.7, min_distance=1.0)
        else:
            n = anchor.remap_by_proximity(target, detections)
            if n > 0:
                logger.debug("GeoAnchor: remapped %d detection(s) to '%s'",
                             n, target)
            # Only auto-unlock when NO remaps are active AND we're very close.
            # If remaps ARE active, the anchor is working correctly — don't
            # destroy it just because we're close (that's the whole point).
            if n == 0 and anchor.closest_distance < 0.5:
                anchor.unlock()

        # Update temporal label history
        self._update_label_history(detections)

        # --- Safety timeout ---
        if step_count >= self._max_steps:
            return ActionDecision(
                action="STOP", done=True, confidence=1.0,
                reason=f"Max steps ({self._max_steps}) reached.",
            )

        # --- Sidestep in progress?  Complete the sequence uninterrupted ---
        if self._state.sidestep_phase > 0:
            return self._sidestep_continue(agent_state.is_colliding)

        # --- Step 1: try to find target in detections ---
        matched = self._find_target(target, detections)

        if matched is not None:
            return self._perception_guided(matched, agent_state, task)

        # --- Step 3: not found → check memory → scan → fallback to A* ---
        return self._handle_not_found(target, detections, spatial_memory)

    def reset(self) -> None:
        """Reset internal state between episodes."""
        self._state = _RulePolicyState()

    # ------------------------------------------------------------------
    # Geometric anchoring
    # ------------------------------------------------------------------

    def _inject_world_positions(
        self,
        detections: list[Detection],
        depth_frame: np.ndarray,
        agent_state: AgentState,
    ) -> None:
        """Compute 3D world position for each detection's bbox centre and attach
        it as ``_world_xyz`` on the Detection object."""
        if depth_frame is None:
            return
        h, w = depth_frame.shape[:2]
        for d in detections:
            cx = int(d.bbox.center_x)
            cy = int(d.bbox.center_y)
            # Clamp to valid depth image bounds
            cx = max(0, min(w - 1, cx))
            cy = max(0, min(h - 1, cy))
            depth_val = float(depth_frame[cy, cx])
            if depth_val <= 0 or depth_val > 20:
                continue  # invalid depth
            d._world_xyz = pixel_to_world(
                cx, cy, depth_val,
                width=w, height=h,
                agent_x=agent_state.position.x,
                agent_y=agent_state.position.y,
                agent_z=agent_state.position.z,
                heading_deg=agent_state.heading_deg,
                horizon_deg=agent_state.horizon_deg,
            )

    # ------------------------------------------------------------------
    # Perception-guided
    # ------------------------------------------------------------------

    def _perception_guided(self, det: Detection, agent_state: AgentState,
                           task: TaskSpec) -> ActionDecision:
        distance = self._detection_distance(det)
        pos = det.screen_position
        self._state.scan_counter = 0

        # Track that we were following this target (for not-found recovery).
        # Use a generous threshold — the target may disappear due to FOV /
        # elevation before reaching NEAR range.
        self._state.was_near_target = task.target
        self._state.was_near_distance = distance

        # --- Navigation: must be within 1.5 m AND facing the target ---
        if task.is_navigation and distance <= _DISTANCE_NEAR:
            self._state.consecutive_moves = 0
            if pos != "center":
                turn = "TURN_LEFT_SMALL" if pos == "left" else "TURN_RIGHT_SMALL"
                return ActionDecision(
                    action=turn, done=False, confidence=0.9,
                    reason=f"Near '{det.label}' ({distance:.2f} m), turning to face.",
                )
            return ActionDecision(
                action="STOP", done=True, confidence=1.0,
                reason=f"Arrived at '{det.label}' ({distance:.2f} m).",
            )

        # --- Interaction: close enough for interaction, execute ---
        if task.is_interaction and distance <= _DISTANCE_STOP:
            self._state.consecutive_moves = 0
            return ActionDecision(
                action=f"INTERACT_{task.interact_action}",
                done=True, confidence=1.0,
                reason=f"At '{det.label}' ({distance:.2f} m), executing {task.interact_action}.",
            )

        # --- Close but not facing: turn to face first ---
        if distance <= _DISTANCE_NEAR and pos != "center":
            self._state.consecutive_moves = 0
            turn = "TURN_LEFT_SMALL" if pos == "left" else "TURN_RIGHT_SMALL"
            return ActionDecision(
                action=turn, done=False, confidence=0.9,
                reason=f"Close to '{det.label}' ({distance:.2f} m), turning to face.",
            )

        # --- Close and facing but not at stop distance ---
        if distance <= _DISTANCE_NEAR and pos == "center":
            self._state.consecutive_moves += 1
            if self._state.consecutive_moves >= 3 and agent_state.is_colliding:
                self._state.consecutive_moves = 0
                # Multi-step unblock: turn → back up → re-approach
                step = self._state.stuck_unblock_step
                self._state.stuck_unblock_step += 1
                if task.is_interaction and step == 0:
                    # First try: INTERACT from here — AI2-THOR may still reach
                    # across a counter or through a shelf edge.
                    self._state.stuck_unblock_step = 0
                    return ActionDecision(
                        action=f"INTERACT_{task.interact_action}",
                        done=True, confidence=0.7,
                        reason=f"Blocked near '{det.label}' ({distance:.2f} m), "
                               f"attempting {task.interact_action}.",
                    )
                if step == 0:
                    # Turn 90° left — find a new approach angle
                    return ActionDecision(
                        action="TURN_LEFT", done=False, confidence=0.6,
                        reason=f"Stuck near '{det.label}' — turning to find "
                               f"clear approach.",
                    )
                elif step == 1:
                    # Step back to get perspective / widen FOV
                    return ActionDecision(
                        action="MOVE_BACK", done=False, confidence=0.6,
                        reason=f"Stuck near '{det.label}' — backing up to "
                               f"regain perspective.",
                    )
                else:
                    # Re-approach from the new angle
                    self._state.stuck_unblock_step = 0
                    return ActionDecision(
                        action="MOVE_FORWARD", done=False, confidence=0.7,
                        reason=f"Re-approaching '{det.label}' from new angle.",
                    )
            return ActionDecision(
                action="MOVE_FORWARD", done=False, confidence=0.9,
                reason=f"Closing in on '{det.label}' ({distance:.2f} m).",
            )

        # --- Far away: approach (sidestep circumvention) ---
        if pos == "center":
            self._state.consecutive_moves += 1
            self._state.consecutive_turns = 0
            if not agent_state.is_colliding:
                self._state.consecutive_blocks = 0
                self._state.sidestep_phase = 0

            # SIDESTEP when blocked: straight approach blocked by a
            # counter/island/table.  First try right, then left on retry.
            if agent_state.is_colliding:
                self._state.consecutive_blocks += 1
                if self._state.consecutive_blocks >= 3 and distance > _DISTANCE_NEAR:
                    n_blocks = self._state.consecutive_blocks
                    self._state.consecutive_blocks = 0
                    self._state.consecutive_moves = 0
                    self._state.sidestep_phase = 1
                    # Alternate direction on retry
                    if self._state.sidestep_retries == 0:
                        self._state.sidestep_direction = "right"
                    else:
                        self._state.sidestep_direction = (
                            "left" if self._state.sidestep_direction == "right"
                            else "right")
                    turn = ("TURN_RIGHT" if self._state.sidestep_direction == "right"
                            else "TURN_LEFT")
                    return ActionDecision(
                        action=turn, done=False, confidence=0.8,
                        reason=f"Blocked {n_blocks}x at {distance:.1f}m "
                               f"— sidestepping {self._state.sidestep_direction}.",
                    )

            return ActionDecision(
                action="MOVE_FORWARD", done=False, confidence=0.85,
                reason=f"Approaching '{det.label}' (center, {distance:.2f} m).",
            )

        self._state.consecutive_moves = 0
        turn = "TURN_LEFT_SMALL" if pos == "left" else "TURN_RIGHT_SMALL"
        self._state.consecutive_turns += 1

        # Guard: many consecutive turns (any direction) without centering
        # means the object is too far/small to align to.  Break out by
        # moving forward to change perspective.
        if self._state.consecutive_turns >= 4:
            self._state.consecutive_turns = 0
            return ActionDecision(
                action="MOVE_FORWARD", done=False, confidence=0.6,
                reason=f"Turn-loop break: advancing to shift view of '{det.label}'.",
            )

        return ActionDecision(
            action=turn, done=False, confidence=0.85,
            reason=f"Turning toward '{det.label}' ({pos}, {distance:.1f} m).",
        )

    # ------------------------------------------------------------------
    # Not-found handling
    # ------------------------------------------------------------------

    def _navigation_failed(self, target: str) -> ActionDecision:
        """Called when A* to a remembered position fails (e.g. blocked path).
        If VLM is enabled, hand off for visual guidance.
        """
        if self._use_vlm:
            return ActionDecision(
                action="VLM_EXPLORE", done=False, confidence=0.7,
                reason=f"A* to '{target}' failed — handing to VLM for "
                       f"visual exploration.",
            )
        return ActionDecision(
            action="STOP", done=True, confidence=0.2,
            reason=f"A* to '{target}' failed — giving up.",
        )

    def _handle_not_found(self, target: str, detections: list[Detection] | None = None,
                           spatial_memory: object | None = None) -> ActionDecision:
        # --- Spatial memory: primary navigation source (quick win) ---
        if spatial_memory is not None:
            mem_pos = spatial_memory.lookup(target)
            if mem_pos is not None:
                self._state.scan_counter = 0
                # Pass the remembered XZ position to A* (vision-based, not metadata)
                return ActionDecision(
                    action=f"NAVIGATE_TO:{target}:{mem_pos.x:.2f}:{mem_pos.z:.2f}",
                    done=False, confidence=0.9,
                    reason=f"Memory: '{target}' at ({mem_pos.x:.1f}, {mem_pos.z:.1f})"
                           f" — navigating directly.",
                )
            # Not in vision-based spatial memory — don't give up immediately.
            # Fall through: try was_near_target recovery, then a 360° scan.
            # Small objects (cup, spoon, etc.) are often missed during the
            # room tour but detectable at close range after a few turns.
            # After scan exhausts: VLM → VLM_EXPLORE, no-VLM → STOP.

        # --- was_near_target recovery ---
        if self._state.was_near_target == target:
            # We were tracking this target and it suddenly vanished.
            if self._state.was_near_distance <= _DISTANCE_STOP:
                self._state.was_near_target = ""
                return ActionDecision(
                    action="STOP", done=True, confidence=0.9,
                    reason=f"Was at {self._state.was_near_distance:.2f}m from "
                           f"'{target}' before detection changed — stopping.",
                )

            # Close-range recovery: target was recently within 1.5 m
            # (e.g. just handed back by VLM).  Try quick small turns to
            # re-acquire before falling back to scan/VLM.
            if self._state.was_near_distance <= _DISTANCE_NEAR:
                recovery = self._state.consecutive_moves
                self._state.consecutive_moves += 1
                if recovery == 0:
                    return ActionDecision(
                        action="TURN_LEFT_SMALL", done=False, confidence=0.7,
                        reason=f"Close-range loss of '{target}' "
                               f"({self._state.was_near_distance:.1f}m) — "
                               f"small turn left to re-acquire.",
                    )
                if recovery == 1:
                    return ActionDecision(
                        action="TURN_RIGHT_SMALL", done=False, confidence=0.7,
                        reason=f"Close-range loss of '{target}' — "
                               f"small turn right to re-acquire.",
                    )
                if recovery == 2:
                    return ActionDecision(
                        action="TURN_RIGHT_SMALL", done=False, confidence=0.6,
                        reason=f"Close-range loss of '{target}' — "
                               f"widening search.",
                    )
                # Close-range recovery failed — fall through to scan
                self._state.was_near_target = ""
                self._state.consecutive_moves = 0
                logger.info("Close-range recovery failed for '%s' — "
                            "falling back to scan", target)

            else:
                # Far-range loss: if nearby objects suggest a label swap,
                # skip recovery and go straight to scan.
                if detections:
                    nearby_any = any(
                        d.distance_meters > 0 and d.distance_meters < 2.0
                        for d in detections)
                    if nearby_any:
                        self._state.was_near_target = ""
                        self._state.consecutive_moves = 0
                        logger.info("Target '%s' lost but nearby objects present — "
                                    "deferring to scan", target)

                # Run recovery steps BEFORE considering VLM — LOOK_UP and
                # MOVE_BACK are cheap and often re-acquire the target.
                recovery = self._state.consecutive_moves
                self._state.consecutive_moves += 1
                if recovery == 0:
                    return ActionDecision(
                        action="LOOK_UP", done=False, confidence=0.6,
                        reason=f"Lost '{target}' — looking up (may be elevated).",
                    )
                if recovery == 1:
                    return ActionDecision(
                        action="MOVE_BACK", done=False, confidence=0.6,
                        reason=f"Lost '{target}' — backing up to widen view.",
                    )
                # Recovery failed — fall through to scan
                self._state.was_near_target = ""
                self._state.consecutive_moves = 0
                logger.info("Recovery failed for '%s' — falling back to scan", target)

        # --- Systematic scan ---
        # Only reached when: (a) spatial_memory not provided, or
        # (b) use_vlm=True and target not in spatial_memory, or
        # (c) was_near_target recovery exhausted.
        self._state.scan_counter += 1
        if self._state.scan_counter > self._scan_attempts:
            self._state.scan_counter = 0
            # Scan exhausted — VLM recovery as last resort
            if self._use_vlm:
                return ActionDecision(
                    action="VLM_EXPLORE", done=False, confidence=0.7,
                    reason=f"'{target}' not found after full scan — "
                           f"handing to VLM for visual search.",
                )
            # No VLM: try metadata-based fallback (only when spatial_memory
            # was not provided — i.e. interactive/demo mode).
            return ActionDecision(
                action=f"NAVIGATE_TO:{target}",
                done=False, confidence=0.5,
                reason=f"'{target}' not detected after full scan — "
                       f"delegating to A* (metadata).",
            )

        return ActionDecision(
            action="TURN_LEFT", done=False, confidence=0.7,
            reason=f"Scanning for '{target}' ({self._state.scan_counter}/{self._scan_attempts}).",
        )

    # ------------------------------------------------------------------
    # Sidestep circumvention
    # ------------------------------------------------------------------

    def _sidestep_continue(self, colliding: bool) -> ActionDecision:
        """Bidirectional obstacle circumvention.

        Phase 1-3: turn to look along the obstacle edge.
        Phase 4-5: move forward along the edge.
        Phase 6: turn back toward target + clear state.
        If colliding during move phases, abort this side and try the other.

        Max 2 retries (right → left → give up).
        """
        MAX_RETRIES = 2
        PHASES_PER_SIDE = 6

        if colliding:
            # Collision during sidestep — this side is blocked.
            # Abort and try the other direction.
            if self._state.sidestep_retries < MAX_RETRIES:
                self._state.sidestep_retries += 1
                old_dir = self._state.sidestep_direction
                self._state.sidestep_direction = (
                    "left" if old_dir == "right" else "right")
                self._state.sidestep_phase = 0
                self._state.consecutive_blocks = 0
                self._state.was_near_target = ""
                turn = ("TURN_LEFT" if self._state.sidestep_direction == "left"
                        else "TURN_RIGHT")
                return ActionDecision(
                    action=turn, done=False, confidence=0.7,
                    reason=f"Sidestep {old_dir} blocked "
                           f"— trying {self._state.sidestep_direction} side.",
                )
            # Both sides exhausted — give up
            self._state.sidestep_phase = 0
            self._state.sidestep_retries = 0
            self._state.sidestep_direction = ""
            self._state.consecutive_blocks = 0
            self._state.was_near_target = ""
            return ActionDecision(
                action="TURN_LEFT", done=False, confidence=0.3,
                reason="Sidestep exhausted both sides — falling back.",
            )

        # Not colliding — proceed through phases
        self._state.sidestep_phase += 1
        turn = ("TURN_RIGHT" if self._state.sidestep_direction == "right"
                else "TURN_LEFT")
        turn_back = ("TURN_LEFT" if self._state.sidestep_direction == "right"
                     else "TURN_RIGHT")

        if self._state.sidestep_phase <= 3:
            # Initial turns to face along the obstacle edge
            return ActionDecision(
                action=turn, done=False, confidence=0.8,
                reason=f"Sidestep {self._state.sidestep_direction} "
                       f"{self._state.sidestep_phase}/{PHASES_PER_SIDE}: "
                       f"turning along edge.",
            )
        elif self._state.sidestep_phase <= 5:
            # Move forward along the cleared edge
            return ActionDecision(
                action="MOVE_FORWARD", done=False, confidence=0.8,
                reason=f"Sidestep {self._state.sidestep_direction} "
                       f"{self._state.sidestep_phase}/{PHASES_PER_SIDE}: "
                       f"stepping along edge.",
            )
        else:
            # Done — turn back to re-acquire target
            self._state.sidestep_phase = 0
            self._state.sidestep_retries = 0
            self._state.sidestep_direction = ""
            self._state.consecutive_blocks = 0
            self._state.consecutive_moves = 0
            self._state.was_near_target = ""  # skip LOOK_UP/MOVE_BACK recovery
            return ActionDecision(
                action=turn_back, done=False, confidence=0.8,
                reason="Sidestep complete — turning back to re-acquire target.",
            )

    # ------------------------------------------------------------------
    # Temporal label filtering
    # ------------------------------------------------------------------

    def _update_label_history(self, detections: list[Detection]) -> None:
        """Record each detection's label+confidence to a sliding window."""
        for d in detections:
            key = d.label.lower()
            if key not in self._state.label_history:
                self._state.label_history[key] = []
            self._state.label_history[key].append(
                (d.confidence, d.distance_meters if d.distance_meters > 0 else 999))
        # Trim to last 12 frames per label
        for key in list(self._state.label_history):
            if len(self._state.label_history[key]) > 12:
                self._state.label_history[key] = \
                    self._state.label_history[key][-12:]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detection_distance(det: Detection) -> float:
        """Best-effort distance from a Detection."""
        if det.distance_meters > 0:
            return det.distance_meters
        # Fallback: map level to rough midpoint
        return {"NEAR": 0.75, "MEDIUM": 2.25, "FAR": 5.0}.get(
            det.distance_level, 2.0)

    def _find_target(self, target: str, detections: list[Detection]) -> Detection | None:
        """Fuzzy-match *target* in *detections*, with temporal label smoothing.

        1. Direct label match (exact / alias / substring)
        2. Temporal fallback: if the target label appeared frequently in recent
           frames with high confidence, but the current frame labels it differently,
           look for a detection at a similar distance — YOLO probably switched labels.
        """
        t = target.lower()
        alias = _ALIASES.get(t, t)

        # 1. Direct label match
        candidates: list[Detection] = []
        for d in detections:
            dl = d.label.lower()
            if dl == t or dl == alias:
                return d
            if t in dl or dl in t or alias in dl or dl in alias:
                candidates.append(d)
        if candidates:
            return max(candidates, key=lambda d: d.confidence)

        # 2. Temporal fallback: did the target appear recently with high confidence?
        hist = self._state.label_history.get(t, [])
        if not hist:
            hist = []
            for key, entries in self._state.label_history.items():
                if t in key or key in t:
                    hist.extend(entries)
        if hist:
            avg_conf = sum(c for c, _ in hist) / len(hist)
            if avg_conf >= 0.4:
                # Target was seen recently. Look for detection at similar distance.
                avg_dist = sum(d for _, d in hist if d < 999) / max(
                    sum(1 for _, d in hist if d < 999), 1)
                for d in detections:
                    dd = d.distance_meters if d.distance_meters > 0 else 999
                    if dd < 999 and avg_dist < 999:
                        ratio = max(dd, avg_dist) / max(min(dd, avg_dist), 0.01)
                        if ratio <= 1.6:
                            logger.debug("Temporal: '%s' (avg_conf=%.2f, avg_dist=%.1f) "
                                         "→ using '%s' (%.1fm, conf=%.2f)",
                                         target, avg_conf, avg_dist,
                                         d.label, dd, d.confidence)
                            d.label = target  # override to target label
                            return d

        return None
