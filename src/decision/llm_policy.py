"""LLMPolicy — Vision-Language Model (VLM) driven decision making.

Sends the robot's current RGB camera frame **and** a structured text report
(detection summary + agent state + task) to a multimodal VLM every step.
The VLM sees exactly what the robot sees and makes decisions based on the
visual input, not just YOLO text summaries.

Uses OpenAI Vision API format (image_url + text in a content array),
compatible with Qwen-VL (qwen-vl-max / qwen-vl-plus), GPT-4V, and any
OpenAI-compatible multimodal endpoint.

Requires ``OPENAI_API_KEY`` (and optionally ``OPENAI_BASE_URL``) set
in the environment.  Falls back to a safe scanning behaviour when no
API key is configured.

Usage::

    policy = LLMPolicy(model="qwen-vl-max")
    policy.reset()
    decision = policy.decide(
        rgb=frame,             # ← sent as base64 JPEG to VLM every step
        detections=pipeline_output,
        agent_state=state,
        task=task_spec,
        step_count=step,
    )
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass

import numpy as np

from src.common.types import Detection, AgentState
from src.common.logger import setup_logger
from src.common.utils import img_to_b64
from src.decision.base import DecisionPolicy
from src.decision.types import TaskSpec, ActionDecision

logger = setup_logger("llm_policy")

# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------

_ACTION_SPACE = frozenset({
    "MOVE_FORWARD", "MOVE_BACK",
    "TURN_LEFT", "TURN_RIGHT",
    "TURN_LEFT_SMALL", "TURN_RIGHT_SMALL",
    "LOOK_UP", "LOOK_DOWN",
    "INTERACT_OPEN", "INTERACT_PICKUP", "INTERACT_TOGGLE",
    "NAVIGATE_TO",   # delegate to A* pathfinding (target taken from task)
    "STOP",
})

_SYSTEM_PROMPT = """\
You are an embodied agent in a 3D indoor environment (AI2-THOR simulation).

Each step you receive:
- **A first-person camera image** — this is your primary source of truth.
  Look at it carefully. Identify what objects you see, their approximate
  positions (left/center/right), and distances.
- **A text report** — supplementary information with YOLO detections and
  agent state.  YOLO is fast but UNRELIABLE: labels often change at close
  range ("microwave"→"oven"→"cabinet", "cup"→"bottle", etc.).

CRITICAL: In your "reason" field, you MUST explicitly compare what you SEE
in the image against what YOLO reports.  Examples of good reasons:
  "I SEE microwave in image center (~2m) — YOLO correctly labels it."
  "I SEE microwave in image but YOLO says 'cabinet' — trusting my vision."
  "YOLO missed the chair entirely, but I SEE it on the right (~3m)."
  "Target is NOT visible in the image — turning left to scan."

Available actions:
  MOVE_FORWARD / MOVE_BACK      — step 0.25 m
  TURN_LEFT / TURN_RIGHT        — rotate 90 degrees
  TURN_LEFT_SMALL / TURN_RIGHT_SMALL — rotate 30 degrees
  LOOK_UP / LOOK_DOWN           — tilt camera
  INTERACT_OPEN                 — open a nearby object (door, fridge, cabinet)
  INTERACT_PICKUP               — pick up a nearby object
  INTERACT_TOGGLE               — toggle a nearby object (lamp, switch)
  NAVIGATE_TO                   — use A* pathfinding (only when target is NOT
                                  visible AT ALL in the image)
  STOP                          — task is complete

Rules:
- Navigation: approach target to ≤ 1.5 m, then STOP (done=true).
- Interaction: approach to ≤ 1.0 m, then INTERACT_* ONCE, then STOP (done=true).
- Target visible but far → turn to face it and MOVE_FORWARD.
- Target NOT visible → TURN_LEFT/TURN_RIGHT to scan. After 3-4 turns, NAVIGATE_TO.
- Colliding/blocked → NEVER MOVE_FORWARD, turn instead.
- >5 repeated same action → do something different.
- Small objects on counters/shelves → try LOOK_UP if lost.

Respond with a single JSON object:
  {"action": "<action>", "done": true/false, "confidence": 0.0-1.0, "reason": "I SEE ..."}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_user_message(task: TaskSpec, detections: list[Detection],
                        agent_state: AgentState, step_count: int,
                        last_action: str = "",
                        consecutive_same: int = 0) -> str:
    """Build the user message with detection summary and agent state."""
    if detections:
        lines = []
        for d in detections[:20]:
            dist_str = (f"{d.distance_meters:.2f} m"
                        if d.distance_meters else d.distance_level)
            lines.append(
                f"  - {d.label} (pos={d.screen_position}, dist={dist_str}, "
                f"conf={d.confidence:.2f})"
            )
        det_summary = "\n".join(lines)
    else:
        det_summary = "  (nothing detected)"

    stuck_note = ""
    if consecutive_same >= 3:
        stuck_note = (f"\nWARNING: You have repeated '{last_action}' "
                      f"{consecutive_same} times. You may be stuck. "
                      f"Do something different.")

    return (
        f"Task: {task.raw_text}\n"
        f"Task type: {task.task_type}\n"
        f"Target: {task.target}\n"
        f"Step count: {step_count}\n"
        f"\nDetected objects:\n{det_summary}\n"
        f"\nAgent state: heading={agent_state.heading_deg:.0f} deg, "
        f"colliding={agent_state.is_colliding}"
        f"{stuck_note}\n"
        f"\nDecide the next action. Respond with JSON only."
    )


def _extract_json(text: str) -> str | None:
    """Extract a JSON object string from LLM output.  Returns None on failure."""
    if not text or not text.strip():
        return None

    # Strategy 1: extract from ```json ... ``` code block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return m.group(1)

    # Strategy 2: find balanced braces
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    # Unclosed brace — try to complete it
    return text[start:] + "}"


def _parse_json_response(text: str) -> ActionDecision:
    """Extract a JSON decision from an LLM response.

    Tries: balanced-brace extraction → code-block extraction → completion
    → repair truncated strings.  Falls back to TURN_LEFT on any failure.
    """
    json_str = _extract_json(text)

    if json_str is None:
        logger.warning("LLM returned empty or unparseable response")
        return ActionDecision(
            action="TURN_LEFT", done=False, confidence=0.1,
            reason="[LLM fallback] Empty response.",
        )

    for attempt in range(2):
        try:
            data = json.loads(json_str)
            break
        except json.JSONDecodeError:
            if attempt == 0:
                # Repair truncated JSON from token-limited LLM output.
                stripped = json_str.rstrip()
                # If the response is just '{"' or similar — complete it as a
                # safe fallback action.
                if len(stripped) < 10:
                    json_str = ('{"action": "TURN_LEFT", "done": false, '
                                '"confidence": 0.3, "reason": "truncated"}')
                elif stripped[-1] not in ("}", "]", '"'):
                    if json_str.count('"') % 2 == 1:
                        json_str = json_str.rstrip() + '"'
                    if json_str.strip()[-1] != "}":
                        json_str = json_str.rstrip() + "}"
            else:
                logger.warning("Could not parse LLM JSON: %s", json_str[:200])
                return ActionDecision(
                    action="TURN_LEFT", done=False, confidence=0.1,
                    reason=f"[LLM fallback] Bad JSON: {json_str[:60]}",
                )

    action = str(data.get("action", "STOP"))
    # Accept NAVIGATE_TO:xxx (LLM sometimes appends the target name)
    base_action = action.split(":")[0] if ":" in action else action
    if base_action not in _ACTION_SPACE:
        logger.warning("Unknown action '%s' — defaulting to TURN_LEFT", action)
        action = "TURN_LEFT"

    done = bool(data.get("done", False))
    confidence = float(data.get("confidence", 0.5))
    confidence = max(0.0, min(1.0, confidence))
    reason = str(data.get("reason", ""))

    return ActionDecision(
        action=action, done=done, confidence=confidence, reason=reason,
    )


# ---------------------------------------------------------------------------
# LLMPolicy internal state
# ---------------------------------------------------------------------------

@dataclass
class _LLMPolicyState:
    last_action: str = ""
    last_good_action: str = ""   # most recent action from a non-empty response
    consecutive_same: int = 0
    consecutive_moves: int = 0


# ---------------------------------------------------------------------------
# LLMPolicy
# ---------------------------------------------------------------------------

class LLMPolicy(DecisionPolicy):
    """LLM-based decision policy using OpenAI Vision API format.

    Parameters
    ----------
    model :
        VLM model name.  Defaults to ``VLM_MODEL`` env var, then ``deepseek-v4-pro``.
    temperature :
        LLM sampling temperature (default 0.0 for deterministic).
    max_tokens :
        Max tokens in the LLM response (default 256).
    max_steps :
        Built-in step limit for safety (default 200).
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        max_steps: int = 200,
    ):
        self.model = model or os.environ.get("VLM_MODEL", "deepseek-v4-pro")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._max_steps = max_steps
        self._client = None  # lazy-init
        self._history: list[dict] = []  # full conversation history
        self._state = _LLMPolicyState()

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
        # --- Safety timeout ---
        if step_count >= self._max_steps:
            return ActionDecision(
                action="STOP", done=True, confidence=1.0,
                reason=f"Max steps ({self._max_steps}) reached.",
            )

        # --- Stuck detection: same action 5+ times in a row → force break ---
        if self._state.consecutive_same >= 5:
            self._state.consecutive_same = 0
            self._state.consecutive_moves = 0
            return ActionDecision(
                action="TURN_LEFT", done=False, confidence=0.5,
                reason="[LLM safety] Repeated action detected — forcing turn.",
            )

        # --- Stuck detection: collision 4+ times in a row → force break ---
        if agent_state.is_colliding and self._state.consecutive_moves >= 4:
            self._state.consecutive_moves = 0
            return ActionDecision(
                action="TURN_LEFT", done=False, confidence=0.5,
                reason="[LLM safety] Collision loop detected — forcing turn.",
            )

        # --- Run the VLM call ---
        client = self._get_client()
        if client is None:
            return ActionDecision(
                action="TURN_LEFT", done=False, confidence=0.3,
                reason="[LLM fallback] No API key configured.",
            )

        user_text = _build_user_message(
            task, detections, agent_state, step_count,
            last_action=self._state.last_action,
            consecutive_same=self._state.consecutive_same)

        # Build multimodal message: camera image + text report.
        # The VLM sees exactly what the robot sees — it can verify YOLO
        # labels visually and spot objects YOLO missed entirely.
        user_content: list[dict] = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_to_b64(rgb)}",
                    "detail": "auto",
                },
            },
            {
                "type": "text",
                "text": user_text,
            },
        ]

        # Stateless: only system prompt + current observation.
        # Conversation history adds noise for reactive decisions.
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        # --- Call VLM (with retries for empty responses and transient errors) ---
        logger.info("[step %d] Calling VLM (%s)...", step_count, self.model)
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    timeout=30.0,       # per-call timeout as well
                )
                text = (response.choices[0].message.content or "").strip()

                # Empty response — retry after a short wait
                if not text:
                    if attempt < 2:
                        logger.warning("[step %d] VLM empty response, retrying (%d/2)...",
                                       step_count, attempt + 1)
                        time.sleep(2.0)
                        continue
                    else:
                        logger.warning("[step %d] VLM empty after 3 attempts → fallback", step_count)
                        fallback_action = self._state.last_good_action or "TURN_LEFT"
                        self._track_action(fallback_action)
                        return ActionDecision(
                            action=fallback_action, done=False, confidence=0.1,
                            reason="[LLM fallback] Empty response after retries.",
                        )

                # Store in history
                self._history.append({
                    "role": "user",
                    "content": f"[step {step_count}] {user_text[:200]}...",
                })
                self._history.append({"role": "assistant", "content": text})
                if len(self._history) > 10:
                    self._history = self._history[-6:]

                decision = _parse_json_response(text)

                # Map NAVIGATE_TO to include target from task
                if decision.action == "NAVIGATE_TO":
                    decision.action = f"NAVIGATE_TO:{task.target}"
                    decision.reason = f"[LLM] A* navigating to '{task.target}'"

                decision.reason = f"[LLM] {decision.reason}"

                # Track state
                self._track_action(decision.action)
                self._state.last_good_action = decision.action.split(":")[0]

                logger.info("[step %d] VLM → %s (conf=%.2f)", step_count,
                            decision.action, decision.confidence)
                return decision

            except Exception as e:
                logger.warning("[step %d] VLM call attempt %d failed: %s",
                               step_count, attempt + 1, e)
                if attempt < 2:
                    time.sleep(1.0)
                else:
                    fallback_action = self._state.last_good_action or "TURN_LEFT"
                    self._track_action(fallback_action)
                    return ActionDecision(
                        action=fallback_action, done=False, confidence=0.1,
                        reason=f"[LLM error] Failed after 3 attempts: {e}",
                    )

        # Unreachable
        return ActionDecision(
            action=self._state.last_good_action or "TURN_LEFT",
            done=False, confidence=0.1,
            reason="[LLM error] Unexpected code path.",
        )

    def reset(self) -> None:
        """Reset conversation history and internal state between episodes."""
        self._history.clear()
        self._state = _LLMPolicyState()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _track_action(self, action: str) -> None:
        """Update internal state for stuck / loop detection."""
        base = action.split(":")[0]  # strip NAVIGATE_TO:xxx prefix
        if base == self._state.last_action:
            self._state.consecutive_same += 1
        else:
            self._state.last_action = base
            self._state.consecutive_same = 1

        if base == "MOVE_FORWARD":
            self._state.consecutive_moves += 1
        else:
            self._state.consecutive_moves = 0

    def _get_client(self):
        """Lazy-init the OpenAI-compatible client."""
        if self._client is not None:
            return self._client
        # Auto-load .env if not already loaded (idempotent)
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        api_key = os.environ.get("OPENAI_API_KEY", "")
        base_url = os.environ.get("OPENAI_BASE_URL", "")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set — LLM policy unavailable.")
            return None
        try:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=api_key,
                base_url=base_url or None,
                timeout=30.0,       # per-request timeout (connect + read)
                max_retries=0,      # we handle retries ourselves
            )
            logger.info("LLM client ready: model=%s base_url=%s timeout=30s",
                        self.model, base_url or "(default)")
            return self._client
        except Exception as e:
            logger.warning("Could not create OpenAI client: %s", e)
            return None
