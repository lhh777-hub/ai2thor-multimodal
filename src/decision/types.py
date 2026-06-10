"""Data types for the decision module.

TaskSpec   — parsed NL instruction
ActionDecision — single-step decision from a policy
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TaskSpec:
    """Parsed natural-language instruction.

    Produced by ``TaskParser.interpret()`` and consumed by ``DecisionPolicy.decide()``.
    """

    task_type: str          # "navigation" | "interaction"
    target: str             # object name, e.g. "chair", "fridge"
    interact_action: str    # "OPEN" | "PICKUP" | "TOGGLE" | "" (empty for navigation)
    raw_text: str           # original NL instruction, preserved for LLM policy

    @property
    def is_navigation(self) -> bool:
        return self.task_type == "navigation"

    @property
    def is_interaction(self) -> bool:
        return self.task_type == "interaction"


@dataclass
class ActionDecision:
    """Single-step decision from a ``DecisionPolicy``.

    The decision loop reads ``action`` and dispatches to ``ThorController.step()``,
    or handles the special ``NAVIGATE_TO:<name>`` signal for A* fallback.
    """

    action: str      # "MOVE_FORWARD" | "TURN_LEFT" | ... | "INTERACT_OPEN" | ... | "STOP"
                     # Special: "NAVIGATE_TO:<object_name>" triggers A* navigation fallback
    done: bool       # True → task complete, loop terminates
    confidence: float  # [0, 1], policy's confidence in this decision
    reason: str      # human-readable explanation
