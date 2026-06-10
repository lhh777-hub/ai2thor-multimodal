"""Abstract base class for decision policies."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.common.types import Detection, AgentState
from src.decision.types import TaskSpec, ActionDecision


class DecisionPolicy(ABC):
    """Abstract policy: given observations and a task, decide the next action.

    Each call to ``decide()`` returns exactly one ``ActionDecision``.
    The policy is stateless with respect to the environment — all state
    comes in via parameters.  The decision loop owns the episode state.
    """

    @abstractmethod
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
        """Return the next action to execute.

        Parameters
        ----------
        rgb :
            H×W×3 uint8 array, the current camera frame.
        detections :
            Output of ``PerceptionPipeline.process()`` for this frame.
        agent_state :
            Current position, heading, horizon, and collision flag.
        task :
            Parsed ``TaskSpec`` from the user's NL instruction.
        step_count :
            How many steps have been taken so far (for timeout logic).
        depth_frame :
            H×W float32 depth map (metres), or None if depth is disabled.
        spatial_memory :
            Optional SpatialMemory with remembered object positions.

        Returns
        -------
        ActionDecision
        """
        ...

    def reset(self) -> None:
        """Reset any internal state.  Called between episodes."""
        pass
