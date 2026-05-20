"""
AI2-THOR controller with explicit lifecycle, A* pathfinding, and navigation.

Design principles:
- Context-manager based (with statement) for guaranteed cleanup.
- No lazy initialisation — the controller is ready after __init__.
- Actions return ActionResult (agent state + sensor data).
- A* pathfinding on reachable positions for deterministic navigation.
- Query methods do NOT mutate step state.
"""

from __future__ import annotations

import heapq
import math
from typing import Any, Iterator

import ai2thor.controller
import numpy as np

from src.common.types import AgentState, ActionResult, SceneInfo, SensorData, Vec3
from src.common.logger import setup_logger

logger = setup_logger("controller")

# ---------------------------------------------------------------------------
# Action mapping
# ---------------------------------------------------------------------------

_ACTION_MAP: dict[str, str] = {
    "MOVE_FORWARD":  "MoveAhead",
    "MOVE_BACK":     "MoveBack",
    "TURN_LEFT":     "RotateLeft",
    "TURN_RIGHT":    "RotateRight",
    "LOOK_UP":       "LookUp",
    "LOOK_DOWN":     "LookDown",
}

_SMALL_TURN_DEG = 30
_GRID_SIZE = 0.25      # AI2-THOR default step size (metres)
_A_STAR_MAX_ITER = 5000


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class ThorController:
    """Encapsulates an AI2-THOR simulation session.

    Usage::

        with ThorController(width=600, height=400) as ctrl:
            ctrl.load_scene("FloorPlan1")
            for result in ctrl.navigate_to(target_pos):
                save_frame(result.sensor_data.rgb)   # generator, one frame per step
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __init__(
        self,
        *,
        width: int = 600,
        height: int = 400,
        field_of_view: float = 90.0,
        visibility_distance: float = 1.5,
        render_depth: bool = False,
        render_instance_seg: bool = False,
    ):
        self._width = width
        self._height = height
        self._fov = field_of_view

        self._ctrl = ai2thor.controller.Controller(
            width=width,
            height=height,
            fieldOfView=field_of_view,
            visibilityDistance=visibility_distance,
            renderDepthImage=render_depth,
            renderInstanceSegmentation=render_instance_seg,
        )
        self._step_count = 0
        self._scene_name: str = ""
        logger.info("ThorController initialised (%dx%d, FOV=%.0f)", width, height, field_of_view)

    def close(self) -> None:
        """Stop the simulation and release resources.  Idempotent."""
        if self._ctrl is not None:
            self._ctrl.stop()
            self._ctrl = None
            logger.info("ThorController closed")

    def __enter__(self) -> "ThorController":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Scene management
    # ------------------------------------------------------------------

    def load_scene(self, scene_name: str, seed: int | None = None) -> SceneInfo:
        """Load (or reset to) *scene_name*.  Optional random seed for spawn."""
        self._step_count = 0
        self._scene_name = scene_name
        self._ctrl.reset(scene_name)
        if seed is not None:
            self._random_spawn(seed)
        return self._build_scene_info()

    @property
    def scene_name(self) -> str:
        return self._scene_name

    @property
    def step_count(self) -> int:
        return self._step_count

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    def step(self, action: str) -> ActionResult:
        """Execute *action* and return the result.

        A ``Pass`` is issued after every action to obtain a frame that was
        actually rendered at the post-action position.  AI2-THOR's
        ``event.cv2img`` is the *pre*-action frame; the *post*-action frame
        only becomes available after the next controller step.
        """
        self._step_count += 1
        action_event = self._dispatch(action)
        success = action_event.metadata.get("lastActionSuccess", True)

        # Always step once more so the returned image matches the agent state.
        # ``Pass`` does not move the agent — it only triggers a fresh render.
        post_event = self._ctrl.step(action="Pass")

        return ActionResult(
            success=bool(success),
            action_name=action,
            agent_state=AgentState.from_thor(action_event.metadata, action_success=bool(success)),
            sensor_data=self._extract_sensor_data(post_event),
            scene_name=action_event.metadata.get("sceneName", self._scene_name),
            step_count=self._step_count,
            raw_event=action_event,
        )

    def get_current_view(self) -> ActionResult:
        """Return the current sensor reading without executing an action."""
        event = self._ctrl.last_event
        return ActionResult(
            success=True,
            action_name="(observe)",
            agent_state=AgentState.from_thor(event.metadata),
            sensor_data=self._extract_sensor_data(event),
            scene_name=event.metadata.get("sceneName", self._scene_name),
            step_count=self._step_count,
            raw_event=event,
        )

    @property
    def agent_state(self) -> AgentState:
        return AgentState.from_thor(self._ctrl.last_event.metadata)

    # ------------------------------------------------------------------
    # A* Pathfinding & Navigation
    # ------------------------------------------------------------------

    def plan_path(self, target: Vec3) -> list[Vec3] | None:
        """A* on the 0.25 m reachable-positions grid.  Returns waypoints or None."""
        positions = self._get_reachable_positions_raw()
        if not positions:
            return None

        walkable = {self._to_grid(p) for p in positions}
        start = self._to_grid(self.agent_state.position)
        goal = self._to_grid(target)

        if start not in walkable:
            logger.warning("Agent position not in reachable set — may be mid-teleport")
            return None

        # A*
        heap: list[tuple[float, int, tuple, tuple | None]] = []
        heapq.heappush(heap, (self._heuristic(start, goal), 0, start, None))
        g_score: dict[tuple, int] = {start: 0}
        came_from: dict[tuple, tuple] = {}

        iters = 0
        while heap and iters < _A_STAR_MAX_ITER:
            iters += 1
            _, g, current, _ = heapq.heappop(heap)
            if current == goal:
                return self._reconstruct_path(came_from, current)

            for nxt in self._neighbours(current, walkable):
                tentative = g + 1
                if tentative < g_score.get(nxt, 999999):
                    g_score[nxt] = tentative
                    came_from[nxt] = current
                    heapq.heappush(heap, (tentative + self._heuristic(nxt, goal), tentative, nxt, None))

        logger.info("A*: no path found after %d iterations", iters)
        return None

    def navigate_to(self, target: Vec3) -> Iterator[ActionResult]:
        """Navigate to *target* using A*, yielding one ActionResult per step.

        The agent rotates toward each waypoint, then moves forward.
        Caller can observe/record every frame.
        """
        path = self.plan_path(target)
        if path is None:
            logger.warning("navigate_to: no path to %s", target)
            return

        logger.info("navigate_to: %d waypoints", len(path))

        for waypoint in path:
            # 1) rotate to face the waypoint
            for result in self._rotate_toward(waypoint):
                yield result
            # 2) step forward
            yield self.step("MOVE_FORWARD")

    def navigate_to_object(self, object_type: str) -> Iterator[ActionResult]:
        """Convenience: find *object_type* in the scene, then navigate to it."""
        obj_map = self.get_object_map()
        info = obj_map.get(object_type)
        if info is None:
            # case-insensitive fallback
            for k, v in obj_map.items():
                if k.lower() == object_type.lower():
                    info = v
                    break
        if info is None:
            logger.warning("navigate_to_object: '%s' not found in scene", object_type)
            return
        yield from self.navigate_to(info["position"])

    def look_at(self, target: Vec3) -> Iterator[ActionResult]:
        """Rotate the agent to face *target* (generator, for frame-by-frame)."""
        yield from self._rotate_toward(target)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def distance_to(self, object_type: str) -> float | None:
        """Euclidean (XZ) distance to the closest object matching *object_type*."""
        ax, az = self.agent_state.position.x, self.agent_state.position.z
        target_lower = object_type.lower()
        best: float | None = None
        for obj in self._ctrl.last_event.metadata["objects"]:
            ot = obj["objectType"].lower()
            if ot == target_lower or target_lower in ot or ot in target_lower:
                d = np.sqrt((ax - obj["position"]["x"]) ** 2 + (az - obj["position"]["z"]) ** 2)
                if best is None or d < best:
                    best = float(d)
        return best

    def get_object_map(self) -> dict[str, dict]:
        """Return ``{ObjectType: {position, visible, object_id}}`` for all objects."""
        result: dict[str, dict] = {}
        for obj in self._ctrl.last_event.metadata["objects"]:
            ot = obj["objectType"]
            if ot not in result:
                result[ot] = {
                    "position": Vec3.from_dict(obj["position"]),
                    "visible": obj.get("visible", False),
                    "object_id": obj["objectId"],
                }
        return result

    # ------------------------------------------------------------------
    # Internal — A* helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_grid(v: Vec3) -> tuple:
        return (round(v.x / _GRID_SIZE), round(v.z / _GRID_SIZE))

    @staticmethod
    def _heuristic(a: tuple, b: tuple) -> float:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @staticmethod
    def _neighbours(node: tuple, walkable: set) -> list[tuple]:
        results = []
        for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (node[0] + dx, node[1] + dz)
            if n in walkable:
                results.append(n)
        return results

    def _reconstruct_path(self, came_from: dict, current: tuple) -> list[Vec3]:
        path: list[tuple] = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return [Vec3(x=g[0] * _GRID_SIZE, y=0.0, z=g[1] * _GRID_SIZE) for g in path]

    def _get_reachable_positions_raw(self) -> list[Vec3]:
        event = self._ctrl.step(action="GetReachablePositions")
        return [Vec3.from_dict(p) for p in event.metadata["actionReturn"]]

    def _rotate_toward(self, target: Vec3) -> Iterator[ActionResult]:
        """Generator: rotate agent to face *target*, yielding each ActionResult."""
        state = self.agent_state
        dx = target.x - state.position.x
        dz = target.z - state.position.z
        desired = math.degrees(math.atan2(dx, dz)) % 360

        current = state.heading_deg % 360
        # shortest rotation direction
        diff = (desired - current + 540) % 360 - 180  # [-180, 180]
        n_rotations = int(round(abs(diff) / 90))

        action = "TURN_LEFT" if diff < 0 else "TURN_RIGHT"
        for _ in range(n_rotations):
            yield self.step(action)

    # ------------------------------------------------------------------
    # Internal — action dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, action: str) -> Any:
        if action in _ACTION_MAP:
            return self._ctrl.step(action=_ACTION_MAP[action])
        if action == "TURN_LEFT_SMALL":
            return self._ctrl.step(action="RotateLeft", degrees=_SMALL_TURN_DEG)
        if action == "TURN_RIGHT_SMALL":
            return self._ctrl.step(action="RotateRight", degrees=_SMALL_TURN_DEG)
        if action.startswith("INTERACT_"):
            return self._dispatch_interact(action)
        logger.warning("Unknown action: %s", action)
        return self._ctrl.last_event

    def _dispatch_interact(self, action: str) -> Any:
        verb = action.removeprefix("INTERACT_")
        if verb == "OPEN":
            return self._ctrl.step(action="OpenObject", objectId=self._find_visible_with("openable"))
        if verb == "PICKUP":
            return self._ctrl.step(action="PickupObject", objectId=self._find_visible_with("pickupable"))
        if verb == "TOGGLE":
            obj_id = self._find_visible_with("toggleable")
            for obj in self._ctrl.last_event.metadata["objects"]:
                if obj["objectId"] == obj_id:
                    toggle = "ToggleObjectOff" if obj.get("isOn") else "ToggleObjectOn"
                    return self._ctrl.step(action=toggle, objectId=obj_id)
        return self._ctrl.last_event

    def _find_visible_with(self, prop: str) -> str:
        for obj in self._ctrl.last_event.metadata["objects"]:
            if obj.get("visible") and obj.get(prop):
                return obj["objectId"]
        return ""

    def _random_spawn(self, seed: int) -> None:
        self._ctrl.step(action="InitialRandomSpawn", randomSeed=seed)
        a = self._ctrl.last_event.metadata["agent"]
        self._ctrl.step(
            action="TeleportFull",
            x=a["position"]["x"], y=a["position"]["y"], z=a["position"]["z"],
            rotation=a["rotation"]["y"], horizon=0.0, standing=True,
        )

    def _extract_sensor_data(self, event: Any) -> SensorData:
        return SensorData(
            rgb=event.cv2img,
            depth=event.depth_frame if hasattr(event, "depth_frame") else None,
        )

    def _build_scene_info(self) -> SceneInfo:
        event = self._ctrl.last_event
        return SceneInfo(
            name=event.metadata["sceneName"],
            agent_state=AgentState.from_thor(event.metadata),
            objects=event.metadata["objects"],
        )
