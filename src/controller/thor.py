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
        self._last_action_ok: bool = True  # tracked separately because Pass overwrites it
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
        self._last_action_ok = bool(success)

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
            agent_state=AgentState.from_thor(event.metadata,
                                             action_success=self._last_action_ok),
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

        # If the exact goal is not on the walkable grid (object on a table/shelf),
        # snap to the nearest walkable neighbour.
        original_goal = goal
        if goal not in walkable:
            goal = self._snap_to_walkable(goal, walkable)
            if goal is None:
                logger.info("A*: target %s has no nearby walkable neighbour", original_goal)
                return None
            logger.debug("A*: snapped goal %s → %s (nearest walkable)", original_goal, goal)

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

        logger.info("A*: no path found after %d iterations (start=%s → goal=%s)",
                     iters, start, goal)
        return None

    @staticmethod
    def _snap_to_walkable(goal: tuple, walkable: set) -> tuple | None:
        """Find the nearest walkable grid point to *goal* by Manhattan distance.

        Uses a generous search radius (8 grid units = 2.0 m) because objects
        on counters/shelves may be well above the walkable floor grid.
        """
        best: tuple | None = None
        best_dist = 999999
        for w in walkable:
            d = abs(w[0] - goal[0]) + abs(w[1] - goal[1])
            if d < best_dist:
                best_dist = d
                best = w
        # Accept if within 2.0 m — counter-top objects can be ~1 m from floor
        return best if best_dist <= 8 else None

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
        """Convenience: find *object_type* in the scene, then navigate to a
        walkable position ~1.0 m in front of it (from the agent's current side).

        This avoids navigating to the object's exact coordinates, which may be
        on top of a counter (y > 0) with no nearby walkable floor.
        """
        obj_map = self.get_object_map()
        info = obj_map.get(object_type)
        if info is None:
            target_norm = object_type.lower().replace(" ", "")
            for k, v in obj_map.items():
                k_norm = k.lower().replace(" ", "")
                if k.lower() == object_type.lower() or k_norm == target_norm:
                    info = v
                    break
        # Reverse lookup via classes.yaml (e.g. "oven" → "StoveBurner")
        if info is None:
            info = self._reverse_lookup(object_type, obj_map)
        if info is None:
            logger.warning("navigate_to_object: '%s' not found in scene", object_type)
            return

        obj_pos = info["position"]  # Vec3
        # Navigate to a point 1.0 m from the object toward the agent —
        # this places the agent in front of the counter, not behind it.
        ax, az = self.agent_state.position.x, self.agent_state.position.z
        dx = ax - obj_pos.x
        dz = az - obj_pos.z
        dist = math.sqrt(dx * dx + dz * dz)
        if dist > 0.01:
            approach_x = obj_pos.x + dx / dist * 1.0
            approach_z = obj_pos.z + dz / dist * 1.0
        else:
            approach_x = obj_pos.x
            approach_z = obj_pos.z
        yield from self.navigate_to(Vec3(x=approach_x, y=obj_pos.y, z=approach_z))

    @staticmethod
    def _reverse_lookup(yolo_name: str, obj_map: dict) -> dict | None:
        """Use classes.yaml to map YOLO label → AI2-THOR objectType."""
        try:
            from src.perception.class_config import load_config
            cfg = load_config()
            for thor_type, yn in cfg.thor_to_names.items():
                if yn.lower() == yolo_name.lower() and thor_type in obj_map:
                    return obj_map[thor_type]
        except Exception:
            pass
        return None

    def look_at(self, target: Vec3) -> Iterator[ActionResult]:
        """Rotate the agent to face *target* (generator, for frame-by-frame)."""
        yield from self._rotate_toward(target)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def distance_to(self, object_type: str) -> float | None:
        """Euclidean (XZ) distance to the closest object matching *object_type*.

        Matches both directly against AI2-THOR ``objectType`` AND via the
        THOR→YOLO mapping in ``config/classes.yaml``, so queries like
        ``"refrigerator"`` correctly find ``Fridge`` objects.
        """
        ax, az = self.agent_state.position.x, self.agent_state.position.z
        target_lower = object_type.lower()
        target_norm = target_lower.replace(" ", "")

        # Lazy-load THOR→YOLO reverse mapping (YOLO name → list of THOR names)
        if not hasattr(self, "_yolo_to_thor"):
            self._yolo_to_thor: dict[str, list[str]] = {}
            try:
                from src.perception.class_config import load_config
                cfg = load_config()
                for thor_name, yolo_name in cfg.thor_to_names.items():
                    yn = yolo_name.lower()
                    if yn not in self._yolo_to_thor:
                        self._yolo_to_thor[yn] = []
                    self._yolo_to_thor[yn].append(thor_name.lower())
            except Exception:
                pass

        best: float | None = None
        for obj in self._ctrl.last_event.metadata["objects"]:
            ot = obj["objectType"].lower()
            ot_norm = ot.replace(" ", "")

            # Direct match (existing logic)
            match = (ot == target_lower or ot_norm == target_norm
                     or target_lower in ot or ot in target_lower
                     or target_norm in ot_norm or ot_norm in target_norm)

            # THOR→YOLO mapping match: query is a YOLO name, find its THOR names
            if not match and self._yolo_to_thor:
                thor_names = self._yolo_to_thor.get(target_lower, [])
                if ot in thor_names or ot_norm in thor_names:
                    match = True

            if match:
                d = np.sqrt((ax - obj["position"]["x"]) ** 2
                            + (az - obj["position"]["z"]) ** 2)
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
        positions = event.metadata.get("actionReturn")
        if not positions:
            return []
        return [Vec3.from_dict(p) for p in positions]

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
        """Return the objectId of the best visible object with *prop*.

        Prefers objects whose AI2-THOR type maps to *_interact_target*
        (set by the decision loop before INTERACT_* actions).  Falls back
        to the nearest visible object.
        """
        candidates = []
        ax, az = self.agent_state.position.x, self.agent_state.position.z
        for obj in self._ctrl.last_event.metadata["objects"]:
            if obj.get("visible") and obj.get(prop):
                ox, oz = obj["position"]["x"], obj["position"]["z"]
                d = (ax - ox) ** 2 + (az - oz) ** 2
                candidates.append((d, obj))

        if not candidates:
            return ""

        # Prefer the object matching the interact target (via class config)
        target = getattr(self, "_interact_target", "")
        if target:
            try:
                from src.perception.class_config import load_config
                cfg = load_config()
                for d, obj in sorted(candidates):
                    thor_type = obj["objectType"]
                    yolo_name = cfg.thor_to_names.get(thor_type, "").lower()
                    if yolo_name and (yolo_name == target.lower()
                                      or target.lower() in yolo_name
                                      or yolo_name in target.lower()):
                        return obj["objectId"]
            except Exception:
                pass

        # Fallback: nearest
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]["objectId"]

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
