"""
Core data types for the embodied agent system.

All types are plain dataclasses — no logic, no dependencies.
"""

from dataclasses import dataclass, field
import numpy as np


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

@dataclass
class Vec3:
    """A point or vector in 3D world space."""
    x: float
    y: float
    z: float

    @classmethod
    def from_dict(cls, d: dict) -> "Vec3":
        return cls(x=float(d["x"]), y=float(d["y"]), z=float(d["z"]))

    def distance_to(self, other: "Vec3") -> float:
        return float(np.sqrt((self.x - other.x) ** 2 + (self.z - other.z) ** 2))

    def __iter__(self):
        return iter((self.x, self.y, self.z))


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------

@dataclass
class AgentState:
    """Full agent pose at a single timestep."""
    position: Vec3
    heading_deg: float    # 0-360, 0 = north
    horizon_deg: float    # vertical camera angle
    is_colliding: bool

    @classmethod
    def from_thor(cls, metadata: dict, *, action_success: bool = True) -> "AgentState":
        a = metadata["agent"]
        # AI2-THOR sets agent.isColliding rarely.  The reliable signal is
        # lastActionSuccess=False in the event metadata.
        collided = bool(a.get("isColliding", False)) or not action_success
        return cls(
            position=Vec3.from_dict(a["position"]),
            heading_deg=float(a["rotation"]["y"]),
            horizon_deg=float(a["rotation"].get("horizon", 0.0)),
            is_colliding=collided,
        )


# ---------------------------------------------------------------------------
# Sensor data
# ---------------------------------------------------------------------------

@dataclass
class SensorData:
    """Raw sensor readings from a single frame.

    All images are in RGB colour order.
    """
    rgb: np.ndarray                     # H x W x 3, uint8
    depth: np.ndarray | None = None     # H x W, float32 (optional)
    width: int = 0
    height: int = 0

    def __post_init__(self):
        if self.width == 0:
            self.height, self.width = self.rgb.shape[:2]


# ---------------------------------------------------------------------------
# Perception
# ---------------------------------------------------------------------------

@dataclass
class BBox:
    """Axis-aligned bounding box in pixel coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2


@dataclass
class Detection:
    """One detected object in an RGB frame."""
    label: str                    # YOLO class name, e.g. "chair", "tv"
    bbox: BBox                    # pixel coordinates
    confidence: float             # YOLO confidence [0, 1]
    screen_position: str          # "left" | "center" | "right" (derived from bbox center)
    distance_level: str = ""      # "NEAR" | "MEDIUM" | "FAR" (filled by depth estimator)
    distance_meters: float = 0.0  # estimated distance in metres (0 = unknown)
    clip_score: float = 0.0       # CLIP similarity score [0, 1] (filled by verifier)


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

@dataclass
class SceneInfo:
    """Metadata about the currently loaded scene."""
    name: str
    agent_state: AgentState
    objects: list[dict]                     # raw AI2-THOR object list
    reachable_positions: list[Vec3] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

@dataclass
class ActionResult:
    """The result of executing one action in the environment."""
    success: bool
    action_name: str
    agent_state: AgentState
    sensor_data: SensorData
    scene_name: str
    step_count: int
    raw_event: object = field(repr=False)   # AI2-THOR Event (for advanced use)
