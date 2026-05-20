"""
Scene priors — weight adjustment for objects based on room type.

AI2-THOR scene names follow patterns like FloorPlan1-30.  Different floor
plans represent different room types (kitchen, bedroom, bathroom, living room).
Objects that are common in a given room get a boost; improbable objects get a
penalty.

Usage::

    prior = ScenePrior("FloorPlan1")
    boosted = prior.apply(detections)
"""

from __future__ import annotations

from src.common.types import Detection

# ---------------------------------------------------------------------------
# Room → {object_label: weight multiplier}
# weight > 1.0 = boost,  weight < 1.0 = penalise,  weight = 1.0 = neutral
# ---------------------------------------------------------------------------

_ROOM_PRIORS: dict[str, dict[str, float]] = {
    "kitchen": {
        "refrigerator": 1.3, "microwave": 1.3, "oven": 1.3,
        "sink": 1.3, "toaster": 1.2, "cup": 1.2, "bowl": 1.2,
        "bottle": 1.1, "dining table": 1.1, "chair": 1.1,
        "bed": 0.5, "toilet": 0.3, "tv": 0.8,
    },
    "bedroom": {
        "bed": 1.3, "tv": 1.1, "laptop": 1.1, "cell phone": 1.1,
        "book": 1.1, "chair": 1.1,
        "refrigerator": 0.3, "oven": 0.3, "toilet": 0.2, "sink": 0.5,
    },
    "bathroom": {
        "toilet": 1.3, "sink": 1.3, "bottle": 1.1,
        "bed": 0.3, "refrigerator": 0.2, "oven": 0.2, "tv": 0.6,
    },
    "living room": {
        "couch": 1.3, "sofa": 1.3, "tv": 1.2, "chair": 1.2,
        "potted plant": 1.1, "book": 1.1, "laptop": 1.1,
        "dining table": 1.1,
        "toilet": 0.2, "bed": 0.4, "oven": 0.4,
    },
}

# AI2-THOR FloorPlan → room type (manually labelled, covers FloorPlan1-30)
_FLOORPLAN_ROOM: dict[str, str] = {
    "FloorPlan1": "kitchen",
    "FloorPlan2": "living room",
    "FloorPlan3": "bedroom",
    "FloorPlan4": "bathroom",
    "FloorPlan5": "living room",
    "FloorPlan6": "bedroom",
    "FloorPlan7": "kitchen",
    "FloorPlan8": "living room",
    "FloorPlan9": "kitchen",
    "FloorPlan10": "bedroom",
    "FloorPlan11": "living room",
    "FloorPlan12": "kitchen",
    "FloorPlan13": "bedroom",
    "FloorPlan14": "bathroom",
    "FloorPlan15": "living room",
    "FloorPlan16": "kitchen",
    "FloorPlan17": "bedroom",
    "FloorPlan18": "living room",
    "FloorPlan19": "kitchen",
    "FloorPlan20": "bathroom",
    "FloorPlan21": "living room",
    "FloorPlan22": "kitchen",
    "FloorPlan23": "bedroom",
    "FloorPlan24": "living room",
    "FloorPlan25": "bathroom",
    "FloorPlan26": "kitchen",
    "FloorPlan27": "bedroom",
    "FloorPlan28": "living room",
    "FloorPlan29": "kitchen",
    "FloorPlan30": "bedroom",
}


class ScenePrior:
    """Adjusts detection confidence with room-type priors.

    Does NOT drop detections — only multiplies confidence by a weight.
    The caller decides the threshold.
    """

    def __init__(self, scene_name: str):
        room = _FLOORPLAN_ROOM.get(scene_name, "living room")
        self._weights = _ROOM_PRIORS.get(room, {})
        self._room = room
        self._default_weight = 1.0

    @property
    def room_type(self) -> str:
        return self._room

    def apply(self, detections: list[Detection]) -> list[Detection]:
        """Multiply each detection's confidence by its room-prior weight."""
        for d in detections:
            w = self._weights.get(d.label, self._default_weight)
            d.confidence = round(d.confidence * w, 4)
        return detections
