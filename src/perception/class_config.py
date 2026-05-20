"""
Load class configuration from YAML.

Usage::

    from src.perception.class_config import load_config
    cfg = load_config()               # uses default config/classes.yaml
    cfg = load_config("my_classes.yaml")

    cfg.class_names                   # ["person", "bicycle", ...]
    cfg.class_id("chair")             # → 56
    cfg.thor_to_class_id("Chair")     # → 56
    cfg.num_classes                   # → 83 (80 COCO + 3 custom)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import yaml


_DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config", "classes.yaml",
)


@dataclass
class ClassConfig:
    """Loaded class configuration."""
    class_names: list[str]          # index → name
    thor_to_names: dict[str, str]   # AI2-THOR type → YOLO class name
    source_path: str = ""

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def class_id(self, name: str) -> int | None:
        """Find the class ID for *name* (case-insensitive)."""
        nl = name.lower()
        for i, n in enumerate(self.class_names):
            if n.lower() == nl:
                return i
        return None

    def thor_to_class_id(self, thor_type: str) -> int | None:
        """AI2-THOR object type → COCO class ID."""
        yolo_name = self.thor_to_names.get(thor_type)
        if yolo_name is not None:
            return self.class_id(yolo_name)
        # Try case-insensitive
        tl = thor_type.lower()
        for tn, yn in self.thor_to_names.items():
            if tn.lower() == tl:
                return self.class_id(yn)
        return None


def load_config(path: str | None = None) -> ClassConfig:
    """Load class config from a YAML file."""
    path = path or _DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names: list[str] = [str(n).strip() for n in data.get("names", [])]
    thor_map: dict[str, str] = {
        str(k).strip(): str(v).strip()
        for k, v in data.get("thor_to_names", {}).items()
    }

    return ClassConfig(
        class_names=names,
        thor_to_names=thor_map,
        source_path=path,
    )
