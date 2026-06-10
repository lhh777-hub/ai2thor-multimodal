"""TaskParser — natural-language instruction → TaskSpec."""

from __future__ import annotations

from src.decision.types import TaskSpec

# Verbs that indicate interaction rather than navigation
_INTERACT_VERBS: list[tuple[str, str]] = [
    # Multi-word phrases first (checked before single-word verbs)
    ("turn on",   "TOGGLE"),
    ("turn off",  "TOGGLE"),
    ("switch on",  "TOGGLE"),
    ("switch off", "TOGGLE"),
    ("toggle on", "TOGGLE"),
    ("toggle off", "TOGGLE"),
    # Single-word verbs
    ("open",   "OPEN"),
    ("close",  "OPEN"),       # closing = same AI2-THOR action on openable objects
    ("pick up", "PICKUP"),
    ("pick",   "PICKUP"),
    ("take",   "PICKUP"),
    ("grab",   "PICKUP"),
    ("toggle", "TOGGLE"),
]

# Navigation phrases — if any appear in the instruction, it's a navigation task
_NAVIGATION_PHRASES = [
    "go to", "find", "navigate to", "walk to", "move to",
    "go near", "approach", "locate", "where is",
]


def _clean_target(text: str) -> str:
    """Remove leading articles ('the', 'a', 'an') and trailing punctuation."""
    t = text.strip()
    for article in ("the ", "an ", "a "):
        if t.startswith(article):
            t = t[len(article):]
            break
    return t.strip().rstrip(".").strip()


class TaskParser:
    """Parse a natural-language instruction into a structured ``TaskSpec``.

    Examples
    --------
    >>> TaskParser.interpret("Go to the chair")
    TaskSpec(type="navigation", target="chair", interact_action="")
    >>> TaskParser.interpret("Open the fridge")
    TaskSpec(type="interaction", target="fridge", interact_action="OPEN")
    >>> TaskParser.interpret("Pick up the mug")
    TaskSpec(type="interaction", target="mug", interact_action="PICKUP")
    >>> TaskParser.interpret("Toggle on the lamp")
    TaskSpec(type="interaction", target="lamp", interact_action="TOGGLE")
    """

    @staticmethod
    def interpret(text: str) -> TaskSpec:
        """Parse *text* into a ``TaskSpec``.

        Raises ``ValueError`` if no target can be extracted.
        """
        lowered = text.lower().strip()

        # 1) Check for interaction verbs
        for verb, action in _INTERACT_VERBS:
            if lowered.startswith(verb + " ") or lowered.startswith(verb + " the "):
                target = lowered.removeprefix(verb).strip()
                target = _clean_target(target)
                if not target:
                    raise ValueError(f"Could not extract target from: '{text}'")
                return TaskSpec(
                    task_type="interaction",
                    target=target,
                    interact_action=action,
                    raw_text=text,
                )

        # 2) Check for navigation phrases
        for phrase in _NAVIGATION_PHRASES:
            if phrase in lowered:
                parts = lowered.split(phrase, 1)
                tail = _clean_target(parts[-1])
                if not tail:
                    raise ValueError(f"Could not extract target from: '{text}'")
                return TaskSpec(
                    task_type="navigation",
                    target=tail,
                    interact_action="",
                    raw_text=text,
                )

        # 3) Fallback: treat entire string as "go to X"
        target = _clean_target(lowered)
        if not target:
            raise ValueError(f"Could not extract target from: '{text}'")
        return TaskSpec(
            task_type="navigation",
            target=target,
            interact_action="",
            raw_text=text,
        )
