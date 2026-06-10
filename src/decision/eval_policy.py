"""
Thin wrapper — delegates to ``eval_full.py`` with mode="quick".

Kept for backward compatibility with existing documentation and scripts.
For new code, use::

    python -m src.decision.eval_full --mode quick --scenes 3 --tasks 10
"""

import sys

from src.decision.eval_full import main as _full_main


def main():
    """Override argv to inject --mode quick, then delegate."""
    # Inject --mode quick if not already present
    if "--mode" not in sys.argv:
        sys.argv.insert(1, "--mode")
        sys.argv.insert(2, "quick")
    # Set quick-appropriate defaults if not overridden
    if "--scenes" not in sys.argv:
        sys.argv.extend(["--scenes", "3"])
    if "--tasks" not in sys.argv and "--tasks-per-scene" not in sys.argv:
        sys.argv.extend(["--tasks", "10"])
    if "--policies" not in sys.argv:
        sys.argv.extend(["--policies", "rule"])
    _full_main()


if __name__ == "__main__":
    main()
