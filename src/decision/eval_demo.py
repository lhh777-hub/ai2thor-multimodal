"""
Thin wrapper — delegates to ``eval_full.py`` with mode="demo".

Kept for backward compatibility with existing scripts.
For new code, use::

    python -m src.decision.eval_full --mode demo --scenes 5 --tasks-per-scene 10
"""

import sys

from src.decision.eval_full import main as _full_main


def main():
    """Override argv to inject --mode demo, then delegate."""
    if "--mode" not in sys.argv:
        sys.argv.insert(1, "--mode")
        sys.argv.insert(2, "demo")
    # Demo-appropriate defaults
    if "--tasks" not in sys.argv and "--tasks-per-scene" not in sys.argv:
        sys.argv.extend(["--tasks-per-scene", "10"])
    if "--policies" not in sys.argv:
        sys.argv.extend(["--policies", "rule"])
    _full_main()


if __name__ == "__main__":
    main()
