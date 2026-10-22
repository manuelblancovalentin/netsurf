"""Minimal dependency-free runner for assertion-based canonical tests."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
import sys
import traceback


def main() -> int:
    root = Path(__file__).resolve().parent
    failures = 0
    executed = 0
    for path in sorted(root.glob("test_*.py")):
        module = importlib.import_module(f"tests.{path.stem}")
        for name, function in sorted(inspect.getmembers(module, inspect.isfunction)):
            if not name.startswith("test_"):
                continue
            executed += 1
            try:
                function()
                print(f"PASS {path.stem}.{name}")
            except Exception:
                failures += 1
                print(f"FAIL {path.stem}.{name}", file=sys.stderr)
                traceback.print_exc()
    print(f"{executed - failures} passed, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
