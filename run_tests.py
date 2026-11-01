"""
Run every phase's tests in order:   python run_tests.py

Each test_phaseN.py keeps its own tests (they document how the project was
built); this file just discovers and runs them all, and stops at the first
failure so the traceback points at the right phase.
"""

import glob
import importlib
import sys
import traceback

import matplotlib
matplotlib.use("Agg")                     # plots go to files, no windows


def main():
    files = sorted(glob.glob("test_phase*.py"))
    total = 0
    for f in files:
        module = importlib.import_module(f[:-3])
        tests = [v for k, v in vars(module).items() if k.startswith("test_")]
        print(f"\n== {f} ({len(tests)} tests) ==")
        for t in tests:
            try:
                t()
                print(f"PASS  {t.__name__}")
                total += 1
            except Exception:
                print(f"FAIL  {t.__name__}")
                traceback.print_exc()
                sys.exit(1)
    print(f"\nAll {total} tests across {len(files)} phases passed.")


if __name__ == "__main__":
    main()
