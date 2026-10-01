"""Run the whole pipeline end to end:  python -m segar_churn.pipeline

    generate_data -> data_checks -> features -> stats_evidence -> train -> explain -> score -> dashboard
"""
from __future__ import annotations

import runpy
import sys
import time

from . import config as C

STEPS = ["generate_data", "data_checks", "features", "stats_evidence", "train", "explain", "score"]


def main() -> None:
    skip_generate = "--use-existing-raw" in sys.argv
    for step in STEPS:
        if step == "generate_data" and skip_generate:
            print("== generate_data: skipped (--use-existing-raw)")
            continue
        t0 = time.time()
        print(f"\n== {step}")
        runpy.run_module(f"segar_churn.{step}", run_name="__main__")
        print(f"   ({time.time() - t0:.1f}s)")
    print("\n== dashboard")
    runpy.run_path(str(C.APP / "build_dashboard.py"), run_name="__main__")
    print("\nDone. Open app/dashboard.html in a browser.")


if __name__ == "__main__":
    main()
