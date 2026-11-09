#!/usr/bin/env python
"""Run the tau-bench retail temperature-0.2 condition.

Examples:
    .venv\\Scripts\\python.exe scripts\\run_tau_bench_500_temp02.py --limit 1
    .venv\\Scripts\\python.exe scripts\\run_tau_bench_500_temp02.py
"""

from pathlib import Path

from run_tau_bench_500 import ROOT, main


DEFAULT_TEMP02_OUT_DIR = ROOT / "runs" / "tau_bench_500_temp02"


if __name__ == "__main__":
    main(
        temperature=0.2,
        default_out_dir=DEFAULT_TEMP02_OUT_DIR,
        description=__doc__,
    )
