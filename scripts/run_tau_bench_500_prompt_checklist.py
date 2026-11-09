#!/usr/bin/env python
"""Run the tau-bench retail system-prompt checklist condition.

Examples:
    .venv\\Scripts\\python.exe scripts\\run_tau_bench_500_prompt_checklist.py --limit 1
    .venv\\Scripts\\python.exe scripts\\run_tau_bench_500_prompt_checklist.py
"""

from run_tau_bench_500 import ROOT, main


DEFAULT_PROMPT_OUT_DIR = ROOT / "runs" / "tau_bench_500_prompt_checklist"


if __name__ == "__main__":
    main(
        temperature=0.0,
        default_out_dir=DEFAULT_PROMPT_OUT_DIR,
        prompt_variant="checklist",
        description=__doc__,
    )
