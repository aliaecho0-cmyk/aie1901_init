#!/usr/bin/env python
"""M4: run the local tau-bench retail split end to end and score every episode.

Each task is an independent episode:
  * a fresh official retail environment is built (initial DB) for the task
  * the 3-action loop (observe / use_tool / submit) runs until submit / budget
  * the recorded tau2 messages are scored by ``EnvironmentEvaluator`` (DB only)
  * the full trajectory, including the official ``tau2`` reward, is written to
    ``<out-dir>/trajectories/<task_id>.jsonl``
  * ``<out-dir>/summary.json`` aggregates the run

Usage:
    .venv\\Scripts\\python.exe scripts/run_tau_bench_500.py --limit 5
    .venv\\Scripts\\python.exe scripts/run_tau_bench_500.py            # all 500
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

try:  # optional: pick up .env if the user has one
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from harness.agent import run_real_episode  # noqa: E402
from harness.tau2_adapter import (  # noqa: E402
    RETAIL_WRITE_TOOLS,
    build_retail_env,
    load_real_task,
)
from harness.util.llm_client import LLMClient  # noqa: E402

DEFAULT_DATA_ROOT = ROOT / "data" / "synthetic_tasks" / "tau_bench_retail"
DEFAULT_OUT_DIR = ROOT / "runs" / "tau_bench_500"


# ---------------------------------------------------------------------------
# Task discovery
# ---------------------------------------------------------------------------


def _rel(path: Path) -> str:
    """Path relative to the project root when possible."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _task_index(path: Path) -> int:
    match = re.search(r"(\d+)$", path.name)
    return int(match.group(1)) if match else 10**9


def discover_tasks(data_root: Path, *, start: int, limit: int | None) -> list[Path]:
    dirs = [
        p
        for p in data_root.iterdir()
        if p.is_dir() and (p / "task.json").exists()
    ]
    dirs.sort(key=lambda p: (_task_index(p), p.name))
    dirs = dirs[start:]
    if limit is not None:
        dirs = dirs[:limit]
    return dirs


# ---------------------------------------------------------------------------
# One episode
# ---------------------------------------------------------------------------


def _count(records: list[dict], action: str) -> int:
    return sum(1 for rec in records if rec.get("action") == action)


def run_one(
    task_dir: Path,
    client: LLMClient,
    *,
    max_steps: int,
) -> tuple[Any, Any, dict]:
    """Run one task in a clean environment and return (logger, task, entry)."""
    task = load_real_task(task_dir)

    # A brand-new environment per task: no state leaks between episodes.
    env = build_retail_env()
    initial_hash = env.get_db_hash()

    t0 = time.monotonic()
    logger, _ = run_real_episode(
        task, client=client, env=env, max_steps_override=max_steps,
    )
    elapsed = time.monotonic() - t0

    terminal = logger.records[-1]
    tau2 = terminal.get("tau2") or {}
    records = logger.records
    tool_calls = _count(records, "use_tool")
    write_calls = sum(
        1
        for rec in records
        if (rec.get("args") or {}).get("tool_name") in RETAIL_WRITE_TOOLS
    )
    final_hash = env.get_db_hash()

    entry = {
        "task_id": task.task_id,
        "task_dir": _rel(task_dir),
        "reward": tau2.get("reward"),
        "db_match": tau2.get("db_match"),
        "db_reward": tau2.get("db_reward"),
        "communicate_reward": tau2.get("communicate_reward"),
        "termination_reason": terminal.get("termination_reason"),
        "steps": len(records),
        "tool_calls": tool_calls,
        "write_calls": write_calls,
        "invalid_actions": _count(records, "__invalid__"),
        "llm_errors": _count(records, "__llm_error__"),
        "db_changed": final_hash != initial_hash,
        "predicted_db_hash": tau2.get("predicted_db_hash"),
        "gold_db_hash": tau2.get("gold_db_hash"),
        "warnings": tau2.get("warnings") or [],
        "scoring_error": tau2.get("error"),
        "cost_usd": round(float(logger.total_cost), 6),
        "elapsed_s": round(elapsed, 2),
        "error": None,
    }
    return logger, task, entry


def run_with_retries(
    task_dir: Path,
    client: LLMClient,
    *,
    max_steps: int,
    retries: int,
) -> tuple[Any, Any, dict]:
    """Retry an episode only when it died on a transport-level LLM error."""
    last: tuple[Any, Any, dict] | None = None
    for attempt in range(retries + 1):
        logger, task, entry = run_one(task_dir, client, max_steps=max_steps)
        entry["attempts"] = attempt + 1
        last = (logger, task, entry)
        if entry["termination_reason"] != "llm_error":
            break
        if attempt < retries:
            print(
                f"  ! {entry['task_id']}: LLM error, retrying "
                f"({attempt + 2}/{retries + 1})",
                flush=True,
            )
            time.sleep(2.0 * (attempt + 1))
    assert last is not None
    return last


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def build_summary(
    entries: list[dict],
    *,
    started_at: str,
    elapsed_s: float,
    config: dict,
    interrupted: bool,
) -> dict:
    ok = [e for e in entries if e.get("error") is None]
    rewards = [float(e["reward"]) for e in ok if isinstance(e.get("reward"), (int, float))]
    db_flags = [bool(e["db_match"]) for e in ok if e.get("db_match") is not None]
    tool_counts = [float(e["tool_calls"]) for e in ok if e.get("tool_calls") is not None]
    steps = [float(e["steps"]) for e in ok if e.get("steps") is not None]
    writes = [float(e["write_calls"]) for e in ok]
    costs = [float(e["cost_usd"]) for e in ok]

    terminations: dict[str, int] = {}
    for entry in ok:
        key = str(entry.get("termination_reason"))
        terminations[key] = terminations.get(key, 0) + 1

    histogram: dict[str, int] = {}
    for reward in rewards:
        key = f"{reward:g}"
        histogram[key] = histogram.get(key, 0) + 1

    return {
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "interrupted": interrupted,
        "config": config,
        "counts": {
            "requested": len(entries),
            "completed": len(ok),
            "errored": len(entries) - len(ok),
            "scored": len(rewards),
        },
        "mean_reward": _mean(rewards),
        "db_match_rate": _mean([1.0 if flag else 0.0 for flag in db_flags]),
        "avg_tool_calls": _mean(tool_counts),
        "avg_write_calls": _mean(writes),
        "avg_steps": _mean(steps),
        "avg_cost_usd": _mean(costs),
        "total_cost_usd": round(sum(costs), 6),
        "elapsed_s": round(elapsed_s, 2),
        "reward_histogram": histogram,
        "termination_histogram": terminations,
        "tasks": entries,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="run only the first N tasks")
    parser.add_argument("--start", type=int, default=0,
                        help="skip the first N tasks")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--task-retries", type=int, default=1,
                        help="retries when an episode ends on an LLM error")
    parser.add_argument("--show-tool-logs", action="store_true",
                        help="keep tau2's per-tool DEBUG output (very noisy)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not args.show_tool_logs:
        try:
            from loguru import logger as _loguru

            # tau2 logs every tool response at DEBUG; over 500 tasks that is
            # megabytes of noise around the actual progress lines.
            _loguru.disable("tau2.environment.environment")
            _loguru.disable("tau2.registry")
        except ImportError:
            pass

    if not os.getenv("OPENAI_API_KEY"):
        print("[fatal] OPENAI_API_KEY is not set (check .env)", flush=True)
        sys.exit(2)

    data_root = args.data_root if args.data_root.is_absolute() else ROOT / args.data_root
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    traj_dir = out_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)

    task_dirs = discover_tasks(data_root, start=args.start, limit=args.limit)
    if not task_dirs:
        print(f"[fatal] no tasks found under {data_root}", flush=True)
        sys.exit(2)

    client = LLMClient(verbose=False)
    config = {
        "data_root": _rel(data_root),
        "limit": args.limit,
        "start": args.start,
        "max_steps": args.max_steps,
        "task_retries": args.task_retries,
        "model": client.deployment,
        "endpoint": client.endpoint,
    }

    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    t_start = time.monotonic()
    entries: list[dict] = []
    interrupted = False

    print("=" * 78)
    print(f"tau-bench retail run: {len(task_dirs)} task(s)")
    print(f"data:  {data_root}")
    print(f"out:   {out_dir}")
    print(f"model: {client.deployment} @ {client.endpoint}")
    print("=" * 78, flush=True)

    try:
        for idx, task_dir in enumerate(task_dirs, 1):
            tag = f"[{idx}/{len(task_dirs)}] {task_dir.name}"
            try:
                logger, task, entry = run_with_retries(
                    task_dir, client,
                    max_steps=args.max_steps,
                    retries=args.task_retries,
                )
                traj_path = logger.to_jsonl(traj_dir / f"{task.task_id}.jsonl")
                entry["trajectory_file"] = _rel(traj_path)
            except Exception as exc:  # keep going; the run must not die
                err = traceback.format_exc()
                print(f"{tag} FAILED: {type(exc).__name__}: {exc}", flush=True)
                if not args.quiet:
                    print(err, flush=True)
                entries.append({
                    "task_id": task_dir.name,
                    "task_dir": _rel(task_dir),
                    "reward": None,
                    "db_match": None,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": err.splitlines()[-6:],
                })
                continue

            entries.append(entry)
            reward = entry["reward"]
            reward_txt = "n/a" if reward is None else f"{reward:.2f}"
            print(
                f"{tag} reward={reward_txt} "
                f"db_match={entry['db_match']} "
                f"steps={entry['steps']} tools={entry['tool_calls']} "
                f"writes={entry['write_calls']} "
                f"{entry['termination_reason']} "
                f"{entry['elapsed_s']:.1f}s",
                flush=True,
            )
    except KeyboardInterrupt:
        interrupted = True
        print("\n[interrupt] stopping after the current task", flush=True)

    elapsed = time.monotonic() - t_start
    summary = build_summary(
        entries, started_at=started_at, elapsed_s=elapsed,
        config=config, interrupted=interrupted,
    )
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for key in (
        "counts", "mean_reward", "db_match_rate", "avg_tool_calls",
        "avg_write_calls", "avg_steps", "avg_cost_usd", "total_cost_usd",
        "reward_histogram", "termination_histogram", "elapsed_s",
    ):
        print(f"{key}: {summary[key]}")
    print(f"\nsummary: {summary_path}")
    print(f"trajectories: {traj_dir}")


if __name__ == "__main__":
    main()
