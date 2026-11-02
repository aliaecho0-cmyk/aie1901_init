"""M2 smoke test: 3-action dynamic loop (observe / use_tool / submit).

Run:
    .venv\\Scripts\\python.exe test_m2.py
    .venv\\Scripts\\python.exe test_m2.py --task-dir data/synthetic_tasks/tau_bench_retail/task_82
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
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
    parse_gold_actions,
)
from harness.util.llm_client import LLMClient  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-dir",
        default="data/synthetic_tasks/tau_bench_retail/task_0",
    )
    parser.add_argument("--max-steps", type=int, default=15)
    args = parser.parse_args()

    task_dir = Path(args.task_dir)
    if not task_dir.is_absolute():
        task_dir = ROOT / task_dir

    missing = [key for key in ("OPENAI_API_KEY",) if not os.getenv(key)]
    if missing:
        print(f"[warn] missing env vars: {missing}")

    task = load_real_task(task_dir)
    env = build_retail_env()
    initial_hash = env.get_db_hash()

    print("=" * 78)
    print(f"task:      {task.task_id}")
    print(f"customer:  {task.prompt}")
    print(f"gt actions:{json.dumps(parse_gold_actions(task), ensure_ascii=False)}")
    print(f"max steps: {args.max_steps}")
    print("=" * 78)

    client = LLMClient(verbose=False)
    logger, _ = run_real_episode(
        task, client=client, env=env, max_steps_override=args.max_steps,
    )

    for rec in logger.records:
        obs = rec.get("observation") or {}
        tool_name = (rec.get("args") or {}).get("tool_name")
        tag = ""
        if tool_name in RETAIL_WRITE_TOOLS:
            tag = "  <-- WRITE"
        print(f"\n[{rec['step']:02d}] {rec['action']}{tag}")
        if tool_name:
            print(f"     tool: {tool_name}")
            print(
                "     args: "
                + json.dumps(
                    (rec.get("args") or {}).get("tool_args", {}),
                    ensure_ascii=False,
                )
            )
        print(f"     status: {obs.get('status')}")
        print(f"     result: {(obs.get('summary') or '')[:700]}")

    terminal = logger.records[-1]
    print("\n" + "=" * 78)
    print("termination:", terminal.get("termination_reason"))
    print("tau2 messages:", len(logger.tau2_messages))
    print("tool calls:", sum(1 for r in logger.records if r["action"] == "use_tool"))
    print(
        "write calls:",
        sum(
            1
            for r in logger.records
            if (r.get("args") or {}).get("tool_name") in RETAIL_WRITE_TOOLS
        ),
    )
    print("db changed:", env.get_db_hash() != initial_hash)
    print("initial db hash:", initial_hash)
    print("final   db hash:", env.get_db_hash())

    print("\n--- tau2 replay contract check ---")
    pending: list[str] = []
    ok = True
    for message in logger.tau2_messages:
        name = type(message).__name__
        tool_calls = getattr(message, "tool_calls", None) or []
        if tool_calls:
            pending = [tc.id for tc in tool_calls]
            print(f"{name}: tool_calls={[tc.name for tc in tool_calls]}")
        elif name == "ToolMessage":
            if not pending or message.id != pending[0]:
                ok = False
                print(f"{name}: id={message.id} WITHOUT matching ToolCall")
            else:
                pending.pop(0)
                print(f"{name}: id={message.id} error={message.error}")
        else:
            content = (message.content or "").replace("\n", " ")
            print(f"{name}: {content[:160]!r}")
    print("every ToolCall followed by matching ToolMessage:", ok and not pending)


if __name__ == "__main__":
    main()
