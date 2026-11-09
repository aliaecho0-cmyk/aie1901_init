"""M3 test: official tau2 EnvironmentEvaluator reward, including a negative control.

Run every mode (oracle / negative / real):
    .venv\\Scripts\\python.exe test_m3.py

Pick one:
    .venv\\Scripts\\python.exe test_m3.py --mode oracle
    .venv\\Scripts\\python.exe test_m3.py --mode negative
    .venv\\Scripts\\python.exe test_m3.py --mode tamper
    .venv\\Scripts\\python.exe test_m3.py --mode real
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

    load_dotenv(ROOT / ".env", override=True)
except ImportError:
    pass

from harness.agent import run_real_episode  # noqa: E402
from harness.tau2_adapter import (  # noqa: E402
    RETAIL_WRITE_TOOLS,
    build_retail_env,
    build_tau2_task,
    load_real_task,
    parse_gold_actions,
    task_brief,
)
from harness.tau2_scorer import build_simulation_run, score_run  # noqa: E402
from harness.util.llm_client import LLMClient  # noqa: E402


def _score(task, messages, termination: str = "submit") -> dict:
    tau2_task = build_tau2_task(task)
    simulation = build_simulation_run(
        messages, task_id=task.task_id, termination=termination,
    )
    return score_run(simulation, tau2_task)


def _oracle_messages(task, *, skip_write: bool = False,
                     tamper: bool = False) -> list:
    """Replay the ground-truth actions without any LLM."""
    from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage

    env = build_retail_env()
    messages: list = [UserMessage(role="user", content=task_brief(task))]

    for idx, action in enumerate(parse_gold_actions(task)):
        name = action["name"]
        kwargs = dict(action["kwargs"])

        if name in RETAIL_WRITE_TOOLS:
            if skip_write:
                continue
            if tamper:
                kwargs = _tamper(name, kwargs, env)

        tool_call = ToolCall(id=f"oracle_{idx}", name=name, arguments=kwargs)
        tool_message = env.get_response(tool_call)
        messages.append(
            AssistantMessage.text(content="", tool_calls=[tool_call])
        )
        messages.append(tool_message)

    messages.append(AssistantMessage.text(content="done"))
    return messages


def _tamper(name: str, kwargs: dict, env) -> dict:
    """Make a plausible-but-wrong write so the final DB must differ from gold."""
    if name == "return_delivered_order_items":
        order = env.tools.db.orders[kwargs["order_id"]]
        other = [
            item.item_id for item in order.items
            if item.item_id not in kwargs.get("item_ids", [])
        ]
        if other:
            kwargs["item_ids"] = other[:1]
    elif name == "exchange_delivered_order_items":
        order = env.tools.db.orders[kwargs["order_id"]]
        kwargs["new_item_ids"] = [item.item_id for item in order.items][
            : len(kwargs.get("new_item_ids", []))
        ]
    elif name == "cancel_pending_order":
        kwargs["reason"] = (
            "ordered by mistake"
            if kwargs.get("reason") == "no longer needed"
            else "no longer needed"
        )
    elif name in ("modify_pending_order_address", "modify_user_address"):
        kwargs["address1"] = str(kwargs.get("address1", "")) + " (tampered)"
    elif name == "modify_pending_order_payment":
        order = env.tools.db.orders[kwargs["order_id"]]
        user = env.tools.db.users[order.user_id]
        others = [
            pid for pid in user.payment_methods
            if pid != kwargs.get("payment_method_id")
        ]
        if others:
            kwargs["payment_method_id"] = others[0]
    return kwargs


def _report(label: str, result: dict, expected: float | None = None) -> None:
    print(f"\n{'=' * 78}")
    print(f"{label}")
    print("=" * 78)
    print("reward:            ", result.get("reward"))
    print("db_match:          ", result.get("db_match"))
    print("db_reward:         ", result.get("db_reward"))
    print("communicate_reward:", result.get("communicate_reward"))
    print("reward_basis:      ", result.get("reward_basis"))
    print("predicted_db_hash: ", result.get("predicted_db_hash"))
    print("gold_db_hash:      ", result.get("gold_db_hash"))
    print(
        "hashes match:      ",
        result.get("predicted_db_hash") == result.get("gold_db_hash"),
    )
    if result.get("warnings"):
        print("warnings:          ", result["warnings"])
    if result.get("info"):
        print("info:              ", result["info"])
    if expected is not None:
        verdict = "PASS" if result.get("reward") == expected else "FAIL"
        print(f"\n[{verdict}] expected reward {expected}, got {result.get('reward')}")


def _run_oracle(task, *, skip_write: bool = False, tamper: bool = False) -> dict:
    messages = _oracle_messages(task, skip_write=skip_write, tamper=tamper)
    return _score(task, messages)


def _run_real(task, max_steps: int) -> dict:
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[skip] real mode: OPENAI_API_KEY is not set")
        return {}
    client = LLMClient(verbose=False)
    logger, _ = run_real_episode(
        task, client=client, env=build_retail_env(),
        max_steps_override=max_steps,
    )
    terminal = logger.records[-1]
    print(
        "\nreal trajectory:",
        [(r["action"], (r.get("args") or {}).get("tool_name"))
         for r in logger.records],
    )
    print("real termination:", terminal.get("termination_reason"))
    return terminal.get("tau2") or {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-dir",
        default="data/synthetic_tasks/tau_bench_retail/task_0",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "oracle", "negative", "tamper", "real"],
        default="all",
    )
    parser.add_argument("--max-steps", type=int, default=15)
    args = parser.parse_args()

    task_dir = Path(args.task_dir)
    if not task_dir.is_absolute():
        task_dir = ROOT / task_dir

    task = load_real_task(task_dir)
    print("=" * 78)
    print(f"task:       {task.task_id}")
    print(f"customer:   {task.prompt}")
    print(f"gt actions: {json.dumps(parse_gold_actions(task), ensure_ascii=False)}")
    print("=" * 78)

    if args.mode in ("all", "oracle"):
        _report("ORACLE: replay ground-truth actions", _run_oracle(task), 1.0)

    if args.mode in ("all", "negative"):
        _report(
            "NEGATIVE: skip the write action",
            _run_oracle(task, skip_write=True),
            0.0,
        )

    if args.mode in ("all", "tamper"):
        _report(
            "NEGATIVE: tamper with the write action",
            _run_oracle(task, tamper=True),
            0.0,
        )

    if args.mode in ("all", "real"):
        result = _run_real(task, args.max_steps)
        if result:
            _report("REAL: LLM harness run", result)


if __name__ == "__main__":
    main()
