"""Adapter between the local tau_bench retail tasks and the official tau2 retail env.

M1 scope:
  - load a local task directory into the harness ``Task`` dataclass
  - force-expose the official 16 retail tools (ignore task.json available_tools)
  - parse the ground-truth actions out of ``rubric.json``
  - build a fresh tau2 ``Environment`` and its OpenAI tool catalog
  - build the tau2 ``Task`` needed by the official evaluator later

This module deliberately does NOT import ``tau_bench``: its package ``__init__``
pulls in the whole env stack (litellm, etc.).  The GT actions are already
serialized in ``rubric.json`` by ``convert_data.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .actions import Task

# ---------------------------------------------------------------------------
# Official tau2 retail tool inventory (the same 16 in every retail environment)
# ---------------------------------------------------------------------------

ALL_RETAIL_TOOLS: list[str] = [
    "calculate",
    "cancel_pending_order",
    "exchange_delivered_order_items",
    "find_user_id_by_name_zip",
    "find_user_id_by_email",
    "get_order_details",
    "get_product_details",
    "get_item_details",
    "get_user_details",
    "list_all_product_types",
    "modify_pending_order_address",
    "modify_pending_order_items",
    "modify_pending_order_payment",
    "modify_user_address",
    "return_delivered_order_items",
    "transfer_to_human_agents",
]

RETAIL_READ_TOOLS: list[str] = [
    "find_user_id_by_name_zip",
    "find_user_id_by_email",
    "get_order_details",
    "get_product_details",
    "get_item_details",
    "get_user_details",
    "list_all_product_types",
]

RETAIL_WRITE_TOOLS: list[str] = [
    "cancel_pending_order",
    "exchange_delivered_order_items",
    "modify_pending_order_address",
    "modify_pending_order_items",
    "modify_pending_order_payment",
    "modify_user_address",
    "return_delivered_order_items",
]

RETAIL_GENERIC_TOOLS: list[str] = ["calculate", "transfer_to_human_agents"]

assert len(ALL_RETAIL_TOOLS) == 16
assert set(ALL_RETAIL_TOOLS) == set(RETAIL_READ_TOOLS) | set(
    RETAIL_WRITE_TOOLS
) | set(RETAIL_GENERIC_TOOLS)


# ---------------------------------------------------------------------------
# Task loading
# ---------------------------------------------------------------------------


def load_real_task(task_dir: str | Path) -> Task:
    """Load one local task directory and force the official tool inventory.

    ``task.json`` only lists the write tool used by the ground truth.  For a
    real run the agent must be able to look records up first, so the field is
    overwritten with all 16 official tools.  The original value is kept in
    ``metadata`` for reference.
    """
    task = Task.load(task_dir)
    task.metadata.setdefault("declared_tools", list(task.available_tools))
    task.available_tools = list(ALL_RETAIL_TOOLS)
    return task


def parse_gold_actions(task: Task) -> list[dict[str, Any]]:
    """Extract the ground-truth actions serialized in ``rubric.json``.

    ``convert_data.py`` writes one criterion per task whose description is:

        Agent must execute actions: [{"name": ..., "kwargs": {...}}, ...]

    We locate the JSON array after the ``actions:`` marker and decode it.
    """
    actions: list[dict[str, Any]] = []
    for criterion in task.rubric.get("criteria", []):
        desc = criterion.get("description") or ""
        actions.extend(_extract_action_array(desc))
    return actions


def _extract_action_array(description: str) -> list[dict[str, Any]]:
    marker = "actions:"
    marker_idx = description.find(marker)
    if marker_idx < 0:
        return []
    start = description.find("[", marker_idx)
    if start < 0:
        return []
    try:
        payload, _ = json.JSONDecoder().raw_decode(description[start:])
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    cleaned: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict) or "name" not in item:
            continue
        cleaned.append(
            {
                "name": item["name"],
                "kwargs": item.get("kwargs") or {},
            }
        )
    return cleaned


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def build_retail_env():
    """Build a fresh official retail environment reset to the initial DB."""
    from tau2.domains.retail.environment import get_environment

    env = get_environment()
    env.set_state(
        initialization_data=None,
        initialization_actions=None,
        message_history=[],
    )
    return env


def tool_catalog(env) -> list[dict[str, Any]]:
    """Return the official tools in OpenAI function-calling schema form."""
    return [tool.openai_schema for tool in env.get_tools()]


def render_tool_catalog(catalog: list[dict[str, Any]]) -> str:
    """Render the catalog as compact JSON lines for a system prompt."""
    rendered = []
    for entry in catalog:
        fn = entry.get("function", {})
        rendered.append(
            {
                "name": fn.get("name"),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
            }
        )
    return json.dumps(rendered, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Prompt / tau2 Task construction
# ---------------------------------------------------------------------------


def task_brief(task: Task) -> str:
    """Turn the user-simulator prompt into an agent-facing customer message.

    The local prompt is written as the *user's* instruction ("Your name is
    Omar Anderson...").  Feeding it to the agent as if it were an instruction
    makes the model adopt the customer's identity, so it is wrapped in an
    explicit first-person-of-the-customer framing.
    """
    return (
        "A customer has just written to you:\n\n"
        f'"{task.prompt}"\n\n'
        "You are the customer-service agent, not the customer. "
        "Use the retail tools to look up the real records and complete the "
        "customer's request. When the request is fully handled, call the "
        "`submit` action."
    )


def build_tau2_task(task: Task):
    """Build the tau2 ``Task`` object the official evaluator consumes."""
    from tau2.data_model.tasks import (
        Action,
        EvaluationCriteria,
        RewardType,
        Task as Tau2Task,
        UserScenario,
    )

    gold = parse_gold_actions(task)
    actions = [
        Action(
            action_id=f"{task.task_id}_{idx}",
            name=item["name"],
            arguments=item.get("kwargs") or {},
        )
        for idx, item in enumerate(gold)
    ]
    return Tau2Task(
        id=task.task_id,
        description=None,
        user_scenario=UserScenario(persona=None, instructions=task.prompt),
        initial_state=None,
        evaluation_criteria=EvaluationCriteria(
            actions=actions,
            env_assertions=None,
            communicate_info=[],
            nl_assertions=None,
            # DB is the real signal; COMMUNICATE is a neutral 1.0 placeholder.
            reward_basis=[RewardType.DB, RewardType.COMMUNICATE],
        ),
    )


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------


def _default_task_root() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "data"
        / "synthetic_tasks"
        / "tau_bench_retail"
    )


def main() -> None:
    env = build_retail_env()
    catalog = tool_catalog(env)
    names = [entry["function"]["name"] for entry in catalog]
    print(f"official tools: {len(catalog)}")
    print(f"tool names: {sorted(names)}")

    root = _default_task_root()
    task_dirs = sorted(
        p for p in root.iterdir() if p.is_dir() and (p / "task.json").exists()
    )
    print(f"task dirs: {len(task_dirs)}")

    total_actions = 0
    empty: list[str] = []
    for task_dir in task_dirs:
        task = load_real_task(task_dir)
        if len(task.available_tools) != 16:
            raise AssertionError(
                f"{task.task_id}: available_tools={len(task.available_tools)}"
            )
        gold = parse_gold_actions(task)
        if not gold:
            empty.append(task.task_id)
        total_actions += len(gold)

    print(f"gold actions: {len(task_dirs) - len(empty)} tasks OK, total={total_actions}")
    if empty:
        print(f"tasks without parsed actions ({len(empty)}): {empty[:10]}")


if __name__ == "__main__":
    main()
