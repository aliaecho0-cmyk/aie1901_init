from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from . import actions as A
from .actions import Task
from .state_builder import build_state
from .submission import score_trajectory
from .trajectory_logger import TrajectoryLogger
from .util.llm_client import LLMClient, get_default_client

log = logging.getLogger(__name__)

MAX_REAL_STEPS = 15
REAL_ACTIONS = ("observe", "use_tool", "submit")


def _seq_for(task_type: str) -> list[tuple[str, dict]]:
    if task_type in ("knowledge_work", "knowledge_work_deliverable"):
        return [
            ("read_input", {}),
            ("draft_solution", {}),
            ("check_rubric", {}),
            ("submit", {}),
        ]
    if task_type == "coding":
        return [
            ("read_problem", {}),
            ("write_code", {}),
            ("run_tests", {}),
            ("submit", {}),
        ]
    if task_type == "research":
        return [
            ("search", {"query": ""}),
            ("search", {"query": "compare evidence"}),
            ("draft_solution", {}),
            ("check_rubric", {}),
            ("submit", {}),
        ]
    if task_type == "multi_tool":
        return [
            ("observe", {}),
            ("use_tool", {"tool_name": "read_pdf"}),
            ("use_tool", {"tool_name": "extract_table"}),
            ("use_tool", {"tool_name": "calculator", "tool_args": {"expr": "12*2"}}),
            ("draft_solution", {}),
            ("check_rubric", {}),
            ("submit", {}),
        ]
    if task_type == "long_memory":
        return [
            ("search_memory", {"query": ""}),
            ("search_memory", {"query": "first session"}),
            ("draft_solution", {}),
            ("submit", {}),
        ]
    if task_type == "planning":
        return [
            ("observe", {}),
            ("plan", {}),
            ("verify_solution", {}),
            ("submit", {}),
        ]
    return [("observe", {}), ("draft_solution", {}), ("submit", {})]


def _exec(action_name: str, task: Task, logger: TrajectoryLogger,
          client: LLMClient, args: dict) -> dict:
    if action_name == "read_input":
        return A.act_read_input(task, **args)
    if action_name == "read_problem":
        return A.act_read_problem(task)
    if action_name == "search":
        return A.act_search(task, **args)
    if action_name == "search_memory":
        return A.act_search_memory(task, **args)
    if action_name == "use_tool":
        return A.act_use_tool(task, llm_client=client, **args)
    if action_name == "draft_solution":
        facts = []
        for r in logger.records:
            facts.extend((r.get("observation") or {}).get("extracted_facts") or [])
        return A.act_draft_solution(task, facts_so_far=facts, llm_client=client)
    if action_name == "write_code":
        return A.act_write_code(task, llm_client=client)
    if action_name == "run_tests":
        return A.act_run_tests(task, code_blob=logger.draft_state.get("code_blob") or "")
    if action_name == "revise_code":
        errs = []
        for r in reversed(logger.records):
            tr = (r.get("observation") or {}).get("test_results")
            if tr:
                errs = tr.get("errors") or []
                break
        return A.act_revise_code(task,
                                 code_blob=logger.draft_state.get("code_blob") or "",
                                 test_errors=errs, llm_client=client)
    if action_name == "check_rubric":
        return A.act_check_rubric(task, draft_state=logger.draft_state)
    if action_name == "verify_solution":
        return A.act_verify_solution(task, draft_state=logger.draft_state)
    if action_name == "observe":
        return A.act_observe(task)
    if action_name == "plan":
        return A.act_plan(task, llm_client=client)
    if action_name == "submit":
        return A.act_submit()
    return dict(status="error", summary=f"unknown action {action_name}", cost=0.0)


# ---------------------------------------------------------------------------
# Real tau2 retail runner (M2): 3 actions, dynamic loop
# ---------------------------------------------------------------------------


def _clip_text(text: Any, limit: int = 1200) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"... <{len(text) - limit} chars truncated>"


def _render_trace(trace: list[dict]) -> str:
    if not trace:
        return "(no actions yet)"
    lines: list[str] = []
    for item in trace:
        args = item.get("args") or {}
        if args:
            lines.append(
                f"[step {item['step']}] {item['action']} "
                f"{json.dumps(args, ensure_ascii=False)}"
            )
        else:
            lines.append(f"[step {item['step']}] {item['action']}")
        lines.append(f"  -> {item.get('result', '')}")
    return "\n".join(lines)


def _real_system_prompt(env, max_steps: int) -> str:
    from .tau2_adapter import render_tool_catalog, tool_catalog

    catalog = tool_catalog(env)
    return f"""You are a customer-service agent for an online retail store.

## Store policy
{env.get_policy()}

## Available business tools ({len(catalog)})
You can only act through the `use_tool` action. `tool_name` MUST be one of the
names below and `tool_args` MUST match that tool's JSON schema.
{render_tool_catalog(catalog)}

## Action protocol
Return exactly ONE JSON object per step, with nothing outside it.

1. Inspect the environment (read-only, no side effects):
{{"action": "observe", "thought": "..."}}

2. Execute a business tool against the real database:
{{"action": "use_tool", "tool_name": "<one of the tool names>", "tool_args": {{...}}, "thought": "..."}}

3. Finish the episode and reply to the customer:
{{"action": "submit", "final": "<reply to the customer>", "thought": "..."}}

## Operating mode
There is no live customer on the other end of this conversation. The customer
message you receive is the complete, final instruction: execute it directly.
Do not ask for confirmation, do not wait for a reply, and do not stop at a plan.
Complete the request in this single episode.

## Rules
- Budget: at most {max_steps} steps.
- Never guess ids. Look up the user, order, product and variant records with
  the read tools before calling a write tool.
- Read tools are safe to call repeatedly. Write tools change the database, so
  get their arguments right the first time.
- If a tool returns an error, read the message and fix your arguments.
- Prefer `use_tool` over `observe`; `observe` only prints database statistics.
- `submit` ends the episode. Only use it after the requested database change
  has actually been applied. Never use `submit` to ask a question or request
  confirmation.
"""


def _real_user_prompt(task: Task, trace: list[dict], step: int,
                      max_steps: int) -> str:
    from .tau2_adapter import task_brief

    return (
        f"{task_brief(task)}\n\n"
        f"--- trajectory so far ({step}/{max_steps} steps used) ---\n"
        f"{_render_trace(trace)}\n\n"
        "Return the next action as one JSON object."
    )


def _parse_action(text: str) -> dict:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    start = raw.find("{")
    if start < 0:
        return {"action": "__invalid__", "error": "no JSON object in response"}
    try:
        obj, _ = json.JSONDecoder().raw_decode(raw[start:])
    except json.JSONDecodeError as exc:
        return {"action": "__invalid__", "error": f"invalid JSON: {exc}"}
    if not isinstance(obj, dict) or not isinstance(obj.get("action"), str):
        return {"action": "__invalid__", "error": "missing string `action` field"}
    if obj["action"] not in REAL_ACTIONS:
        return {
            "action": "__invalid__",
            "error": (
                f"unsupported action {obj['action']!r}; "
                f"allowed: {list(REAL_ACTIONS)}"
            ),
        }
    return obj


def run_real_episode(
    task: Task,
    *,
    client: LLMClient | None = None,
    env=None,
    max_steps_override: int | None = None,
) -> tuple[TrajectoryLogger, dict]:
    """3-action dynamic loop against the real tau2 retail environment.

    Contract:
      * the task prompt enters the tau2 history as a ``UserMessage``
      * every ``ToolCall`` lives in an ``AssistantMessage`` and is followed
        immediately by a ``ToolMessage`` with the same id
      * the loop is fully dynamic: the LLM decides when to stop calling tools
    """
    from tau2.data_model.message import AssistantMessage, UserMessage

    from .tau2_adapter import RETAIL_WRITE_TOOLS, build_retail_env, task_brief

    client = client or get_default_client()
    env = env or build_retail_env()
    max_steps = max_steps_override or MAX_REAL_STEPS

    n_criteria = len(task.rubric.get("criteria", []))
    logger = TrajectoryLogger(
        task_id=task.task_id, task_type=task.task_type,
        available_tools=task.available_tools,
        max_steps=max_steps, cost_budget=task.cost_budget,
        n_criteria=n_criteria,
    )

    system_prompt = _real_system_prompt(env, max_steps)
    messages: list = [UserMessage(role="user", content=task_brief(task))]
    trace: list[dict] = []
    termination = "max_steps"
    final_text = ""

    for step in range(max_steps):
        if logger.total_cost >= task.cost_budget:
            logger.log_step(
                action="submit", args={"reason": "budget_exceeded"},
                status="success", summary="forced submit: budget exceeded",
                cost=0.0,
            )
            messages.append(AssistantMessage.text(content=final_text))
            termination = "submit"
            break

        res = client.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",
                 "content": _real_user_prompt(task, trace, step, max_steps)},
            ],
            max_completion_tokens=700,
            temperature=0.0,
        )
        if not res.ok:
            logger.log_step(
                action="__llm_error__", args={}, status="error",
                summary=res.error or "LLM call failed", cost=0.0,
            )
            termination = "llm_error"
            break

        action = _parse_action(res.text)
        name = action.get("action")
        cost = float(res.cost_usd or 0.0)
        thought = action.get("thought") or ""
        if not isinstance(thought, str):
            thought = json.dumps(thought, ensure_ascii=False)

        if name == "observe":
            obs = A.act_observe(task, env=env)
            logger.log_step(
                action="observe", args={},
                status=obs.get("status", "success"),
                summary=obs.get("summary", ""),
                extracted_facts=obs.get("extracted_facts"),
                cost=cost,
            )
            trace.append({"step": step, "action": "observe", "args": {},
                          "result": _clip_text(obs.get("summary"))})
            continue

        if name == "use_tool":
            tool_name = action.get("tool_name")
            tool_args = action.get("tool_args") or {}
            if not isinstance(tool_name, str) or not tool_name:
                summary = "missing `tool_name` for use_tool"
                logger.log_step(action="use_tool",
                                args={"tool_args": tool_args},
                                status="error", summary=summary, cost=cost)
                trace.append({"step": step, "action": "use_tool", "args": {},
                              "result": summary})
                continue

            call_id = f"call_{step}_{uuid.uuid4().hex[:8]}"
            obs = A.act_use_tool(
                task, tool_name=tool_name, tool_args=tool_args,
                env=env, tool_call_id=call_id,
            )
            tool_call = obs.get("tool_call")
            tool_message = obs.get("tool_message")
            if tool_call is not None and tool_message is not None:
                messages.append(
                    AssistantMessage.text(content=thought,
                                          tool_calls=[tool_call])
                )
                messages.append(tool_message)

            logger.log_step(
                action="use_tool",
                args={
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "tool_call_id": call_id,
                    "write": tool_name in RETAIL_WRITE_TOOLS,
                },
                status=obs.get("status", "success"),
                summary=obs.get("summary", ""),
                extracted_facts=obs.get("extracted_facts"),
                tool_name=tool_name,
                tool_args_valid=obs.get("tool_args_valid", True),
                cost=cost,
            )
            trace.append({
                "step": step, "action": "use_tool",
                "args": {"tool_name": tool_name, "tool_args": tool_args},
                "result": _clip_text(obs.get("summary")),
            })
            continue

        if name == "submit":
            final_text = action.get("final") or ""
            if not isinstance(final_text, str):
                final_text = json.dumps(final_text, ensure_ascii=False)
            messages.append(AssistantMessage.text(content=final_text))
            logger.log_step(
                action="submit", args={"final": final_text},
                status="success", summary=_clip_text(final_text, 500),
                cost=cost,
            )
            trace.append({"step": step, "action": "submit", "args": {},
                          "result": final_text})
            termination = "submit"
            break

        summary = action.get("error") or "invalid action"
        logger.log_step(
            action="__invalid__",
            args={"raw": _clip_text(res.text, 400)},
            status="error", summary=summary, cost=cost,
        )
        trace.append({"step": step, "action": "__invalid__", "args": {},
                      "result": summary})
    else:
        messages.append(AssistantMessage.text(content=final_text))

    logger.tau2_messages = messages

    # Official tau2 DB scoring.
    tau2_reward: dict = {}
    try:
        from .tau2_adapter import build_tau2_task
        from .tau2_scorer import build_simulation_run, score_run

        simulation = build_simulation_run(
            messages, task_id=task.task_id, termination=termination,
        )
        tau2_reward = score_run(simulation, build_tau2_task(task))
    except Exception as exc:  # keep the trajectory even if scoring fails
        tau2_reward = {
            "reward": None,
            "error": f"{type(exc).__name__}: {exc}",
        }

    logger.finalize(termination_reason=termination, tau2_reward=tau2_reward)

    scored = score_trajectory(
        task=task, trajectory=logger.records,
        draft_state=logger.draft_state, rubric_status=logger.rubric_status,
        total_cost=logger.total_cost,
    )
    terminal = logger.records[-1]
    final_score = {
        "rubric_score_norm": scored["rubric_score_norm"],
        "rubric_score_raw": scored["rubric_score_raw"],
        "missing_items": scored["missing_items"],
    }
    official_reward = tau2_reward.get("reward")
    if official_reward is not None:
        final_score["legacy_rubric_score_norm"] = final_score["rubric_score_norm"]
        final_score["rubric_score_norm"] = official_reward
        final_score["score_source"] = "tau2_env"
    else:
        final_score["score_source"] = "legacy_rubric"
    terminal["final_rubric_score"] = final_score
    terminal["tau2_message_count"] = len(messages)
    scored["tau2"] = tau2_reward
    return logger, scored


def run_episode(
    task: Task,
    *,
    client: LLMClient | None = None,
    max_steps_override: int | None = None,
    env=None,
) -> tuple[TrajectoryLogger, dict]:
    if task.metadata.get("source") == "tau_bench":
        return run_real_episode(
            task, client=client, env=env,
            max_steps_override=max_steps_override,
        )

    client = client or get_default_client()
    max_steps = max_steps_override or task.max_steps
    n_criteria = len(task.rubric.get("criteria", []))
    logger = TrajectoryLogger(
        task_id=task.task_id, task_type=task.task_type,
        available_tools=task.available_tools,
        max_steps=max_steps, cost_budget=task.cost_budget,
        n_criteria=n_criteria,
    )
    seq = _seq_for(task.task_type)

    seq_idx = 0
    while seq_idx < len(seq) and len(logger.records) < max_steps:
        action_name, args = seq[seq_idx]

        if action_name == "submit" and task.task_type == "coding":
            last = next((r for r in reversed(logger.records)
                         if (r.get("observation") or {}).get("test_results")),
                        None)
            if last:
                tr = (last.get("observation") or {}).get("test_results") or {}
                if tr.get("failed", 0) > 0 and len(logger.records) <= max_steps - 3:
                    seq.insert(seq_idx, ("revise_code", {}))
                    seq.insert(seq_idx + 1, ("run_tests", {}))
                    action_name, args = seq[seq_idx]

        if logger.total_cost >= task.cost_budget:
            logger.log_step(action="submit",
                            args={}, status="success",
                            summary="forced submit due to budget", cost=0.0)
            logger.finalize(termination_reason="budget_exceeded")
            break

        obs = _exec(action_name, task, logger, client, args)
        logger.log_step(
            action=action_name, args=args,
            status=obs.get("status", "success"),
            summary=obs.get("summary", ""),
            extracted_facts=obs.get("extracted_facts"),
            test_results=obs.get("test_results"),
            tool_name=obs.get("tool_name"),
            tool_args_valid=obs.get("tool_args_valid", True),
            cost=obs.get("cost", 0.0),
            draft_update=obs.get("draft_update"),
            rubric_update=obs.get("rubric_update"),
        )
        if action_name == "submit":
            logger.finalize(termination_reason="submit")
            break
        seq_idx += 1
    else:
        if not logger.records or not logger.records[-1].get("terminal"):
            logger.finalize(termination_reason="max_steps")

    scored = score_trajectory(
        task=task, trajectory=logger.records,
        draft_state=logger.draft_state, rubric_status=logger.rubric_status,
        total_cost=logger.total_cost,
    )
    term = logger.records[-1]
    term["final_rubric_score"] = {
        "rubric_score_norm": scored["rubric_score_norm"],
        "rubric_score_raw": scored["rubric_score_raw"],
        "missing_items": scored["missing_items"],
    }
    return logger, scored
