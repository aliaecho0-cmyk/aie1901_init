"""Official tau2 scoring bridge for the real harness runs.

``score_run`` wraps ``tau2.evaluator.evaluate_simulation`` with
``EvaluationType.ENV``: the agent's recorded tau2 messages are replayed into a
fresh retail environment, the ground-truth actions are replayed into another,
and the two final DB hashes are compared.

``communicate`` has no task-side data in this dataset, so it is reported as a
neutral 1.0.  The official reward is therefore exactly the DB reward.

All tau2 imports are lazy so the mock domains do not depend on tau2 being
installed.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _termination_reason(name: str | None):
    from tau2.data_model.simulation import TerminationReason

    mapping = {
        "submit": TerminationReason.AGENT_STOP,
        "agent_stop": TerminationReason.AGENT_STOP,
        "user_stop": TerminationReason.USER_STOP,
        "max_steps": TerminationReason.MAX_STEPS,
        "llm_error": TerminationReason.AGENT_ERROR,
    }
    return mapping.get(name or "", TerminationReason.AGENT_ERROR)


def build_simulation_run(
    messages: list,
    *,
    task_id: str,
    termination: str = "submit",
    start_time: str | None = None,
    end_time: str | None = None,
):
    """Wrap recorded tau2 messages into a ``SimulationRun``."""
    from tau2.data_model.simulation import SimulationRun
    from tau2.orchestrator.modes import CommunicationMode

    return SimulationRun(
        id=f"sim_{task_id}_{uuid.uuid4().hex[:8]}",
        task_id=task_id,
        start_time=start_time or _now(),
        end_time=end_time or _now(),
        duration=0.0,
        termination_reason=_termination_reason(termination),
        messages=list(messages),
        seed=None,
        mode=CommunicationMode.HALF_DUPLEX.value,
    )


def score_run(
    simulation,
    tau2_task,
    *,
    strict_replay: bool = True,
) -> dict[str, Any]:
    """Score one simulation with the official environment evaluator.

    Returns a flat dict with the reward, the DB check, both hashes and the full
    official ``RewardInfo`` payload (``official``).
    """
    from tau2.data_model.semantic_communication import (
        CommunicateScoring,
        CommunicateScorer,
    )
    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
    from tau2.orchestrator.modes import CommunicationMode

    warnings: list[str] = []
    try:
        reward_info = evaluate_simulation(
            simulation=simulation,
            task=tau2_task,
            evaluation_type=EvaluationType.ENV,
            solo_mode=False,
            domain="retail",
            mode=CommunicationMode.HALF_DUPLEX,
            env_kwargs={},
            strict_replay=strict_replay,
            # ENV does not consume communicate info, but the argument is
            # required. SUBSTRING avoids constructing an LLM judge.
            communicate_scoring=CommunicateScoring(
                scorer=CommunicateScorer.SUBSTRING
            ),
        )
    except Exception as exc:
        if strict_replay:
            warnings.append(
                "strict replay failed, retried lenient: "
                f"{type(exc).__name__}: {exc}"
            )
            lenient = score_run(simulation, tau2_task, strict_replay=False)
            lenient["warnings"] = warnings + list(lenient.get("warnings") or [])
            return lenient
        raise

    hashes = _db_hashes(tau2_task, simulation.messages or [], warnings)
    db_check = reward_info.db_check

    return {
        "reward": reward_info.reward,
        "db_match": None if db_check is None else db_check.db_match,
        "db_reward": None if db_check is None else db_check.db_reward,
        "communicate_reward": 1.0,
        "reward_basis": [r.value for r in (reward_info.reward_basis or [])],
        "reward_breakdown": {
            _enum_key(key): value
            for key, value in (reward_info.reward_breakdown or {}).items()
        },
        "termination_reason": simulation.termination_reason.value,
        "info": reward_info.info,
        **hashes,
        "warnings": warnings,
        "official": reward_info.model_dump(mode="json"),
    }


def _enum_key(value: Any) -> str:
    return str(getattr(value, "value", value))


def _db_hashes(tau2_task, messages: list, warnings: list[str]) -> dict[str, Any]:
    """Recompute both final DB hashes for diagnostics.

    This mirrors ``EnvironmentEvaluator`` but always returns both hashes, even
    when the simulation did not terminate with AGENT_STOP.
    """
    from tau2.registry import registry

    constructor = registry.get_env_constructor("retail")

    predicted = constructor(solo_mode=False)
    predicted.set_state(None, None, list(messages), strict=False)

    gold = constructor(solo_mode=False)
    gold.set_state(None, None, [], strict=False)
    for action in tau2_task.evaluation_criteria.actions or []:
        try:
            gold.make_tool_call(
                tool_name=action.name,
                requestor=action.requestor,
                **action.arguments,
            )
        except Exception as exc:
            warnings.append(f"gold action {action.name} failed: {exc}")

    return {
        "predicted_db_hash": predicted.get_db_hash(),
        "gold_db_hash": gold.get_db_hash(),
        "predicted_user_db_hash": predicted.get_user_db_hash(),
        "gold_user_db_hash": gold.get_user_db_hash(),
    }
