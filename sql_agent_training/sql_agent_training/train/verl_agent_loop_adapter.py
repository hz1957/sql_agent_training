"""Import-safe helpers for adapting internal trajectories to VERL AgentLoop APIs."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from sql_agent_training.agent.trace_format import TokenizedTrajectory


@dataclass(frozen=True)
class VerlAgentLoopApi:
    """Resolved VERL Agent Loop API objects."""

    agent_loop_base: Any
    agent_loop_output: Any
    agent_loop_metrics: Any | None = None
    register: Any | None = None
    module_name: str = ""


def resolve_verl_agent_loop_api() -> VerlAgentLoopApi:
    """Resolve VERL AgentLoopBase and AgentLoopOutput.

    Raises:
        RuntimeError: If VERL is not installed or the expected alpha API moved.
    """

    candidates = [
        "verl.experimental.agent_loop.agent_loop",
        "verl.agent_loop.agent_loop",
    ]
    errors: list[str] = []
    for module_name in candidates:
        try:
            module = import_module(module_name)
            return VerlAgentLoopApi(
                agent_loop_base=getattr(module, "AgentLoopBase"),
                agent_loop_output=getattr(module, "AgentLoopOutput"),
                agent_loop_metrics=getattr(module, "AgentLoopMetrics", None),
                register=getattr(module, "register", None),
                module_name=module_name,
            )
        except Exception as exc:  # pragma: no cover - depends on installed VERL
            errors.append(f"{module_name}: {exc}")
    raise RuntimeError("Unable to resolve VERL Agent Loop API. Tried: " + "; ".join(errors))


def to_verl_agent_loop_output(tokenized: TokenizedTrajectory, agent_loop_output_cls: Any) -> Any:
    """Convert an internal TokenizedTrajectory to VERL AgentLoopOutput."""

    tokenized.validate()
    kwargs: dict[str, Any] = {
        "prompt_ids": tokenized.prompt_ids,
        "response_ids": tokenized.response_ids,
        "response_mask": tokenized.response_mask,
    }
    parameters = inspect.signature(agent_loop_output_cls).parameters
    if "reward_score" in parameters:
        kwargs["reward_score"] = tokenized.reward
    if "num_turns" in parameters:
        kwargs["num_turns"] = 0
    if "metrics" in parameters:
        metrics_annotation = parameters["metrics"].annotation
        if metrics_annotation is inspect.Signature.empty or isinstance(metrics_annotation, str):
            module = import_module(agent_loop_output_cls.__module__)
            metrics_cls = getattr(module, "AgentLoopMetrics", None)
        else:
            metrics_cls = metrics_annotation
        kwargs["metrics"] = metrics_cls() if callable(metrics_cls) else {}
    if "extra_fields" in parameters:
        kwargs["extra_fields"] = {
            "uid": tokenized.uid,
            "rollout_id": tokenized.rollout_id,
            "metadata": tokenized.metadata,
        }
    return agent_loop_output_cls(**kwargs)


def describe_verl_agent_loop_api() -> dict[str, Any]:
    """Return a JSON-serializable description of the installed VERL Agent Loop API."""

    api = resolve_verl_agent_loop_api()
    output_fields = {}
    if hasattr(api.agent_loop_output, "model_fields"):
        output_fields = {
            name: {
                "required": field.is_required(),
                "annotation": str(field.annotation),
            }
            for name, field in api.agent_loop_output.model_fields.items()
        }
    return {
        "module": api.module_name,
        "agent_loop_base": f"{api.agent_loop_base.__module__}.{api.agent_loop_base.__qualname__}",
        "agent_loop_output": f"{api.agent_loop_output.__module__}.{api.agent_loop_output.__qualname__}",
        "agent_loop_metrics": (
            f"{api.agent_loop_metrics.__module__}.{api.agent_loop_metrics.__qualname__}"
            if api.agent_loop_metrics is not None
            else None
        ),
        "has_register": api.register is not None,
        "base_init_signature": str(inspect.signature(api.agent_loop_base.__init__)),
        "base_run_signature": str(inspect.signature(api.agent_loop_base.run)),
        "output_fields": output_fields,
    }
