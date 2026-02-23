"""Token and cost tracking assertions for agent traces.

Layer 3: Cost efficiency evaluation — a gap no competitor covers.
Zero dependencies. All pricing data supplied by the user.
"""
from typing import Any, Dict, List, Optional

from .trace import Trace
from .assertions import EvalFailure


def _get_usage(msg) -> Optional[dict]:
    """Extract usage dict from a message."""
    return getattr(msg, "usage", None)


def _get_model(msg) -> Optional[str]:
    """Extract model name from usage or metadata."""
    usage = _get_usage(msg)
    if usage and isinstance(usage, dict):
        model = usage.get("model")
        if model:
            return model
    return msg.metadata.get("model")


def _parse_int(value: Any, field: str) -> int:
    """Parse an integer usage field and raise EvalFailure on malformed values."""
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise EvalFailure(
            "usage_malformed",
            f"Usage field '{field}' must be an integer-compatible value, got {value!r}.",
            {"field": field, "value": value},
        ) from exc


def _total_tokens_from_usage(usage: dict) -> int:
    """Get total tokens from a usage dict."""
    total = usage.get("total_tokens")
    if total is not None:
        return _parse_int(total, "total_tokens")
    prompt = usage.get("prompt_tokens", 0) or 0
    completion = usage.get("completion_tokens", 0) or 0
    return _parse_int(prompt, "prompt_tokens") + _parse_int(completion, "completion_tokens")


def _compute_cost(
    msg,
    pricing: Dict[str, Dict[str, float]],
    strict: bool = False,
) -> float:
    """Compute cost for a single message given pricing table.

    pricing format: {"model-name": {"input": price_per_1M, "output": price_per_1M}}
    """
    usage = _get_usage(msg)
    if not usage or not isinstance(usage, dict):
        return 0.0

    model = _get_model(msg)
    if not model or model not in pricing:
        if strict:
            raise EvalFailure(
                "total_cost_unknown_model",
                f"Model {model!r} missing from pricing table in strict mode.",
                {"model": model, "known_models": sorted(pricing.keys())},
            )
        return 0.0

    rates = pricing[model]
    prompt = _parse_int(usage.get("prompt_tokens", 0) or 0, "prompt_tokens")
    completion = _parse_int(usage.get("completion_tokens", 0) or 0, "completion_tokens")

    input_cost = (prompt / 1_000_000) * rates.get("input", 0)
    output_cost = (completion / 1_000_000) * rates.get("output", 0)
    return input_cost + output_cost


def _sum_tokens(trace: Trace) -> int:
    """Sum total tokens across all messages with usage data."""
    total = 0
    for m in trace.messages:
        usage = _get_usage(m)
        if usage and isinstance(usage, dict):
            total += _total_tokens_from_usage(usage)
    return total


def _sum_cost(
    trace: Trace,
    pricing: Dict[str, Dict[str, float]],
    strict: bool = False,
) -> float:
    """Sum cost across all messages."""
    return sum(_compute_cost(m, pricing, strict=strict) for m in trace.messages)


def assert_total_tokens(trace: Trace, max_tokens: int):
    """Assert total token usage is within budget.

    Sums usage.total_tokens (or prompt_tokens + completion_tokens) across
    all messages that have a usage field.

    Raises EvalFailure if total exceeds max_tokens.
    """
    total = _sum_tokens(trace)
    if total > max_tokens:
        raise EvalFailure(
            "total_tokens",
            f"Total tokens {total} exceeds limit of {max_tokens}.",
            {"actual": total, "max_tokens": max_tokens},
        )


def assert_total_cost(
    trace: Trace,
    max_usd: float,
    pricing: Dict[str, Dict[str, float]],
    strict: bool = False,
):
    """Assert total cost is within budget.

    Args:
        trace: The agent trace.
        max_usd: Maximum allowed cost in USD.
        pricing: Model pricing table, e.g.
            {"gpt-4o": {"input": 2.5, "output": 10.0}}
            Prices are per 1M tokens.
        strict: If True, raise EvalFailure when a message has a model missing
            from pricing. If False, unknown models contribute $0.

    Raises EvalFailure if total cost exceeds max_usd.
    """
    total = _sum_cost(trace, pricing, strict=strict)
    if total > max_usd:
        raise EvalFailure(
            "total_cost",
            f"Total cost ${total:.6f} exceeds limit of ${max_usd:.6f}.",
            {"actual_usd": total, "max_usd": max_usd},
        )


def assert_tokens_per_step(trace: Trace, max_avg: float):
    """Assert average tokens per assistant message is within limit.

    Raises EvalFailure if average exceeds max_avg.
    """
    assistant_msgs = trace.assistant_messages
    if not assistant_msgs:
        return  # No assistant messages, nothing to check

    total = _sum_tokens(trace)
    avg = total / len(assistant_msgs)
    if avg > max_avg:
        raise EvalFailure(
            "tokens_per_step",
            f"Average tokens per step {avg:.1f} exceeds limit of {max_avg:.1f}.",
            {"actual_avg": avg, "max_avg": max_avg, "steps": len(assistant_msgs)},
        )


def assert_cost_efficiency(
    trace: Trace,
    max_cost_per_tool_call: float,
    pricing: Dict[str, Dict[str, float]],
    strict: bool = False,
):
    """Assert cost per tool call is within limit.

    Raises EvalFailure if (total_cost / num_tool_calls) exceeds limit.
    Skips if no tool calls were made.
    """
    tool_calls = trace.tool_calls
    if not tool_calls:
        return  # No tool calls, skip

    total_cost = _sum_cost(trace, pricing, strict=strict)
    num_calls = sum(len(m.tool_names) for m in tool_calls)
    if num_calls == 0:
        return

    cost_per_call = total_cost / num_calls
    if cost_per_call > max_cost_per_tool_call:
        raise EvalFailure(
            "cost_efficiency",
            f"Cost per tool call ${cost_per_call:.6f} exceeds limit of "
            f"${max_cost_per_tool_call:.6f}.",
            {
                "cost_per_call": cost_per_call,
                "max_cost_per_call": max_cost_per_tool_call,
                "total_cost": total_cost,
                "num_tool_calls": num_calls,
            },
        )
