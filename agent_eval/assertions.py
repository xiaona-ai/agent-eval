"""Deterministic assertions for agent traces.

Layer 1: Zero dependencies, zero API calls. Pure rule-based checks.
All assertions raise AssertionError with descriptive messages on failure.
"""
import json
import re
from typing import Any, Dict, List, Optional, Sequence, Union

from .trace import Trace, Message


class EvalFailure(AssertionError):
    """An evaluation assertion failure with structured details."""

    def __init__(self, check: str, message: str, details: Optional[dict] = None):
        self.check = check
        self.details = details or {}
        super().__init__(f"[{check}] {message}")


# === Tool Call Assertions ===

def assert_tool_called(
    trace: Trace,
    tool_name: str,
    args: Optional[dict] = None,
    min_times: int = 1,
    max_times: Optional[int] = None,
):
    """Assert a tool was called (optionally with specific args)."""
    calls = []
    for m in trace.tool_calls:
        if tool_name in m.tool_names:
            calls.append(m)

    count = len(calls)
    if count < min_times:
        raise EvalFailure(
            "tool_called",
            f"Expected '{tool_name}' to be called at least {min_times} time(s), "
            f"but was called {count} time(s). Tools called: {trace.all_tool_names}",
            {"tool": tool_name, "expected_min": min_times, "actual": count},
        )

    if max_times is not None and count > max_times:
        raise EvalFailure(
            "tool_called",
            f"Expected '{tool_name}' to be called at most {max_times} time(s), "
            f"but was called {count} time(s).",
            {"tool": tool_name, "expected_max": max_times, "actual": count},
        )

    if args is not None:
        matched = False
        for m in calls:
            actual_args = m.tool_args(tool_name)
            if actual_args is not None and _dict_contains(actual_args, args):
                matched = True
                break
        if not matched:
            raise EvalFailure(
                "tool_called_args",
                f"'{tool_name}' was called but not with expected args {args}.",
                {"tool": tool_name, "expected_args": args},
            )


def assert_tool_not_called(trace: Trace, tool_name: str):
    """Assert a tool was NOT called."""
    if tool_name in trace.all_tool_names:
        raise EvalFailure(
            "tool_not_called",
            f"Expected '{tool_name}' to NOT be called, but it was.",
            {"tool": tool_name, "all_tools": trace.all_tool_names},
        )


def assert_tool_call_order(trace: Trace, expected_order: List[str]):
    """Assert tools were called in a specific order (subsequence match)."""
    actual = trace.all_tool_names
    idx = 0
    for tool in expected_order:
        found = False
        while idx < len(actual):
            if actual[idx] == tool:
                found = True
                idx += 1
                break
            idx += 1
        if not found:
            raise EvalFailure(
                "tool_call_order",
                f"Expected tool call order {expected_order}, "
                f"but actual order was {actual}. "
                f"'{tool}' not found in expected position.",
                {"expected": expected_order, "actual": actual},
            )


# === Control Flow Assertions ===

def assert_no_loop(trace: Trace, max_repeats: int = 3):
    """Assert the agent didn't get stuck in a loop (same tool called repeatedly)."""
    tools = trace.all_tool_names
    if len(tools) < max_repeats:
        return

    for i in range(len(tools) - max_repeats + 1):
        window = tools[i:i + max_repeats]
        if len(set(window)) == 1:
            raise EvalFailure(
                "no_loop",
                f"Detected loop: '{window[0]}' called {max_repeats}+ times consecutively.",
                {"tool": window[0], "max_repeats": max_repeats, "position": i},
            )


def assert_max_steps(trace: Trace, max_steps: int):
    """Assert the agent completed within a step limit."""
    actual = trace.step_count
    if actual > max_steps:
        raise EvalFailure(
            "max_steps",
            f"Agent took {actual} steps, exceeding limit of {max_steps}.",
            {"actual": actual, "max_steps": max_steps},
        )


# === Output Assertions ===

def assert_final_answer_contains(
    trace: Trace,
    text: str,
    case_sensitive: bool = False,
):
    """Assert the final response contains specific text."""
    final = trace.final_response
    content = final.text_content if final else ""
    if not content.strip():
        raise EvalFailure(
            "final_answer_contains",
            "No final assistant response found in trace.",
        )

    check_text = text
    if not case_sensitive:
        content = content.lower()
        check_text = text.lower()

    if check_text not in content:
        raise EvalFailure(
            "final_answer_contains",
            f"Final answer does not contain '{text}'. "
            f"Got: '{content[:200]}'",
            {"expected_text": text, "actual": content[:500]},
        )


def assert_final_answer_matches(trace: Trace, pattern: str):
    """Assert the final response matches a regex pattern."""
    final = trace.final_response
    content = final.text_content if final else ""
    if not content.strip():
        raise EvalFailure("final_answer_matches", "No final assistant response found.")

    if not re.search(pattern, content):
        raise EvalFailure(
            "final_answer_matches",
            f"Final answer does not match pattern '{pattern}'. "
            f"Got: '{content[:200]}'",
            {"pattern": pattern, "actual": content[:500]},
        )


def assert_no_empty_response(trace: Trace):
    """Assert no assistant message has empty/null content (excluding tool calls)."""
    for i, m in enumerate(trace.messages):
        if m.is_assistant and not m.is_tool_call:
            if not m.text_content.strip():
                raise EvalFailure(
                    "no_empty_response",
                    f"Empty assistant response at message index {i}.",
                    {"index": i},
                )


# === Performance Assertions ===

def assert_latency(trace: Trace, max_seconds: float):
    """Assert total trace latency is within bounds."""
    total = trace.total_latency_ms
    if total is None:
        return  # No latency data, skip
    max_ms = max_seconds * 1000
    if total > max_ms:
        raise EvalFailure(
            "latency",
            f"Total latency {total:.0f}ms exceeds limit of {max_ms:.0f}ms ({max_seconds}s).",
            {"actual_ms": total, "max_ms": max_ms},
        )


# === Repetition Detection ===

def assert_no_repetition(trace: Trace, threshold: float = 0.85):
    """Assert no two consecutive assistant responses are near-identical.

    Uses simple token overlap ratio (Jaccard similarity).
    """
    responses = [
        m.text_content
        for m in trace.messages
        if m.is_assistant and not m.is_tool_call and m.text_content.strip()
    ]

    for i in range(1, len(responses)):
        sim = _jaccard_similarity(responses[i - 1], responses[i])
        if sim >= threshold:
            raise EvalFailure(
                "no_repetition",
                f"Consecutive responses at positions {i-1} and {i} are {sim:.0%} similar "
                f"(threshold: {threshold:.0%}).",
                {"index_a": i - 1, "index_b": i, "similarity": sim},
            )


def assert_tool_call_efficiency(trace: Trace, max_redundant: int = 1):
    """Assert the agent doesn't make too many redundant tool calls.

    A redundant call = same tool with same args called more than once.
    """
    seen = []
    redundant = 0
    for m in trace.tool_calls:
        for name in m.tool_names:
            args = m.tool_args(name)
            try:
                key = (name, json.dumps(args, sort_keys=True) if args else "")
            except TypeError as exc:
                raise EvalFailure(
                    "tool_call_efficiency",
                    f"Tool '{name}' has non-serializable arguments.",
                    {"tool": name, "args_type": type(args).__name__},
                ) from exc
            if key in seen:
                redundant += 1
            else:
                seen.append(key)

    if redundant > max_redundant:
        raise EvalFailure(
            "tool_call_efficiency",
            f"Found {redundant} redundant tool call(s), exceeding limit of {max_redundant}.",
            {"redundant": redundant, "max_redundant": max_redundant},
        )


# === Helpers ===

def _dict_contains(actual: dict, expected: dict) -> bool:
    """Check if actual dict contains all key-value pairs from expected."""
    for k, v in expected.items():
        if k not in actual or actual[k] != v:
            return False
    return True


def _jaccard_similarity(a: str, b: str) -> float:
    """Token-level Jaccard similarity."""
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)
