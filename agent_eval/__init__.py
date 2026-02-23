"""agent-eval: Lightweight evaluation framework for AI agents."""
__version__ = "0.2.0"

from .trace import Trace, Message
from .diff import diff_traces, TraceDiff
from .assertions import (
    assert_tool_called,
    assert_tool_not_called,
    assert_no_loop,
    assert_max_steps,
    assert_final_answer_contains,
    assert_latency,
    assert_no_empty_response,
    assert_tool_call_order,
    assert_no_repetition,
    assert_tool_call_efficiency,
    assert_final_answer_matches,
)

__all__ = [
    "Trace", "Message", "diff_traces", "TraceDiff",
    "assert_tool_called", "assert_tool_not_called",
    "assert_no_loop", "assert_max_steps",
    "assert_final_answer_contains", "assert_final_answer_matches",
    "assert_latency", "assert_no_empty_response",
    "assert_tool_call_order", "assert_no_repetition",
    "assert_tool_call_efficiency",
]
