"""agent-eval: Lightweight evaluation framework for AI agents."""
__version__ = "0.1.0"

from .trace import Trace, Message
from .assertions import (
    assert_tool_called,
    assert_tool_not_called,
    assert_no_loop,
    assert_max_steps,
    assert_final_answer_contains,
    assert_latency,
    assert_no_empty_response,
    assert_tool_call_order,
)

__all__ = [
    "Trace",
    "Message",
    "assert_tool_called",
    "assert_tool_not_called",
    "assert_no_loop",
    "assert_max_steps",
    "assert_final_answer_contains",
    "assert_latency",
    "assert_no_empty_response",
    "assert_tool_call_order",
]
