"""agent-eval: Lightweight evaluation framework for AI agents."""
__version__ = "0.5.1"

from .trace import Trace, Message
from .diff import diff_traces, TraceDiff
from .assertions import (
    EvalFailure,
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
from .cost import (
    assert_total_tokens,
    assert_total_cost,
    assert_tokens_per_step,
    assert_cost_efficiency,
)
from .consistency import (
    ConsistencyReport,
    assert_consistency,
)
from .safety import (
    assert_no_sensitive_data,
    assert_no_injection_leak,
)
from .judge import (
    JudgeProvider,
    JudgeResult,
    JudgeCost,
    Rubric,
    judge_goal_completion,
    judge_trajectory,
    judge_faithfulness,
    judge_reasoning,
    judge_pairwise,
    create_custom_judge,
)
from .jury import JudgeJury, JuryVerdict

__all__ = [
    # Core
    "Trace", "Message", "diff_traces", "TraceDiff", "EvalFailure",
    # Assertions (v0.1)
    "assert_tool_called", "assert_tool_not_called",
    "assert_no_loop", "assert_max_steps",
    "assert_final_answer_contains", "assert_final_answer_matches",
    "assert_latency", "assert_no_empty_response",
    "assert_tool_call_order", "assert_no_repetition",
    "assert_tool_call_efficiency",
    # Cost (v0.3)
    "assert_total_tokens", "assert_total_cost",
    "assert_tokens_per_step", "assert_cost_efficiency",
    # Consistency (v0.3)
    "ConsistencyReport", "assert_consistency",
    # Safety (v0.3)
    "assert_no_sensitive_data", "assert_no_injection_leak",
    # Judge (v0.4)
    "JudgeProvider", "JudgeResult", "JudgeCost", "Rubric",
    "judge_goal_completion", "judge_trajectory",
    "judge_faithfulness", "judge_reasoning",
    "judge_pairwise",
    "create_custom_judge",
    # Jury (v0.5)
    "JudgeJury", "JuryVerdict",
]
