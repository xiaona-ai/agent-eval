"""Consistency evaluation for multi-run agent traces.

Run the same input multiple times, collect traces, measure how consistent
the agent's behavior is. Zero dependencies.
"""
import math
from typing import List, Optional

from .trace import Trace
from .assertions import EvalFailure, _jaccard_similarity


def _levenshtein(a: List[str], b: List[str]) -> int:
    """Compute Levenshtein edit distance between two sequences."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n

    # Use two-row optimization for memory
    prev = list(range(m + 1))
    curr = [0] * (m + 1)

    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,       # deletion
                curr[j - 1] + 1,   # insertion
                prev[j - 1] + cost  # substitution
            )
        prev, curr = curr, prev

    return prev[m]


def _normalized_similarity(a: List[str], b: List[str]) -> float:
    """Normalized similarity (1 - normalized edit distance)."""
    if not a and not b:
        return 1.0
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    dist = _levenshtein(a, b)
    return 1.0 - (dist / max_len)


def _pairwise_avg(values: List, similarity_fn) -> float:
    """Average pairwise similarity."""
    if len(values) < 2:
        return 1.0
    total = 0.0
    count = 0
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            total += similarity_fn(values[i], values[j])
            count += 1
    return total / count if count > 0 else 1.0


class ConsistencyReport:
    """Analyze consistency across multiple runs of the same input.

    Compares tool call sequences, final answers, and step counts
    to measure behavioral stability.
    """

    def __init__(self, traces: List[Trace]):
        if len(traces) < 2:
            raise ValueError("ConsistencyReport requires at least 2 traces.")
        self._traces = traces
        self._tool_consistency: Optional[float] = None
        self._answer_consistency: Optional[float] = None
        self._step_variance: Optional[float] = None

    @property
    def tool_call_consistency(self) -> float:
        """Average pairwise normalized similarity of tool call sequences (0-1)."""
        if self._tool_consistency is None:
            sequences = [t.all_tool_names for t in self._traces]
            self._tool_consistency = _pairwise_avg(sequences, _normalized_similarity)
        return self._tool_consistency

    @property
    def final_answer_consistency(self) -> float:
        """Average pairwise Jaccard similarity of final answers (0-1)."""
        if self._answer_consistency is None:
            answers = []
            for t in self._traces:
                final = t.final_response
                answers.append(final.text_content if final else "")
            self._answer_consistency = _pairwise_avg(
                answers,
                lambda a, b: _jaccard_similarity(a, b)
            )
        return self._answer_consistency

    @property
    def step_count_variance(self) -> float:
        """Standard deviation of step counts across traces."""
        if self._step_variance is None:
            counts = [t.step_count for t in self._traces]
            mean = sum(counts) / len(counts)
            variance = sum((c - mean) ** 2 for c in counts) / len(counts)
            self._step_variance = math.sqrt(variance)
        return self._step_variance

    def summary(self) -> str:
        """Human-readable consistency report."""
        lines = [
            f"Consistency Report ({len(self._traces)} traces)",
            f"  Tool call consistency: {self.tool_call_consistency:.1%}",
            f"  Final answer consistency: {self.final_answer_consistency:.1%}",
            f"  Step count std dev: {self.step_count_variance:.2f}",
        ]

        # Overall assessment
        tc = self.tool_call_consistency
        ac = self.final_answer_consistency
        if tc >= 0.9 and ac >= 0.9:
            lines.append("  ✅ Highly consistent")
        elif tc >= 0.7 and ac >= 0.7:
            lines.append("  ⚠️ Moderately consistent")
        else:
            lines.append("  ❌ Low consistency — investigate variance sources")

        return "\n".join(lines)


def assert_consistency(
    traces: List[Trace],
    min_tool_consistency: Optional[float] = None,
    min_answer_consistency: Optional[float] = None,
    max_step_variance: Optional[float] = None,
):
    """Assert multi-run consistency meets thresholds.

    Args:
        traces: List of Trace objects from multiple runs of the same input.
        min_tool_consistency: Minimum tool call sequence similarity (0-1).
        min_answer_consistency: Minimum final answer similarity (0-1).
        max_step_variance: Maximum allowed step count standard deviation.

    Raises EvalFailure if any threshold is violated.
    """
    report = ConsistencyReport(traces)

    if min_tool_consistency is not None:
        actual = report.tool_call_consistency
        if actual < min_tool_consistency:
            raise EvalFailure(
                "consistency_tools",
                f"Tool call consistency {actual:.1%} below threshold "
                f"{min_tool_consistency:.1%}.",
                {"actual": actual, "threshold": min_tool_consistency},
            )

    if min_answer_consistency is not None:
        actual = report.final_answer_consistency
        if actual < min_answer_consistency:
            raise EvalFailure(
                "consistency_answers",
                f"Final answer consistency {actual:.1%} below threshold "
                f"{min_answer_consistency:.1%}.",
                {"actual": actual, "threshold": min_answer_consistency},
            )

    if max_step_variance is not None:
        actual = report.step_count_variance
        if actual > max_step_variance:
            raise EvalFailure(
                "consistency_steps",
                f"Step count std dev {actual:.2f} exceeds limit of "
                f"{max_step_variance:.2f}.",
                {"actual": actual, "threshold": max_step_variance},
            )
