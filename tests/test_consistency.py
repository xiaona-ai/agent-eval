"""Tests for consistency evaluation."""
import json
import unittest

from agent_eval import Trace
from agent_eval.consistency import (
    ConsistencyReport, assert_consistency,
    _levenshtein, _normalized_similarity,
)
from agent_eval.assertions import EvalFailure


# Identical traces
TRACE_A = [
    {"role": "user", "content": "What's the weather?"},
    {"role": "assistant", "tool_calls": [
        {"function": {"name": "get_weather", "arguments": json.dumps({"city": "SF"})}}
    ]},
    {"role": "tool", "name": "get_weather", "content": "Sunny 80F"},
    {"role": "assistant", "content": "The weather in SF is sunny and 80F."},
]

# Same tools, slightly different answer
TRACE_B = [
    {"role": "user", "content": "What's the weather?"},
    {"role": "assistant", "tool_calls": [
        {"function": {"name": "get_weather", "arguments": json.dumps({"city": "SF"})}}
    ]},
    {"role": "tool", "name": "get_weather", "content": "Sunny 80F"},
    {"role": "assistant", "content": "It's sunny and 80 degrees in San Francisco."},
]

# Different tools
TRACE_C = [
    {"role": "user", "content": "What's the weather?"},
    {"role": "assistant", "tool_calls": [
        {"function": {"name": "search_web", "arguments": json.dumps({"q": "SF weather"})}}
    ]},
    {"role": "tool", "name": "search_web", "content": "80F sunny"},
    {"role": "assistant", "tool_calls": [
        {"function": {"name": "get_weather", "arguments": json.dumps({"city": "SF"})}}
    ]},
    {"role": "tool", "name": "get_weather", "content": "Sunny 80F"},
    {"role": "assistant", "content": "Based on my research, SF is sunny at 80F today."},
]


class TestLevenshtein(unittest.TestCase):
    def test_identical(self):
        self.assertEqual(_levenshtein(["a", "b", "c"], ["a", "b", "c"]), 0)

    def test_empty(self):
        self.assertEqual(_levenshtein([], ["a", "b"]), 2)
        self.assertEqual(_levenshtein(["a"], []), 1)

    def test_different(self):
        self.assertEqual(_levenshtein(["a", "b"], ["b", "a"]), 2)

    def test_insertion(self):
        self.assertEqual(_levenshtein(["a", "c"], ["a", "b", "c"]), 1)


class TestNormalizedSimilarity(unittest.TestCase):
    def test_identical(self):
        self.assertAlmostEqual(_normalized_similarity(["a", "b"], ["a", "b"]), 1.0)

    def test_empty_both(self):
        self.assertAlmostEqual(_normalized_similarity([], []), 1.0)

    def test_completely_different(self):
        sim = _normalized_similarity(["a"], ["b"])
        self.assertAlmostEqual(sim, 0.0)


class TestConsistencyReport(unittest.TestCase):
    def test_identical_traces(self):
        traces = [Trace(TRACE_A), Trace(TRACE_A)]
        report = ConsistencyReport(traces)
        self.assertAlmostEqual(report.tool_call_consistency, 1.0)
        self.assertAlmostEqual(report.final_answer_consistency, 1.0)
        self.assertAlmostEqual(report.step_count_variance, 0.0)

    def test_similar_traces(self):
        traces = [Trace(TRACE_A), Trace(TRACE_B)]
        report = ConsistencyReport(traces)
        self.assertAlmostEqual(report.tool_call_consistency, 1.0)  # Same tools
        self.assertGreater(report.final_answer_consistency, 0.1)  # Similar answers (Jaccard on short text)
        self.assertAlmostEqual(report.step_count_variance, 0.0)  # Same steps

    def test_different_traces(self):
        traces = [Trace(TRACE_A), Trace(TRACE_C)]
        report = ConsistencyReport(traces)
        self.assertLess(report.tool_call_consistency, 1.0)  # Different tool sequences
        self.assertGreater(report.step_count_variance, 0.0)  # Different step counts

    def test_summary(self):
        traces = [Trace(TRACE_A), Trace(TRACE_A)]
        report = ConsistencyReport(traces)
        summary = report.summary()
        self.assertIn("Consistency Report", summary)
        self.assertIn("2 traces", summary)

    def test_minimum_traces(self):
        with self.assertRaises(ValueError):
            ConsistencyReport([Trace(TRACE_A)])

    def test_three_traces(self):
        traces = [Trace(TRACE_A), Trace(TRACE_B), Trace(TRACE_C)]
        report = ConsistencyReport(traces)
        # Should work with 3 traces (3 pairs)
        self.assertGreater(report.tool_call_consistency, 0.0)
        self.assertLessEqual(report.tool_call_consistency, 1.0)


class TestAssertConsistency(unittest.TestCase):
    def test_passes_high_consistency(self):
        traces = [Trace(TRACE_A), Trace(TRACE_A)]
        assert_consistency(traces, min_tool_consistency=0.9, min_answer_consistency=0.9)

    def test_fails_tool_consistency(self):
        traces = [Trace(TRACE_A), Trace(TRACE_C)]
        with self.assertRaises(EvalFailure) as ctx:
            assert_consistency(traces, min_tool_consistency=0.99)
        self.assertIn("consistency_tools", str(ctx.exception))

    def test_fails_step_variance(self):
        traces = [Trace(TRACE_A), Trace(TRACE_C)]
        with self.assertRaises(EvalFailure):
            assert_consistency(traces, max_step_variance=0.0)

    def test_none_thresholds_skip(self):
        traces = [Trace(TRACE_A), Trace(TRACE_C)]
        # All None → no assertions, passes
        assert_consistency(traces)


if __name__ == "__main__":
    unittest.main()
