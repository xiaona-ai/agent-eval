"""Tests for trace diff and regression detection."""
import json
import unittest

from agent_eval import Trace
from agent_eval.diff import diff_traces, TraceDiff


BASELINE = [
    {"role": "user", "content": "Weather in SF?"},
    {"role": "assistant", "tool_calls": [
        {"function": {"name": "get_weather", "arguments": '{"city": "SF"}'}}
    ], "latency_ms": 500},
    {"role": "tool", "name": "get_weather", "content": "80°F sunny"},
    {"role": "assistant", "content": "It's 80°F and sunny in SF.", "latency_ms": 300},
]


class TestDiffNoChanges(unittest.TestCase):
    def test_identical_traces(self):
        a = Trace.from_messages(BASELINE)
        b = Trace.from_messages(BASELINE)
        d = diff_traces(a, b)
        self.assertFalse(d.has_changes)
        self.assertFalse(d.is_regression)
        self.assertIn("No changes", d.summary())


class TestDiffToolChanges(unittest.TestCase):
    def test_tool_added(self):
        current = BASELINE.copy()
        current.insert(3, {"role": "assistant", "tool_calls": [
            {"function": {"name": "log_result", "arguments": "{}"}}
        ]})
        current.insert(4, {"role": "tool", "name": "log_result", "content": "logged"})
        d = diff_traces(Trace.from_messages(BASELINE), Trace.from_messages(current))
        self.assertTrue(d.has_changes)
        types = [tc["type"] for tc in d.tool_changes]
        self.assertIn("added", types)

    def test_tool_removed_is_regression(self):
        current = [
            {"role": "user", "content": "Weather in SF?"},
            {"role": "assistant", "content": "I don't know.", "latency_ms": 200},
        ]
        d = diff_traces(Trace.from_messages(BASELINE), Trace.from_messages(current))
        self.assertTrue(d.is_regression)
        self.assertIn("removed", d.summary())

    def test_tool_count_changed(self):
        current = BASELINE + [
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "get_weather", "arguments": '{"city": "SF"}'}}
            ]},
            {"role": "tool", "name": "get_weather", "content": "80°F"},
            {"role": "assistant", "content": "Still sunny."},
        ]
        d = diff_traces(Trace.from_messages(BASELINE), Trace.from_messages(current))
        types = [tc["type"] for tc in d.tool_changes]
        self.assertIn("count_changed", types)


class TestDiffOutput(unittest.TestCase):
    def test_answer_changed(self):
        current = [
            {"role": "user", "content": "Weather in SF?"},
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "get_weather", "arguments": '{"city": "SF"}'}}
            ]},
            {"role": "tool", "name": "get_weather", "content": "50°F rainy"},
            {"role": "assistant", "content": "It's 50°F and rainy in SF."},
        ]
        d = diff_traces(Trace.from_messages(BASELINE), Trace.from_messages(current))
        self.assertTrue(any(oc["type"] == "final_answer_changed" for oc in d.output_changes))


class TestDiffPerformance(unittest.TestCase):
    def test_latency_increase(self):
        current = [
            {"role": "user", "content": "Weather?"},
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "get_weather", "arguments": '{"city": "SF"}'}}
            ], "latency_ms": 2000},
            {"role": "tool", "name": "get_weather", "content": "80°F"},
            {"role": "assistant", "content": "Sunny.", "latency_ms": 1000},
        ]
        d = diff_traces(Trace.from_messages(BASELINE), Trace.from_messages(current))
        self.assertTrue(any(pc["type"] == "latency_increase" for pc in d.performance_changes))

    def test_big_latency_increase_is_regression(self):
        current = [
            {"role": "user", "content": "Weather?"},
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "get_weather", "arguments": '{"city": "SF"}'}}
            ], "latency_ms": 5000},
            {"role": "tool", "name": "get_weather", "content": "80°F"},
            {"role": "assistant", "content": "Sunny.", "latency_ms": 3000},
        ]
        d = diff_traces(Trace.from_messages(BASELINE), Trace.from_messages(current))
        # 8000ms vs 800ms = 10x, should be regression
        self.assertTrue(d.is_regression)


class TestDiffStepCount(unittest.TestCase):
    def test_step_count_change(self):
        current = [
            {"role": "user", "content": "Weather?"},
            {"role": "assistant", "content": "It's sunny."},
        ]
        d = diff_traces(Trace.from_messages(BASELINE), Trace.from_messages(current))
        self.assertIsNotNone(d.step_count_change)
        self.assertEqual(d.step_count_change, (2, 1))


if __name__ == "__main__":
    unittest.main()
