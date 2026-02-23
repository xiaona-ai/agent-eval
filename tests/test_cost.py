"""Tests for cost tracking assertions."""
import json
import unittest

from agent_eval import Trace
from agent_eval.cost import (
    assert_total_tokens, assert_total_cost,
    assert_tokens_per_step, assert_cost_efficiency,
)
from agent_eval.assertions import EvalFailure


COST_TRACE = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!", "usage": {
        "model": "gpt-4o",
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
    }},
    {"role": "user", "content": "Search for X"},
    {"role": "assistant", "content": None, "tool_calls": [
        {"function": {"name": "search", "arguments": json.dumps({"q": "X"})}}
    ], "usage": {
        "model": "gpt-4o",
        "prompt_tokens": 200,
        "completion_tokens": 30,
        "total_tokens": 230,
    }},
    {"role": "tool", "name": "search", "content": "Results"},
    {"role": "assistant", "content": "Here are the results.", "usage": {
        "model": "gpt-4o",
        "prompt_tokens": 300,
        "completion_tokens": 80,
        "total_tokens": 380,
    }},
]

PRICING = {"gpt-4o": {"input": 2.5, "output": 10.0}}


class TestTotalTokens(unittest.TestCase):
    def test_within_budget(self):
        trace = Trace(COST_TRACE)
        assert_total_tokens(trace, max_tokens=1000)  # 150+230+380=760

    def test_exceeds_budget(self):
        trace = Trace(COST_TRACE)
        with self.assertRaises(EvalFailure) as ctx:
            assert_total_tokens(trace, max_tokens=500)
        self.assertIn("760", str(ctx.exception))

    def test_no_usage_data(self):
        trace = Trace([
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ])
        assert_total_tokens(trace, max_tokens=100)  # 0 tokens, passes

    def test_missing_total_uses_sum(self):
        trace = Trace([
            {"role": "assistant", "content": "Hi", "usage": {
                "prompt_tokens": 50, "completion_tokens": 25
            }},
        ])
        assert_total_tokens(trace, max_tokens=75)
        with self.assertRaises(EvalFailure):
            assert_total_tokens(trace, max_tokens=74)


class TestTotalCost(unittest.TestCase):
    def test_within_budget(self):
        trace = Trace(COST_TRACE)
        assert_total_cost(trace, max_usd=1.0, pricing=PRICING)

    def test_exceeds_budget(self):
        trace = Trace(COST_TRACE)
        # input: (100+200+300)/1M * 2.5 = 0.0015
        # output: (50+30+80)/1M * 10.0 = 0.0016
        # total ≈ 0.0031
        with self.assertRaises(EvalFailure):
            assert_total_cost(trace, max_usd=0.001, pricing=PRICING)

    def test_unknown_model(self):
        trace = Trace([
            {"role": "assistant", "content": "Hi", "usage": {
                "model": "unknown-model",
                "prompt_tokens": 1000000,
                "completion_tokens": 1000000,
            }},
        ])
        # Unknown model → cost = 0
        assert_total_cost(trace, max_usd=0.001, pricing=PRICING)


class TestTokensPerStep(unittest.TestCase):
    def test_within_limit(self):
        trace = Trace(COST_TRACE)
        # 760 tokens / 3 assistant msgs ≈ 253
        assert_tokens_per_step(trace, max_avg=300)

    def test_exceeds_limit(self):
        trace = Trace(COST_TRACE)
        with self.assertRaises(EvalFailure):
            assert_tokens_per_step(trace, max_avg=200)

    def test_no_assistant_messages(self):
        trace = Trace([{"role": "user", "content": "Hi"}])
        assert_tokens_per_step(trace, max_avg=100)  # Skips


class TestCostEfficiency(unittest.TestCase):
    def test_within_limit(self):
        trace = Trace(COST_TRACE)
        assert_cost_efficiency(trace, max_cost_per_tool_call=1.0, pricing=PRICING)

    def test_exceeds_limit(self):
        trace = Trace(COST_TRACE)
        with self.assertRaises(EvalFailure):
            assert_cost_efficiency(trace, max_cost_per_tool_call=0.0001, pricing=PRICING)

    def test_no_tool_calls(self):
        trace = Trace([
            {"role": "assistant", "content": "Hi", "usage": {
                "model": "gpt-4o", "prompt_tokens": 100, "completion_tokens": 50
            }},
        ])
        # No tool calls → skip
        assert_cost_efficiency(trace, max_cost_per_tool_call=0.0001, pricing=PRICING)


if __name__ == "__main__":
    unittest.main()
