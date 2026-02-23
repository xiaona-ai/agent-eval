"""Tests for agent-eval core: Trace, Message, and Layer 1 assertions."""
import json
import tempfile
import unittest
from pathlib import Path

from agent_eval import (
    Trace, Message,
    assert_tool_called, assert_tool_not_called,
    assert_no_loop, assert_max_steps,
    assert_final_answer_contains, assert_latency,
    assert_no_empty_response, assert_tool_call_order,
)
from agent_eval.assertions import (
    EvalFailure, assert_no_repetition, assert_tool_call_efficiency,
    assert_final_answer_matches,
)


# === Sample traces ===

WEATHER_TRACE = [
    {"role": "user", "content": "What's the weather in SF?"},
    {"role": "assistant", "content": None, "tool_calls": [
        {"function": {"name": "get_weather", "arguments": json.dumps({"city": "SF"})}}
    ], "latency_ms": 500},
    {"role": "tool", "name": "get_weather", "content": "80°F and sunny"},
    {"role": "assistant", "content": "The weather in SF is 80°F and sunny.", "latency_ms": 300},
]

MULTI_TOOL_TRACE = [
    {"role": "user", "content": "Search for X then summarize"},
    {"role": "assistant", "tool_calls": [
        {"function": {"name": "web_search", "arguments": json.dumps({"query": "X"})}}
    ]},
    {"role": "tool", "name": "web_search", "content": "Results..."},
    {"role": "assistant", "tool_calls": [
        {"function": {"name": "read_page", "arguments": json.dumps({"url": "http://x.com"})}}
    ]},
    {"role": "tool", "name": "read_page", "content": "Page content..."},
    {"role": "assistant", "content": "Here is a summary of X."},
]

LOOP_TRACE = [
    {"role": "user", "content": "Do something"},
    {"role": "assistant", "tool_calls": [{"function": {"name": "retry_action", "arguments": "{}"}}]},
    {"role": "tool", "name": "retry_action", "content": "failed"},
    {"role": "assistant", "tool_calls": [{"function": {"name": "retry_action", "arguments": "{}"}}]},
    {"role": "tool", "name": "retry_action", "content": "failed"},
    {"role": "assistant", "tool_calls": [{"function": {"name": "retry_action", "arguments": "{}"}}]},
    {"role": "tool", "name": "retry_action", "content": "failed"},
    {"role": "assistant", "content": "I couldn't complete the action."},
]


class TestMessage(unittest.TestCase):
    def test_basic_properties(self):
        m = Message({"role": "assistant", "content": "Hello"})
        self.assertTrue(m.is_assistant)
        self.assertFalse(m.is_tool_call)
        self.assertFalse(m.is_user)

    def test_tool_call_message(self):
        m = Message(WEATHER_TRACE[1])
        self.assertTrue(m.is_tool_call)
        self.assertEqual(m.tool_names, ["get_weather"])
        args = m.tool_args("get_weather")
        self.assertEqual(args, {"city": "SF"})

    def test_tool_response(self):
        m = Message({"role": "tool", "name": "get_weather", "content": "sunny"})
        self.assertTrue(m.is_tool_response)

    def test_to_dict_roundtrip(self):
        original = WEATHER_TRACE[1]
        m = Message(original)
        d = m.to_dict()
        self.assertEqual(d["role"], "assistant")
        self.assertIsNotNone(d.get("tool_calls"))


class TestTrace(unittest.TestCase):
    def test_from_messages(self):
        t = Trace.from_messages(WEATHER_TRACE)
        self.assertEqual(len(t), 4)
        self.assertEqual(t.step_count, 2)

    def test_tool_names(self):
        t = Trace.from_messages(WEATHER_TRACE)
        self.assertEqual(t.all_tool_names, ["get_weather"])

    def test_final_response(self):
        t = Trace.from_messages(WEATHER_TRACE)
        final = t.final_response
        self.assertIsNotNone(final)
        self.assertIn("sunny", final.content)

    def test_total_latency(self):
        t = Trace.from_messages(WEATHER_TRACE)
        self.assertEqual(t.total_latency_ms, 800)

    def test_jsonl_roundtrip(self):
        t = Trace.from_messages(WEATHER_TRACE)
        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
            path = f.name
        t.to_jsonl(path)
        t2 = Trace.from_jsonl(path)
        self.assertEqual(len(t2), len(t))
        self.assertEqual(t2.all_tool_names, t.all_tool_names)
        Path(path).unlink()

    def test_iteration(self):
        t = Trace.from_messages(WEATHER_TRACE)
        roles = [m.role for m in t]
        self.assertEqual(roles, ["user", "assistant", "tool", "assistant"])


class TestToolAssertions(unittest.TestCase):
    def test_tool_called_passes(self):
        t = Trace.from_messages(WEATHER_TRACE)
        assert_tool_called(t, "get_weather")  # should not raise

    def test_tool_called_fails(self):
        t = Trace.from_messages(WEATHER_TRACE)
        with self.assertRaises(EvalFailure) as ctx:
            assert_tool_called(t, "send_email")
        self.assertIn("send_email", str(ctx.exception))

    def test_tool_called_with_args(self):
        t = Trace.from_messages(WEATHER_TRACE)
        assert_tool_called(t, "get_weather", args={"city": "SF"})

    def test_tool_called_wrong_args(self):
        t = Trace.from_messages(WEATHER_TRACE)
        with self.assertRaises(EvalFailure):
            assert_tool_called(t, "get_weather", args={"city": "NYC"})

    def test_tool_called_count(self):
        t = Trace.from_messages(LOOP_TRACE)
        assert_tool_called(t, "retry_action", min_times=3)
        with self.assertRaises(EvalFailure):
            assert_tool_called(t, "retry_action", max_times=2)

    def test_tool_not_called(self):
        t = Trace.from_messages(WEATHER_TRACE)
        assert_tool_not_called(t, "delete_database")

    def test_tool_not_called_fails(self):
        t = Trace.from_messages(WEATHER_TRACE)
        with self.assertRaises(EvalFailure):
            assert_tool_not_called(t, "get_weather")

    def test_tool_call_order(self):
        t = Trace.from_messages(MULTI_TOOL_TRACE)
        assert_tool_call_order(t, ["web_search", "read_page"])

    def test_tool_call_order_fails(self):
        t = Trace.from_messages(MULTI_TOOL_TRACE)
        with self.assertRaises(EvalFailure):
            assert_tool_call_order(t, ["read_page", "web_search"])


class TestControlFlowAssertions(unittest.TestCase):
    def test_no_loop_passes(self):
        t = Trace.from_messages(WEATHER_TRACE)
        assert_no_loop(t)

    def test_no_loop_fails(self):
        t = Trace.from_messages(LOOP_TRACE)
        with self.assertRaises(EvalFailure) as ctx:
            assert_no_loop(t, max_repeats=3)
        self.assertIn("retry_action", str(ctx.exception))

    def test_max_steps_passes(self):
        t = Trace.from_messages(WEATHER_TRACE)
        assert_max_steps(t, 5)

    def test_max_steps_fails(self):
        t = Trace.from_messages(WEATHER_TRACE)
        with self.assertRaises(EvalFailure):
            assert_max_steps(t, 1)


class TestOutputAssertions(unittest.TestCase):
    def test_final_answer_contains(self):
        t = Trace.from_messages(WEATHER_TRACE)
        assert_final_answer_contains(t, "sunny")
        assert_final_answer_contains(t, "SUNNY")  # case insensitive

    def test_final_answer_contains_case_sensitive(self):
        t = Trace.from_messages(WEATHER_TRACE)
        with self.assertRaises(EvalFailure):
            assert_final_answer_contains(t, "SUNNY", case_sensitive=True)

    def test_final_answer_contains_fails(self):
        t = Trace.from_messages(WEATHER_TRACE)
        with self.assertRaises(EvalFailure):
            assert_final_answer_contains(t, "rainy")

    def test_final_answer_matches(self):
        t = Trace.from_messages(WEATHER_TRACE)
        assert_final_answer_matches(t, r"\d+°F")

    def test_no_empty_response(self):
        t = Trace.from_messages(WEATHER_TRACE)
        assert_no_empty_response(t)

    def test_no_empty_response_fails(self):
        trace = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": ""},
        ]
        t = Trace.from_messages(trace)
        with self.assertRaises(EvalFailure):
            assert_no_empty_response(t)


class TestPerformanceAssertions(unittest.TestCase):
    def test_latency_passes(self):
        t = Trace.from_messages(WEATHER_TRACE)
        assert_latency(t, max_seconds=1.0)

    def test_latency_fails(self):
        t = Trace.from_messages(WEATHER_TRACE)
        with self.assertRaises(EvalFailure):
            assert_latency(t, max_seconds=0.5)

    def test_latency_no_data(self):
        t = Trace.from_messages([{"role": "user", "content": "hi"}])
        assert_latency(t, max_seconds=1.0)  # no data = skip


class TestRepetitionAssertions(unittest.TestCase):
    def test_no_repetition_passes(self):
        t = Trace.from_messages(WEATHER_TRACE)
        assert_no_repetition(t)

    def test_no_repetition_fails(self):
        trace = [
            {"role": "assistant", "content": "The weather is sunny and warm today"},
            {"role": "assistant", "content": "The weather is sunny and warm today"},
        ]
        t = Trace.from_messages(trace)
        with self.assertRaises(EvalFailure):
            assert_no_repetition(t, threshold=0.8)


class TestEfficiencyAssertions(unittest.TestCase):
    def test_efficiency_passes(self):
        t = Trace.from_messages(WEATHER_TRACE)
        assert_tool_call_efficiency(t)

    def test_efficiency_fails(self):
        trace = [
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "search", "arguments": json.dumps({"q": "test"})}}
            ]},
            {"role": "tool", "content": "result"},
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "search", "arguments": json.dumps({"q": "test"})}}
            ]},
            {"role": "tool", "content": "result"},
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "search", "arguments": json.dumps({"q": "test"})}}
            ]},
            {"role": "tool", "content": "result"},
            {"role": "assistant", "content": "Done"},
        ]
        t = Trace.from_messages(trace)
        with self.assertRaises(EvalFailure):
            assert_tool_call_efficiency(t, max_redundant=1)


if __name__ == "__main__":
    unittest.main()
