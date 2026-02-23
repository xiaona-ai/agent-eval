"""Tests for CLI subcommands and JSON output."""
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from agent_eval.cli import main


def _write_jsonl(path: Path, messages):
    path.write_text("\n".join(json.dumps(m) for m in messages) + "\n")


def _run_cli(argv):
    stdout = io.StringIO()
    stderr = io.StringIO()
    with patch("sys.argv", ["agent-eval", *argv]):
        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                main()
                code = 0
            except SystemExit as exc:
                code = int(exc.code) if isinstance(exc.code, int) else 1
    return code, stdout.getvalue(), stderr.getvalue()


TRACE_A = [
    {"role": "user", "content": "What's the weather?"},
    {"role": "assistant", "tool_calls": [
        {"function": {"name": "get_weather", "arguments": "{\"city\":\"SF\"}"}}
    ]},
    {"role": "tool", "name": "get_weather", "content": "80F sunny"},
    {"role": "assistant", "content": "It's sunny in SF.", "usage": {
        "model": "gpt-4o",
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
    }},
]

TRACE_B = [
    {"role": "user", "content": "What's the weather?"},
    {"role": "assistant", "tool_calls": [
        {"function": {"name": "search_web", "arguments": "{\"q\":\"SF weather\"}"}}
    ]},
    {"role": "tool", "name": "search_web", "content": "80F sunny"},
    {"role": "assistant", "content": "Weather info gathered."},
]


class TestCliJsonOutput(unittest.TestCase):
    def test_stats_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.jsonl"
            _write_jsonl(trace_path, TRACE_A)
            code, out, err = _run_cli(["stats", str(trace_path), "--json"])
        self.assertEqual(code, 0, msg=err)
        payload = json.loads(out)
        self.assertEqual(payload["messages"], 4)
        self.assertEqual(payload["tool_calls"], 1)
        self.assertIn("get_weather", payload["tools_used"])

    def test_diff_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_path = Path(tmp) / "base.jsonl"
            curr_path = Path(tmp) / "curr.jsonl"
            _write_jsonl(base_path, TRACE_A)
            _write_jsonl(curr_path, TRACE_A)
            code, out, err = _run_cli(["diff", str(base_path), str(curr_path), "--json"])
        self.assertEqual(code, 0, msg=err)
        payload = json.loads(out)
        self.assertFalse(payload["has_changes"])
        self.assertFalse(payload["is_regression"])
        self.assertIn("summary", payload)


class TestCliCost(unittest.TestCase):
    def test_cost_pass_with_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.jsonl"
            pricing_path = Path(tmp) / "pricing.json"
            _write_jsonl(trace_path, TRACE_A)
            pricing_path.write_text(json.dumps({"gpt-4o": {"input": 2.5, "output": 10.0}}))
            code, out, err = _run_cli([
                "cost", str(trace_path),
                "--max-tokens", "5000",
                "--max-usd", "0.05",
                "--pricing", str(pricing_path),
            ])
        self.assertEqual(code, 0, msg=err)
        self.assertIn("Cost Summary", out)
        self.assertIn("Total tokens:", out)
        self.assertIn("Total cost:", out)

    def test_cost_failure_exit_1(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.jsonl"
            _write_jsonl(trace_path, TRACE_A)
            code, out, err = _run_cli(["cost", str(trace_path), "--max-tokens", "100"])
        self.assertEqual(code, 1)
        self.assertIn("Cost Summary", out)
        self.assertIn("total_tokens", err)

    def test_cost_strict_unknown_model(self):
        trace = [
            {"role": "assistant", "content": "x", "usage": {
                "model": "unknown",
                "prompt_tokens": 100,
                "completion_tokens": 100,
            }},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.jsonl"
            pricing_path = Path(tmp) / "pricing.json"
            _write_jsonl(trace_path, trace)
            pricing_path.write_text(json.dumps({"gpt-4o": {"input": 2.5, "output": 10.0}}))
            code, out, err = _run_cli([
                "cost", str(trace_path),
                "--pricing", str(pricing_path),
                "--strict",
            ])
        self.assertEqual(code, 1)
        self.assertEqual(out.strip(), "")
        self.assertIn("unknown_model", err)


class TestCliConsistency(unittest.TestCase):
    def test_consistency_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "a.jsonl"
            b = Path(tmp) / "b.jsonl"
            _write_jsonl(a, TRACE_A)
            _write_jsonl(b, TRACE_A)
            code, out, err = _run_cli([
                "consistency",
                str(a),
                str(b),
                "--min-tool-consistency", "0.9",
                "--min-answer-consistency", "0.9",
                "--max-step-variance", "0.1",
            ])
        self.assertEqual(code, 0, msg=err)
        self.assertIn("Consistency Report", out)

    def test_consistency_failure_still_prints_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "a.jsonl"
            b = Path(tmp) / "b.jsonl"
            _write_jsonl(a, TRACE_A)
            _write_jsonl(b, TRACE_B)
            code, out, err = _run_cli([
                "consistency",
                str(a),
                str(b),
                "--min-tool-consistency", "0.99",
            ])
        self.assertEqual(code, 1)
        self.assertIn("Consistency Report", out)
        self.assertIn("consistency_tools", err)

    def test_consistency_requires_two_traces(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "a.jsonl"
            _write_jsonl(a, TRACE_A)
            code, out, err = _run_cli(["consistency", str(a)])
        self.assertEqual(code, 1)
        self.assertEqual(out.strip(), "")
        self.assertIn("at least 2 trace files", err)


if __name__ == "__main__":
    unittest.main()
