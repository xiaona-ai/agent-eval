"""Tests for LLM-as-judge evaluation (judge.py + judge_prompts.py).

All tests mock the HTTP layer — no real API calls needed.
"""
import json
import unittest
from unittest.mock import patch, MagicMock
from io import BytesIO

from agent_eval.judge import (
    JudgeProvider,
    JudgeResult,
    JudgeCost,
    Rubric,
    judge_goal_completion,
    judge_trajectory,
    judge_faithfulness,
    judge_reasoning,
    create_custom_judge,
    _parse_json_response,
    _format_rubric,
    _format_trajectory,
    DEFAULT_TRAJECTORY_RUBRIC,
    DEFAULT_REASONING_RUBRIC,
)
from agent_eval.assertions import EvalFailure


def _mock_api_response(content: dict, usage: dict = None):
    """Create a mock urllib response with given content and usage."""
    usage = usage or {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
    response_body = json.dumps({
        "choices": [{"message": {"content": json.dumps(content)}}],
        "usage": usage,
    }).encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.read.return_value = response_body
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


class TestJudgeProvider(unittest.TestCase):
    """Tests for JudgeProvider API client."""

    def test_init_defaults(self):
        p = JudgeProvider(api_key="test-key")
        self.assertEqual(p.base_url, "https://api.openai.com/v1")
        self.assertEqual(p.model, "gpt-4o")
        self.assertEqual(p.temperature, 0.0)

    def test_init_custom(self):
        p = JudgeProvider(
            api_key="k", base_url="https://custom.api/v1/",
            model="claude-3", temperature=0.1, timeout=30,
        )
        self.assertEqual(p.base_url, "https://custom.api/v1")  # trailing slash stripped
        self.assertEqual(p.model, "claude-3")

    @patch("agent_eval.judge.urllib.request.urlopen")
    def test_complete_success(self, mock_urlopen):
        mock_urlopen.return_value = _mock_api_response(
            {"pass": True, "reasoning": "looks good"}
        )
        p = JudgeProvider(api_key="test")
        content, usage = p.complete([{"role": "user", "content": "hi"}])
        parsed = json.loads(content)
        self.assertTrue(parsed["pass"])
        self.assertEqual(usage["total_tokens"], 150)

    @patch("agent_eval.judge.urllib.request.urlopen")
    def test_complete_json_mode(self, mock_urlopen):
        """Verify json_mode sets response_format."""
        mock_urlopen.return_value = _mock_api_response({"pass": True})
        p = JudgeProvider(api_key="test")
        p.complete([{"role": "user", "content": "hi"}], json_mode=True)

        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        body = json.loads(req.data)
        self.assertEqual(body["response_format"], {"type": "json_object"})

    @patch("agent_eval.judge.urllib.request.urlopen")
    def test_complete_no_json_mode(self, mock_urlopen):
        mock_urlopen.return_value = _mock_api_response({"pass": True})
        p = JudgeProvider(api_key="test")
        p.complete([{"role": "user", "content": "hi"}], json_mode=False)

        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        body = json.loads(req.data)
        self.assertNotIn("response_format", body)

    @patch("agent_eval.judge.urllib.request.urlopen")
    def test_complete_http_error(self, mock_urlopen):
        import urllib.error
        error_resp = MagicMock()
        error_resp.read.return_value = b"rate limited"
        mock_urlopen.side_effect = urllib.error.HTTPError(
            "url", 429, "Too Many Requests", {}, error_resp
        )
        p = JudgeProvider(api_key="test")
        with self.assertRaises(EvalFailure) as ctx:
            p.complete([{"role": "user", "content": "hi"}])
        self.assertEqual(ctx.exception.check, "judge_api_error")
        self.assertIn("429", str(ctx.exception))

    @patch("agent_eval.judge.urllib.request.urlopen")
    def test_complete_connection_error(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
        p = JudgeProvider(api_key="test")
        with self.assertRaises(EvalFailure) as ctx:
            p.complete([{"role": "user", "content": "hi"}])
        self.assertEqual(ctx.exception.check, "judge_connection_error")


class TestJudgeCost(unittest.TestCase):
    """Tests for JudgeCost cost computation."""

    def test_compute_cost(self):
        cost = JudgeCost(prompt_tokens=1000, completion_tokens=500, model="gpt-4o")
        pricing = {"gpt-4o": {"input": 2.5, "output": 10.0}}
        cost.compute_cost(pricing)
        expected = (1000 / 1e6) * 2.5 + (500 / 1e6) * 10.0
        self.assertAlmostEqual(cost.estimated_cost_usd, expected)

    def test_compute_cost_no_pricing(self):
        cost = JudgeCost(prompt_tokens=100, model="gpt-4o")
        cost.compute_cost(None)
        self.assertEqual(cost.estimated_cost_usd, 0.0)

    def test_compute_cost_unknown_model(self):
        cost = JudgeCost(prompt_tokens=100, model="unknown")
        cost.compute_cost({"gpt-4o": {"input": 2.5, "output": 10.0}})
        self.assertEqual(cost.estimated_cost_usd, 0.0)


class TestJudgeResult(unittest.TestCase):
    """Tests for JudgeResult.success property."""

    def test_success_binary_pass(self):
        r = JudgeResult(passed=True)
        self.assertTrue(r.success)

    def test_success_binary_fail(self):
        r = JudgeResult(passed=False)
        self.assertFalse(r.success)

    def test_success_score_high(self):
        r = JudgeResult(score=0.75)
        self.assertTrue(r.success)

    def test_success_score_low(self):
        r = JudgeResult(score=0.25)
        self.assertFalse(r.success)

    def test_success_score_threshold(self):
        r = JudgeResult(score=0.6)
        self.assertTrue(r.success)

    def test_success_no_data(self):
        r = JudgeResult()
        self.assertFalse(r.success)


class TestParseJsonResponse(unittest.TestCase):
    """Tests for JSON response parsing robustness."""

    def test_clean_json(self):
        result = _parse_json_response('{"pass": true, "reasoning": "ok"}')
        self.assertTrue(result["pass"])

    def test_markdown_fenced(self):
        result = _parse_json_response('```json\n{"pass": true}\n```')
        self.assertTrue(result["pass"])

    def test_markdown_fenced_no_lang(self):
        result = _parse_json_response('```\n{"score": 4}\n```')
        self.assertEqual(result["score"], 4)

    def test_json_with_surrounding_text(self):
        result = _parse_json_response('Here is my eval: {"pass": false, "reasoning": "bad"}')
        self.assertFalse(result["pass"])

    def test_nested_json(self):
        result = _parse_json_response('{"a": {"b": 1}, "c": 2}')
        self.assertEqual(result["a"]["b"], 1)

    def test_multiple_json_objects(self):
        """Should parse the first valid JSON from start."""
        result = _parse_json_response('{"a": 1} extra {"b": 2}')
        self.assertEqual(result["a"], 1)

    def test_invalid_json(self):
        result = _parse_json_response("not json at all")
        self.assertIn("_raw", result)

    def test_empty_string(self):
        result = _parse_json_response("")
        self.assertIn("_raw", result)


class TestFormatHelpers(unittest.TestCase):

    def test_format_rubric(self):
        rubric = [Rubric(1, "Bad"), Rubric(5, "Great")]
        text = _format_rubric(rubric)
        self.assertIn("1 — Bad", text)
        self.assertIn("5 — Great", text)

    def test_format_trajectory_simple(self):
        steps = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        text = _format_trajectory(steps)
        self.assertIn("Step 1 [user]", text)
        self.assertIn("Step 2 [assistant]", text)

    def test_format_trajectory_with_tool_calls(self):
        steps = [
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "search"}}]},
        ]
        text = _format_trajectory(steps)
        self.assertIn("Tool calls:", text)
        self.assertIn("search", text)


class TestGoalCompletionJudge(unittest.TestCase):

    @patch("agent_eval.judge.urllib.request.urlopen")
    def test_pass(self, mock_urlopen):
        mock_urlopen.return_value = _mock_api_response(
            {"pass": True, "reasoning": "Goal fully achieved"}
        )
        p = JudgeProvider(api_key="test")
        result = judge_goal_completion(p, goal="Find the weather", output="It's 72°F and sunny")
        self.assertTrue(result.passed)
        self.assertTrue(result.success)
        self.assertEqual(result.reasoning, "Goal fully achieved")
        self.assertIsNotNone(result.judge_cost)
        self.assertEqual(result.judge_cost.total_tokens, 150)

    @patch("agent_eval.judge.urllib.request.urlopen")
    def test_fail(self, mock_urlopen):
        mock_urlopen.return_value = _mock_api_response(
            {"pass": False, "reasoning": "Did not answer the question"}
        )
        p = JudgeProvider(api_key="test")
        result = judge_goal_completion(p, goal="Find the weather", output="I like cats")
        self.assertFalse(result.passed)
        self.assertFalse(result.success)

    @patch("agent_eval.judge.urllib.request.urlopen")
    def test_with_tool_calls(self, mock_urlopen):
        mock_urlopen.return_value = _mock_api_response({"pass": True, "reasoning": "ok"})
        p = JudgeProvider(api_key="test")
        result = judge_goal_completion(
            p, goal="Search for X", output="Found X",
            tool_calls=[{"function": {"name": "search", "arguments": '{"q": "X"}'}}],
        )
        self.assertTrue(result.passed)

    @patch("agent_eval.judge.urllib.request.urlopen")
    def test_with_pricing(self, mock_urlopen):
        mock_urlopen.return_value = _mock_api_response(
            {"pass": True, "reasoning": "ok"},
            {"prompt_tokens": 500, "completion_tokens": 100, "total_tokens": 600},
        )
        p = JudgeProvider(api_key="test", model="gpt-4o")
        pricing = {"gpt-4o": {"input": 2.5, "output": 10.0}}
        result = judge_goal_completion(p, goal="test", output="test", pricing=pricing)
        self.assertGreater(result.judge_cost.estimated_cost_usd, 0)


class TestTrajectoryJudge(unittest.TestCase):

    @patch("agent_eval.judge.urllib.request.urlopen")
    def test_high_score(self, mock_urlopen):
        mock_urlopen.return_value = _mock_api_response(
            {"score": 5, "reasoning": "Optimal trajectory"}
        )
        p = JudgeProvider(api_key="test")
        traj = [
            {"role": "user", "content": "search for X"},
            {"role": "assistant", "content": "Found X"},
        ]
        result = judge_trajectory(p, trajectory=traj)
        self.assertEqual(result.raw_score, 5)
        self.assertAlmostEqual(result.score, 1.0)
        self.assertTrue(result.success)

    @patch("agent_eval.judge.urllib.request.urlopen")
    def test_low_score(self, mock_urlopen):
        mock_urlopen.return_value = _mock_api_response(
            {"score": 1, "reasoning": "Incoherent"}
        )
        p = JudgeProvider(api_key="test")
        result = judge_trajectory(p, trajectory=[{"role": "user", "content": "hi"}])
        self.assertEqual(result.raw_score, 1)
        self.assertAlmostEqual(result.score, 0.0)
        self.assertFalse(result.success)

    @patch("agent_eval.judge.urllib.request.urlopen")
    def test_with_reference(self, mock_urlopen):
        mock_urlopen.return_value = _mock_api_response({"score": 4, "reasoning": "close"})
        p = JudgeProvider(api_key="test")
        result = judge_trajectory(
            p,
            trajectory=[{"role": "assistant", "content": "did A then B"}],
            reference=[{"role": "assistant", "content": "did A then B then C"}],
        )
        self.assertEqual(result.raw_score, 4)

    @patch("agent_eval.judge.urllib.request.urlopen")
    def test_custom_rubric(self, mock_urlopen):
        mock_urlopen.return_value = _mock_api_response({"score": 3, "reasoning": "ok"})
        p = JudgeProvider(api_key="test")
        custom = [Rubric(1, "Terrible"), Rubric(3, "OK"), Rubric(5, "Perfect")]
        result = judge_trajectory(p, trajectory=[{"role": "user", "content": "x"}], rubric=custom)
        self.assertEqual(result.raw_score, 3)

    @patch("agent_eval.judge.urllib.request.urlopen")
    def test_invalid_score(self, mock_urlopen):
        mock_urlopen.return_value = _mock_api_response({"score": "bad", "reasoning": "oops"})
        p = JudgeProvider(api_key="test")
        result = judge_trajectory(p, trajectory=[{"role": "user", "content": "x"}])
        self.assertIsNone(result.raw_score)
        self.assertIsNone(result.score)


class TestFaithfulnessJudge(unittest.TestCase):

    @patch("agent_eval.judge.urllib.request.urlopen")
    def test_faithful(self, mock_urlopen):
        mock_urlopen.return_value = _mock_api_response(
            {"pass": True, "unsupported_claims": [], "reasoning": "All claims supported"}
        )
        p = JudgeProvider(api_key="test")
        result = judge_faithfulness(
            p, context="Paris is the capital of France", output="The capital of France is Paris"
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.unsupported_claims, [])

    @patch("agent_eval.judge.urllib.request.urlopen")
    def test_unfaithful(self, mock_urlopen):
        mock_urlopen.return_value = _mock_api_response({
            "pass": False,
            "unsupported_claims": ["Berlin is the capital of France"],
            "reasoning": "Contradicts context",
        })
        p = JudgeProvider(api_key="test")
        result = judge_faithfulness(
            p, context="Paris is the capital of France", output="Berlin is the capital of France"
        )
        self.assertFalse(result.passed)
        self.assertEqual(len(result.unsupported_claims), 1)

    @patch("agent_eval.judge.urllib.request.urlopen")
    def test_thorough_faithful(self, mock_urlopen):
        """Thorough mode: 2 API calls (extract + verify), all supported."""
        responses = [
            _mock_api_response({"claims": ["Paris is the capital of France"]}),
            _mock_api_response({"verdicts": [
                {"claim": "Paris is the capital of France", "verdict": "supported"}
            ]}),
        ]
        mock_urlopen.side_effect = responses
        p = JudgeProvider(api_key="test")
        result = judge_faithfulness(
            p, context="Paris is the capital of France",
            output="The capital of France is Paris",
            mode="thorough",
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.unsupported_claims, [])
        self.assertEqual(mock_urlopen.call_count, 2)
        self.assertIn("1 claims", result.reasoning)

    @patch("agent_eval.judge.urllib.request.urlopen")
    def test_thorough_contradicted(self, mock_urlopen):
        """Thorough mode: contradicted claim → fail."""
        responses = [
            _mock_api_response({"claims": [
                "Berlin is the capital of France",
            ]}),
            _mock_api_response({"verdicts": [
                {"claim": "Berlin is the capital of France", "verdict": "contradicted",
                 "reason": "Context says Paris, not Berlin"},
            ]}),
        ]
        mock_urlopen.side_effect = responses
        p = JudgeProvider(api_key="test")
        result = judge_faithfulness(
            p, context="Paris is the capital of France",
            output="Berlin is the capital of France",
            mode="thorough",
        )
        self.assertFalse(result.passed)
        self.assertEqual(len(result.unsupported_claims), 1)

    @patch("agent_eval.judge.urllib.request.urlopen")
    def test_thorough_idk_not_unfaithful(self, mock_urlopen):
        """Thorough mode: idk claims are NOT counted as unfaithful."""
        responses = [
            _mock_api_response({"claims": [
                "It is 68°F",
                "in San Francisco",
            ]}),
            _mock_api_response({"verdicts": [
                {"claim": "It is 68°F", "verdict": "supported"},
                {"claim": "in San Francisco", "verdict": "idk",
                 "reason": "Context doesn't mention location"},
            ]}),
        ]
        mock_urlopen.side_effect = responses
        p = JudgeProvider(api_key="test")
        result = judge_faithfulness(
            p, context='{"temperature": 68}',
            output="It is 68°F in San Francisco",
            mode="thorough",
        )
        self.assertTrue(result.passed)  # idk doesn't fail!
        self.assertEqual(result.unsupported_claims, [])

    @patch("agent_eval.judge.urllib.request.urlopen")
    def test_thorough_no_claims(self, mock_urlopen):
        """Thorough mode: no claims from non-empty output → falls back to fast mode."""
        responses = [
            _mock_api_response({"claims": []}),  # extraction returns empty
            _mock_api_response({"pass": True, "reasoning": "Fast mode fallback"}),  # fast mode
        ]
        mock_urlopen.side_effect = responses
        p = JudgeProvider(api_key="test")
        result = judge_faithfulness(
            p, context="anything", output="Hello there!",
            mode="thorough",
        )
        # Should fall back to fast mode, not silently pass
        self.assertTrue(result.passed)
        self.assertEqual(mock_urlopen.call_count, 2)  # extraction + fast fallback

    @patch("agent_eval.judge.urllib.request.urlopen")
    def test_thorough_cost_aggregation(self, mock_urlopen):
        """Thorough mode: costs from both calls are aggregated."""
        responses = [
            _mock_api_response(
                {"claims": ["fact1"]},
                usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            ),
            _mock_api_response(
                {"verdicts": [{"claim": "fact1", "verdict": "supported"}]},
                usage={"prompt_tokens": 200, "completion_tokens": 80, "total_tokens": 280},
            ),
        ]
        mock_urlopen.side_effect = responses
        p = JudgeProvider(api_key="test")
        result = judge_faithfulness(
            p, context="ctx", output="fact1", mode="thorough",
        )
        self.assertEqual(result.judge_cost.prompt_tokens, 300)
        self.assertEqual(result.judge_cost.completion_tokens, 130)
        self.assertEqual(result.judge_cost.total_tokens, 430)


class TestReasoningJudge(unittest.TestCase):

    @patch("agent_eval.judge.urllib.request.urlopen")
    def test_strong_reasoning(self, mock_urlopen):
        mock_urlopen.return_value = _mock_api_response(
            {"score": 4, "reasoning": "Clear logical flow"}
        )
        p = JudgeProvider(api_key="test")
        result = judge_reasoning(p, reasoning="A implies B. B implies C. Therefore A implies C.")
        self.assertEqual(result.raw_score, 4)
        self.assertAlmostEqual(result.score, 0.75)

    @patch("agent_eval.judge.urllib.request.urlopen")
    def test_with_expected(self, mock_urlopen):
        mock_urlopen.return_value = _mock_api_response({"score": 5, "reasoning": "correct"})
        p = JudgeProvider(api_key="test")
        result = judge_reasoning(
            p, reasoning="2+2=4 because addition", expected_answer="4"
        )
        self.assertEqual(result.raw_score, 5)


class TestCustomJudge(unittest.TestCase):

    @patch("agent_eval.judge.urllib.request.urlopen")
    def test_binary_custom(self, mock_urlopen):
        mock_urlopen.return_value = _mock_api_response(
            {"pass": True, "reasoning": "Response is concise"}
        )
        judge = create_custom_judge(criteria="Response must be concise", binary=True)
        p = JudgeProvider(api_key="test")
        result = judge(provider=p, input="Summarize X", output="X is Y")
        self.assertTrue(result.passed)

    @patch("agent_eval.judge.urllib.request.urlopen")
    def test_likert_custom(self, mock_urlopen):
        mock_urlopen.return_value = _mock_api_response(
            {"score": 4, "reasoning": "Good helpfulness"}
        )
        judge = create_custom_judge(
            criteria="Helpfulness",
            evaluation_steps=["Check if response addresses the question", "Check tone"],
        )
        p = JudgeProvider(api_key="test")
        result = judge(provider=p, input="How do I X?", output="Do Y then Z")
        self.assertEqual(result.raw_score, 4)

    @patch("agent_eval.judge.urllib.request.urlopen")
    def test_custom_with_rubric(self, mock_urlopen):
        mock_urlopen.return_value = _mock_api_response({"score": 3, "reasoning": "ok"})
        judge = create_custom_judge(
            criteria="Quality",
            rubric=[Rubric(1, "Bad"), Rubric(3, "Average"), Rubric(5, "Excellent")],
        )
        p = JudgeProvider(api_key="test")
        result = judge(provider=p, input="test", output="test")
        self.assertEqual(result.raw_score, 3)

    @patch("agent_eval.judge.urllib.request.urlopen")
    def test_custom_with_context(self, mock_urlopen):
        mock_urlopen.return_value = _mock_api_response({"pass": True, "reasoning": "ok"})
        judge = create_custom_judge(criteria="Grounded", binary=True)
        p = JudgeProvider(api_key="test")
        result = judge(provider=p, input="Q", output="A", context="some context")
        self.assertTrue(result.passed)

    @patch("agent_eval.judge.urllib.request.urlopen")
    def test_custom_with_pricing(self, mock_urlopen):
        mock_urlopen.return_value = _mock_api_response(
            {"pass": True, "reasoning": "ok"},
            {"prompt_tokens": 200, "completion_tokens": 80, "total_tokens": 280},
        )
        judge = create_custom_judge(criteria="test", binary=True)
        p = JudgeProvider(api_key="test", model="gpt-4o")
        pricing = {"gpt-4o": {"input": 2.5, "output": 10.0}}
        result = judge(provider=p, input="x", output="y", pricing=pricing)
        self.assertGreater(result.judge_cost.estimated_cost_usd, 0)


class TestDefaultRubrics(unittest.TestCase):
    """Verify default rubrics are well-formed."""

    def test_trajectory_rubric(self):
        self.assertEqual(len(DEFAULT_TRAJECTORY_RUBRIC), 5)
        scores = [r.score for r in DEFAULT_TRAJECTORY_RUBRIC]
        self.assertEqual(scores, [1, 2, 3, 4, 5])

    def test_reasoning_rubric(self):
        self.assertEqual(len(DEFAULT_REASONING_RUBRIC), 5)
        scores = [r.score for r in DEFAULT_REASONING_RUBRIC]
        self.assertEqual(scores, [1, 2, 3, 4, 5])


class TestCodexReviewFixes(unittest.TestCase):
    """Tests for bugs found by Codex gpt-5.3-codex-xhigh review."""

    def test_bool_string_false(self):
        """bool('false') should be False, not True."""
        from agent_eval.judge import _parse_bool
        self.assertFalse(_parse_bool("false"))
        self.assertFalse(_parse_bool("False"))
        self.assertFalse(_parse_bool("FALSE"))
        self.assertTrue(_parse_bool("true"))
        self.assertTrue(_parse_bool("True"))
        self.assertTrue(_parse_bool(True))
        self.assertFalse(_parse_bool(False))
        self.assertFalse(_parse_bool(0))
        self.assertTrue(_parse_bool(1))

    @patch("agent_eval.judge.urllib.request.urlopen")
    def test_goal_string_false(self, mock_urlopen):
        """LLM returning 'false' as string should still fail."""
        mock_urlopen.return_value = _mock_api_response(
            {"pass": "false", "reasoning": "nope"}
        )
        p = JudgeProvider(api_key="test")
        result = judge_goal_completion(p, goal="X", output="Y")
        self.assertFalse(result.passed)

    @patch("agent_eval.judge.urllib.request.urlopen")
    def test_score_clamped_above_5(self, mock_urlopen):
        """Score > 5 should be clamped to 5 (1.0 normalized)."""
        mock_urlopen.return_value = _mock_api_response(
            {"score": 7, "reasoning": "too generous"}
        )
        p = JudgeProvider(api_key="test")
        result = judge_trajectory(p, trajectory=[{"role": "user", "content": "x"}])
        self.assertEqual(result.raw_score, 5)
        self.assertAlmostEqual(result.score, 1.0)

    @patch("agent_eval.judge.urllib.request.urlopen")
    def test_score_clamped_below_1(self, mock_urlopen):
        """Score < 1 should be clamped to 1 (0.0 normalized)."""
        mock_urlopen.return_value = _mock_api_response(
            {"score": -1, "reasoning": "harsh"}
        )
        p = JudgeProvider(api_key="test")
        result = judge_trajectory(p, trajectory=[{"role": "user", "content": "x"}])
        self.assertEqual(result.raw_score, 1)
        self.assertAlmostEqual(result.score, 0.0)

    @patch("agent_eval.judge.urllib.request.urlopen")
    def test_missing_usage_in_response(self, mock_urlopen):
        """API returning no usage field should not crash."""
        response_body = json.dumps({
            "choices": [{"message": {"content": '{"pass": true, "reasoning": "ok"}'}}],
        }).encode("utf-8")
        mock_resp = MagicMock()
        mock_resp.read.return_value = response_body
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp
        p = JudgeProvider(api_key="test")
        result = judge_goal_completion(p, goal="X", output="Y")
        self.assertTrue(result.passed)
        self.assertIsNotNone(result.judge_cost)
        self.assertEqual(result.judge_cost.total_tokens, 0)


if __name__ == "__main__":
    unittest.main()
