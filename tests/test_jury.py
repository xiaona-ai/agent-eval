"""Tests for JudgeJury (multi-model voting)."""
import json
import unittest
from unittest.mock import MagicMock, patch

from agent_eval.jury import JudgeJury, JuryVerdict
from agent_eval.judge import JudgeProvider, JudgeResult, JudgeCost


def _make_result(passed=None, score=None, raw_score=None, reasoning="", model="test"):
    """Helper to create a JudgeResult."""
    return JudgeResult(
        passed=passed,
        score=score,
        raw_score=raw_score,
        reasoning=reasoning,
        judge_cost=JudgeCost(
            prompt_tokens=100, completion_tokens=50, total_tokens=150, model=model
        ),
    )


def _make_provider(model="test-model"):
    return JudgeProvider(api_key="fake", model=model)


class TestJudgeJuryInit(unittest.TestCase):

    def test_requires_at_least_2(self):
        with self.assertRaises(ValueError):
            JudgeJury([_make_provider()])

    def test_accepts_2(self):
        jury = JudgeJury([_make_provider(), _make_provider()])
        self.assertEqual(len(jury.providers), 2)


class TestBinaryAggregation(unittest.TestCase):

    def test_majority_pass(self):
        """2 pass, 1 fail → pass."""
        jury = JudgeJury([_make_provider("a"), _make_provider("b"), _make_provider("c")])

        def mock_judge(provider, **kwargs):
            if provider.model == "c":
                return _make_result(passed=False, model="c")
            return _make_result(passed=True, model=provider.model)

        verdict = jury.judge(mock_judge)
        self.assertTrue(verdict.passed)
        self.assertEqual(verdict.vote_count, 3)
        self.assertEqual(verdict.agree_count, 2)
        self.assertAlmostEqual(verdict.agreement_ratio, 2/3)

    def test_majority_fail(self):
        """1 pass, 2 fail → fail."""
        jury = JudgeJury([_make_provider("a"), _make_provider("b"), _make_provider("c")])

        def mock_judge(provider, **kwargs):
            if provider.model == "a":
                return _make_result(passed=True, model="a")
            return _make_result(passed=False, model=provider.model,
                                reasoning="bad")

        verdict = jury.judge(mock_judge)
        self.assertFalse(verdict.passed)
        self.assertEqual(verdict.agree_count, 2)

    def test_unanimous_pass(self):
        """All pass → unanimous."""
        jury = JudgeJury([_make_provider("a"), _make_provider("b")])

        def mock_judge(provider, **kwargs):
            return _make_result(passed=True, model=provider.model)

        verdict = jury.judge(mock_judge)
        self.assertTrue(verdict.passed)
        self.assertTrue(verdict.unanimous)
        self.assertAlmostEqual(verdict.agreement_ratio, 1.0)

    def test_unsupported_claims_collected(self):
        """Unsupported claims from dissenting votes are collected."""
        jury = JudgeJury([_make_provider("a"), _make_provider("b"), _make_provider("c")])

        def mock_judge(provider, **kwargs):
            if provider.model == "a":
                r = _make_result(passed=False, model="a")
                r.unsupported_claims = ["claim1", "claim2"]
                return r
            if provider.model == "b":
                r = _make_result(passed=False, model="b")
                r.unsupported_claims = ["claim2", "claim3"]
                return r
            return _make_result(passed=True, model="c")

        verdict = jury.judge(mock_judge)
        self.assertFalse(verdict.passed)
        # Deduplicated claims
        self.assertEqual(len(verdict.unsupported_claims), 3)

    def test_error_handling(self):
        """One provider errors → still works with remaining votes."""
        jury = JudgeJury([_make_provider("a"), _make_provider("b"), _make_provider("c")])

        def mock_judge(provider, **kwargs):
            if provider.model == "b":
                raise Exception("API timeout")
            return _make_result(passed=True, model=provider.model)

        verdict = jury.judge(mock_judge)
        self.assertTrue(verdict.passed)
        # 2 valid votes (a=pass, c=pass), 1 error
        self.assertEqual(verdict.vote_count, 2)

    def test_all_errors(self):
        """All providers error → empty verdict."""
        jury = JudgeJury([_make_provider("a"), _make_provider("b")])

        def mock_judge(provider, **kwargs):
            raise Exception("down")

        verdict = jury.judge(mock_judge)
        self.assertIsNone(verdict.passed)
        self.assertIn("failed", verdict.reasoning)


class TestLikertAggregation(unittest.TestCase):

    def test_average_scores(self):
        """Scores are averaged."""
        jury = JudgeJury([_make_provider("a"), _make_provider("b"), _make_provider("c")])

        def mock_judge(provider, **kwargs):
            scores = {"a": (4, 0.75), "b": (5, 1.0), "c": (3, 0.5)}
            raw, norm = scores[provider.model]
            return _make_result(score=norm, raw_score=raw, model=provider.model)

        verdict = jury.judge(mock_judge)
        self.assertEqual(verdict.raw_score, 4)  # round(4+5+3 / 3) = 4
        self.assertAlmostEqual(verdict.score, 0.75, places=2)

    def test_score_success_threshold(self):
        """Average score >= 0.6 → success."""
        jury = JudgeJury([_make_provider("a"), _make_provider("b")])

        def mock_judge(provider, **kwargs):
            if provider.model == "a":
                return _make_result(score=0.75, raw_score=4, model="a")
            return _make_result(score=0.5, raw_score=3, model="b")

        verdict = jury.judge(mock_judge)
        self.assertTrue(verdict.success)  # avg = 0.625 >= 0.6


class TestCostAggregation(unittest.TestCase):

    def test_costs_summed(self):
        """Total cost is sum of all votes."""
        jury = JudgeJury([_make_provider("a"), _make_provider("b")])

        def mock_judge(provider, **kwargs):
            return _make_result(passed=True, model=provider.model)

        verdict = jury.judge(mock_judge)
        self.assertEqual(verdict.total_cost.total_tokens, 300)  # 150 * 2
        self.assertEqual(verdict.total_cost.prompt_tokens, 200)


class TestSequentialMode(unittest.TestCase):

    def test_sequential(self):
        """parallel=False runs sequentially."""
        jury = JudgeJury([_make_provider("a"), _make_provider("b")], parallel=False)
        call_order = []

        def mock_judge(provider, **kwargs):
            call_order.append(provider.model)
            return _make_result(passed=True, model=provider.model)

        verdict = jury.judge(mock_judge)
        self.assertTrue(verdict.passed)
        self.assertEqual(len(call_order), 2)


if __name__ == "__main__":
    unittest.main()
