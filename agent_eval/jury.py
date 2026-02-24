"""Jury mode: multi-model voting for LLM-as-judge evaluation.

Runs the same judge across multiple providers/models and aggregates
by majority vote. Reduces single-model bias and improves reliability.

Usage:
    jury = JudgeJury([provider1, provider2, provider3])
    result = jury.judge(judge_faithfulness, context="...", output="...")
"""
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .judge import JudgeCost, JudgeProvider, JudgeResult


@dataclass
class JuryVerdict:
    """Aggregated result from multiple judge votes."""
    # Final aggregated result
    passed: Optional[bool] = None  # For binary judges
    score: Optional[float] = None  # For Likert judges (averaged)
    raw_score: Optional[int] = None  # Rounded average
    reasoning: str = ""
    unsupported_claims: List[str] = field(default_factory=list)

    # Jury metadata
    votes: List[JudgeResult] = field(default_factory=list)
    vote_count: int = 0
    agree_count: int = 0  # How many votes match the final verdict
    agreement_ratio: float = 0.0
    total_cost: Optional[JudgeCost] = None

    @property
    def success(self) -> bool:
        """Whether the jury verdict passed."""
        if self.passed is not None:
            return self.passed
        if self.score is not None:
            return self.score >= 0.6
        return False

    @property
    def unanimous(self) -> bool:
        """Whether all judges agreed."""
        return self.agree_count == self.vote_count


class JudgeJury:
    """Multi-model jury for LLM-as-judge evaluation.

    Takes multiple JudgeProvider instances and runs the same judge
    function across all of them, aggregating results by majority vote.

    Args:
        providers: List of JudgeProvider instances (different models/providers).
        parallel: Whether to run judges in parallel (default True).
    """

    def __init__(
        self,
        providers: List[JudgeProvider],
        parallel: bool = True,
    ):
        if len(providers) < 2:
            raise ValueError("Jury requires at least 2 providers")
        self.providers = providers
        self.parallel = parallel

    def judge(
        self,
        judge_fn: Callable[..., JudgeResult],
        **kwargs: Any,
    ) -> JuryVerdict:
        """Run a judge function across all providers and aggregate.

        Args:
            judge_fn: Any judge function (judge_faithfulness, judge_goal_completion, etc.)
            **kwargs: Arguments to pass to the judge function (excluding 'provider').

        Returns:
            JuryVerdict with aggregated result and individual votes.
        """
        votes = self._collect_votes(judge_fn, **kwargs)
        return self._aggregate(votes)

    def _collect_votes(
        self,
        judge_fn: Callable[..., JudgeResult],
        **kwargs: Any,
    ) -> List[JudgeResult]:
        """Collect votes from all providers."""
        votes: List[JudgeResult] = []

        if self.parallel:
            with ThreadPoolExecutor(max_workers=len(self.providers)) as executor:
                futures = {
                    executor.submit(judge_fn, provider=p, **kwargs): p
                    for p in self.providers
                }
                for future in as_completed(futures):
                    provider = futures[future]
                    try:
                        result = future.result()
                        votes.append(result)
                    except Exception as e:
                        # Record failed vote with error info
                        votes.append(JudgeResult(
                            reasoning=f"[ERROR from {provider.model}@{provider.base_url}] {str(e)[:200]}",
                        ))
        else:
            for p in self.providers:
                try:
                    result = judge_fn(provider=p, **kwargs)
                    votes.append(result)
                except Exception as e:
                    votes.append(JudgeResult(
                        reasoning=f"[ERROR from {p.model}@{p.base_url}] {str(e)[:200]}",
                    ))

        return votes

    def _aggregate(self, votes: List[JudgeResult]) -> JuryVerdict:
        """Aggregate votes by majority voting."""
        # Separate valid votes from errors
        valid_votes = [
            v for v in votes
            if isinstance(v, JudgeResult) and (v.passed is not None or v.score is not None)
        ]

        if not valid_votes:
            return JuryVerdict(
                votes=votes,
                vote_count=len(votes),
                reasoning="All jury members failed to produce a verdict.",
            )

        # Aggregate cost
        total_cost = JudgeCost(model="jury")
        for v in valid_votes:
            if v.judge_cost:
                total_cost.prompt_tokens += v.judge_cost.prompt_tokens
                total_cost.completion_tokens += v.judge_cost.completion_tokens
                total_cost.total_tokens += v.judge_cost.total_tokens

        # Check if binary or Likert
        is_binary = any(v.passed is not None for v in valid_votes)

        if is_binary:
            return self._aggregate_binary(votes, valid_votes, total_cost)
        else:
            return self._aggregate_likert(votes, valid_votes, total_cost)

    def _aggregate_binary(
        self,
        all_votes: List[JudgeResult],
        valid_votes: List[JudgeResult],
        total_cost: JudgeCost,
    ) -> JuryVerdict:
        """Majority vote for binary (pass/fail) judges."""
        pass_count = sum(1 for v in valid_votes if v.passed)
        fail_count = len(valid_votes) - pass_count
        # Tie goes to fail (conservative / fail-safe)
        final_passed = pass_count > fail_count

        agree_count = pass_count if final_passed else fail_count

        # Collect unsupported claims from dissenting votes (if applicable)
        all_claims = []
        for v in valid_votes:
            if v.unsupported_claims:
                all_claims.extend(v.unsupported_claims)
        # Deduplicate claims
        unique_claims = list(dict.fromkeys(all_claims))

        # Build reasoning
        model_names = []
        for v in valid_votes:
            model = v.judge_cost.model if v.judge_cost else "unknown"
            verdict = "pass" if v.passed else "fail"
            model_names.append(f"{model}={verdict}")

        reasoning = (
            f"Jury vote: {pass_count} pass, {fail_count} fail "
            f"({', '.join(model_names)}). "
            f"Verdict: {'PASS' if final_passed else 'FAIL'} by majority."
        )

        return JuryVerdict(
            passed=final_passed,
            reasoning=reasoning,
            unsupported_claims=unique_claims if not final_passed else [],
            votes=all_votes,
            vote_count=len(valid_votes),
            agree_count=agree_count,
            agreement_ratio=agree_count / len(valid_votes) if valid_votes else 0,
            total_cost=total_cost,
        )

    def _aggregate_likert(
        self,
        all_votes: List[JudgeResult],
        valid_votes: List[JudgeResult],
        total_cost: JudgeCost,
    ) -> JuryVerdict:
        """Average scores for Likert (1-5) judges."""
        scores = [v.score for v in valid_votes if v.score is not None]
        raw_scores = [v.raw_score for v in valid_votes if v.raw_score is not None]

        if not scores:
            return JuryVerdict(
                votes=all_votes,
                vote_count=len(valid_votes),
                reasoning="No valid scores from jury.",
                total_cost=total_cost,
            )

        avg_score = sum(scores) / len(scores)
        avg_raw = round(sum(raw_scores) / len(raw_scores)) if raw_scores else None

        # Agreement: how many are within 1 point of the average
        if raw_scores:
            agree_count = sum(1 for r in raw_scores if abs(r - (sum(raw_scores)/len(raw_scores))) <= 1)
        else:
            agree_count = len(scores)

        model_names = []
        for v in valid_votes:
            model = v.judge_cost.model if v.judge_cost else "unknown"
            model_names.append(f"{model}={v.raw_score}/5")

        reasoning = (
            f"Jury scores: {', '.join(model_names)}. "
            f"Average: {avg_raw}/5 ({avg_score:.0%})."
        )

        return JuryVerdict(
            score=avg_score,
            raw_score=avg_raw,
            reasoning=reasoning,
            votes=all_votes,
            vote_count=len(valid_votes),
            agree_count=agree_count,
            agreement_ratio=agree_count / len(valid_votes) if valid_votes else 0,
            total_cost=total_cost,
        )
