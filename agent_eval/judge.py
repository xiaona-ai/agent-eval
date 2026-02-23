"""LLM-as-judge evaluation for agent traces.

Layer 4: LLM-powered evaluation using any OpenAI-compatible API.
Zero external dependencies — uses only urllib from the standard library.

Key features:
- JudgeProvider: generic OpenAI-compatible API client (urllib)
- 4 built-in judges: goal completion, trajectory, faithfulness, reasoning
- Custom judge with G-Eval style criteria/steps
- Automatic judge cost tracking (integrates with cost module)
"""
import json
import re
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from .assertions import EvalFailure
from . import judge_prompts as prompts


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class JudgeCost:
    """Token usage and estimated cost for a single judge call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    estimated_cost_usd: float = 0.0

    def compute_cost(self, pricing: Optional[Dict[str, Dict[str, float]]] = None):
        """Compute estimated cost from a pricing table (per-1M-token rates)."""
        if not pricing or self.model not in pricing:
            return
        rates = pricing[self.model]
        self.estimated_cost_usd = (
            (self.prompt_tokens / 1_000_000) * rates.get("input", 0)
            + (self.completion_tokens / 1_000_000) * rates.get("output", 0)
        )


@dataclass
class JudgeResult:
    """Result from an LLM judge evaluation."""
    passed: Optional[bool] = None  # For binary judges
    score: Optional[float] = None  # For Likert judges (1-5 normalized to 0-1)
    raw_score: Optional[int] = None  # Raw 1-5 score before normalization
    reasoning: str = ""
    unsupported_claims: List[str] = field(default_factory=list)
    judge_cost: Optional[JudgeCost] = None
    raw_response: str = ""

    @property
    def success(self) -> bool:
        """Whether the evaluation passed (binary) or scored >= 0.6 (Likert)."""
        if self.passed is not None:
            return self.passed
        if self.score is not None:
            return self.score >= 0.6  # 3/5 threshold
        return False


@dataclass
class Rubric:
    """Score anchor for calibrating judge output."""
    score: int  # 1-5
    description: str


DEFAULT_TRAJECTORY_RUBRIC = [
    Rubric(1, "Incoherent: steps don't logically connect, no clear progression"),
    Rubric(2, "Poor: some logical flow but major inefficiencies, wrong turns, or loops"),
    Rubric(3, "Adequate: reasonable trajectory with minor inefficiencies"),
    Rubric(4, "Good: efficient trajectory with slight room for improvement"),
    Rubric(5, "Excellent: optimal path, logical, efficient, achieves goal directly"),
]

DEFAULT_REASONING_RUBRIC = [
    Rubric(1, "Invalid: contains logical fallacies or completely wrong reasoning"),
    Rubric(2, "Weak: major logical gaps or unsupported jumps"),
    Rubric(3, "Adequate: mostly sound but with minor gaps or unnecessary steps"),
    Rubric(4, "Strong: clear, logical, well-structured with minor imperfections"),
    Rubric(5, "Rigorous: every step is valid, well-justified, and efficiently reaches the conclusion"),
]


# ---------------------------------------------------------------------------
# JudgeProvider — OpenAI-compatible API client (urllib, zero deps)
# ---------------------------------------------------------------------------

class JudgeProvider:
    """Call any OpenAI-compatible chat completions API.

    Uses only ``urllib`` from the standard library — zero external dependencies.

    Args:
        api_key: API key for authentication.
        base_url: Base URL for the API (default: OpenAI).
        model: Model to use for judging.
        temperature: Sampling temperature (0.0 for deterministic output).
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o",
        temperature: float = 0.0,
        timeout: int = 60,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def complete(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool = True,
    ) -> tuple:
        """Send a chat completion request.

        Returns:
            (parsed_content: str, usage: dict)
            usage has keys: prompt_tokens, completion_tokens, total_tokens
        """
        url = f"{self.base_url}/chat/completions"
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body_text = ""
            try:
                body_text = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            raise EvalFailure(
                "judge_api_error",
                f"Judge API returned HTTP {e.code}: {body_text[:500]}",
                {"status": e.code, "body": body_text[:500]},
            ) from e
        except urllib.error.URLError as e:
            raise EvalFailure(
                "judge_connection_error",
                f"Failed to connect to judge API at {self.base_url}: {e.reason}",
                {"base_url": self.base_url, "reason": str(e.reason)},
            ) from e

        content = ""
        choices = result.get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content", "")

        usage = result.get("usage", {})
        return content, usage

    def _make_cost(self, usage: dict) -> JudgeCost:
        """Build a JudgeCost from API usage response."""
        return JudgeCost(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            model=self.model,
        )


# ---------------------------------------------------------------------------
# Response parsing helpers
# ---------------------------------------------------------------------------

def _parse_json_response(raw: str) -> dict:
    """Parse JSON from LLM response, handling markdown code fences."""
    text = raw.strip()
    # Strip markdown code fences if present
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find first { ... } block
        m2 = re.search(r"\{.*\}", text, re.DOTALL)
        if m2:
            try:
                return json.loads(m2.group(0))
            except json.JSONDecodeError:
                pass
        return {"_raw": raw}


def _format_rubric(rubric_list: List[Rubric]) -> str:
    """Format rubric list into prompt text."""
    return "\n".join(f"{r.score} — {r.description}" for r in rubric_list)


def _format_trajectory(steps: List[Dict[str, Any]]) -> str:
    """Format trajectory steps for prompt."""
    parts = []
    for i, step in enumerate(steps, 1):
        role = step.get("role", "unknown")
        content = step.get("content", "")
        if isinstance(content, dict):
            content = json.dumps(content, ensure_ascii=False)
        tool_calls = step.get("tool_calls")
        if tool_calls:
            tc_str = json.dumps(tool_calls, ensure_ascii=False, indent=2)
            parts.append(f"Step {i} [{role}]: {content}\nTool calls: {tc_str}")
        else:
            parts.append(f"Step {i} [{role}]: {content}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Built-in judges
# ---------------------------------------------------------------------------

def judge_goal_completion(
    provider: JudgeProvider,
    goal: str,
    output: str,
    tool_calls: Optional[List[Dict]] = None,
    pricing: Optional[Dict[str, Dict[str, float]]] = None,
) -> JudgeResult:
    """Judge whether the agent completed the user's goal.

    Args:
        provider: JudgeProvider instance.
        goal: The user's goal/request.
        output: The agent's final output.
        tool_calls: Optional list of tool calls made by the agent.
        pricing: Optional pricing table for cost estimation.

    Returns:
        JudgeResult with passed=True/False.
    """
    tc_section = ""
    if tool_calls:
        tc_section = f"\n## Tool Calls\n{json.dumps(tool_calls, ensure_ascii=False, indent=2)}\n"

    user_msg = prompts.GOAL_COMPLETION_USER.format(
        goal=goal, output=output, tool_calls_section=tc_section
    )
    messages = [
        {"role": "system", "content": prompts.GOAL_COMPLETION_SYSTEM},
        {"role": "user", "content": user_msg},
    ]

    raw, usage = provider.complete(messages)
    parsed = _parse_json_response(raw)
    cost = provider._make_cost(usage)
    if pricing:
        cost.compute_cost(pricing)

    return JudgeResult(
        passed=bool(parsed.get("pass", False)),
        reasoning=parsed.get("reasoning", ""),
        judge_cost=cost,
        raw_response=raw,
    )


def judge_trajectory(
    provider: JudgeProvider,
    trajectory: List[Dict[str, Any]],
    reference: Optional[List[Dict[str, Any]]] = None,
    rubric: Optional[List[Rubric]] = None,
    pricing: Optional[Dict[str, Dict[str, float]]] = None,
) -> JudgeResult:
    """Judge the quality of an agent's execution trajectory.

    Args:
        provider: JudgeProvider instance.
        trajectory: List of message dicts representing the trajectory.
        reference: Optional reference trajectory for comparison.
        rubric: Optional custom rubric (default provided).
        pricing: Optional pricing table for cost estimation.

    Returns:
        JudgeResult with score (0-1) and raw_score (1-5).
    """
    rubric = rubric or DEFAULT_TRAJECTORY_RUBRIC
    rubric_text = _format_rubric(rubric)

    traj_text = _format_trajectory(trajectory)
    ref_section = ""
    if reference:
        ref_section = f"\n## Reference Trajectory\n{_format_trajectory(reference)}\n"

    sys_msg = prompts.TRAJECTORY_SYSTEM.format(rubric=rubric_text)
    user_msg = prompts.TRAJECTORY_USER.format(
        trajectory=traj_text, reference_section=ref_section
    )
    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_msg},
    ]

    raw, usage = provider.complete(messages)
    parsed = _parse_json_response(raw)
    cost = provider._make_cost(usage)
    if pricing:
        cost.compute_cost(pricing)

    raw_score = parsed.get("score")
    score = None
    if raw_score is not None:
        try:
            raw_score = int(raw_score)
            score = (raw_score - 1) / 4.0  # Normalize 1-5 to 0-1
        except (ValueError, TypeError):
            raw_score = None

    return JudgeResult(
        score=score,
        raw_score=raw_score,
        reasoning=parsed.get("reasoning", ""),
        judge_cost=cost,
        raw_response=raw,
    )


def judge_faithfulness(
    provider: JudgeProvider,
    context: str,
    output: str,
    pricing: Optional[Dict[str, Dict[str, float]]] = None,
) -> JudgeResult:
    """Judge whether the agent's output is faithful to the provided context.

    Args:
        provider: JudgeProvider instance.
        context: Ground truth context (tool results, retrieved docs, etc.).
        output: The agent's response to evaluate.
        pricing: Optional pricing table for cost estimation.

    Returns:
        JudgeResult with passed=True/False and unsupported_claims list.
    """
    user_msg = prompts.FAITHFULNESS_USER.format(context=context, output=output)
    messages = [
        {"role": "system", "content": prompts.FAITHFULNESS_SYSTEM},
        {"role": "user", "content": user_msg},
    ]

    raw, usage = provider.complete(messages)
    parsed = _parse_json_response(raw)
    cost = provider._make_cost(usage)
    if pricing:
        cost.compute_cost(pricing)

    return JudgeResult(
        passed=bool(parsed.get("pass", False)),
        reasoning=parsed.get("reasoning", ""),
        unsupported_claims=parsed.get("unsupported_claims", []),
        judge_cost=cost,
        raw_response=raw,
    )


def judge_reasoning(
    provider: JudgeProvider,
    reasoning: str,
    expected_answer: Optional[str] = None,
    rubric: Optional[List[Rubric]] = None,
    pricing: Optional[Dict[str, Dict[str, float]]] = None,
) -> JudgeResult:
    """Judge the quality of an agent's reasoning chain.

    Args:
        provider: JudgeProvider instance.
        reasoning: The agent's reasoning trace.
        expected_answer: Optional expected answer to validate against.
        rubric: Optional custom rubric (default provided).
        pricing: Optional pricing table for cost estimation.

    Returns:
        JudgeResult with score (0-1) and raw_score (1-5).
    """
    rubric = rubric or DEFAULT_REASONING_RUBRIC
    rubric_text = _format_rubric(rubric)

    expected_section = ""
    if expected_answer:
        expected_section = f"\n## Expected Answer\n{expected_answer}\n"

    sys_msg = prompts.REASONING_SYSTEM.format(rubric=rubric_text)
    user_msg = prompts.REASONING_USER.format(
        reasoning=reasoning, expected_section=expected_section
    )
    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_msg},
    ]

    raw, usage = provider.complete(messages)
    parsed = _parse_json_response(raw)
    cost = provider._make_cost(usage)
    if pricing:
        cost.compute_cost(pricing)

    raw_score = parsed.get("score")
    score = None
    if raw_score is not None:
        try:
            raw_score = int(raw_score)
            score = (raw_score - 1) / 4.0
        except (ValueError, TypeError):
            raw_score = None

    return JudgeResult(
        score=score,
        raw_score=raw_score,
        reasoning=parsed.get("reasoning", ""),
        judge_cost=cost,
        raw_response=raw,
    )


# ---------------------------------------------------------------------------
# Custom judge (G-Eval style)
# ---------------------------------------------------------------------------

def create_custom_judge(
    criteria: str,
    evaluation_steps: Optional[List[str]] = None,
    rubric: Optional[List[Rubric]] = None,
    binary: bool = False,
) -> Callable[..., JudgeResult]:
    """Create a custom judge with G-Eval style evaluation.

    Args:
        criteria: Natural language description of what to evaluate.
        evaluation_steps: Explicit steps for the judge to follow.
            If not provided, the criteria is used as a single step.
        rubric: Score anchors for Likert scoring (ignored if binary=True).
        binary: If True, judge returns pass/fail instead of 1-5 score.

    Returns:
        A callable that takes (provider, input, output, **kwargs) and
        returns a JudgeResult.
    """
    if evaluation_steps:
        steps_text = "\n".join(f"{i}. {s}" for i, s in enumerate(evaluation_steps, 1))
    else:
        steps_text = f"1. Evaluate the output based on this criteria: {criteria}"

    score_instruction = prompts.CUSTOM_SCORE_BINARY if binary else prompts.CUSTOM_SCORE_LIKERT
    rubric_section = ""
    if not binary and rubric:
        rubric_section = f"## Rubric\n{_format_rubric(rubric)}"

    def _judge(
        provider: JudgeProvider,
        input: str,
        output: str,
        context: str = "",
        pricing: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> JudgeResult:
        extra = ""
        if context:
            extra = f"\n## Context\n{context}\n"

        sys_msg = prompts.CUSTOM_SYSTEM.format(
            score_instruction=score_instruction,
            rubric_section=rubric_section,
        )
        user_msg = prompts.CUSTOM_USER.format(
            steps=steps_text,
            input=input,
            output=output,
            extra_context=extra,
        )
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ]

        raw, usage = provider.complete(messages)
        parsed = _parse_json_response(raw)
        cost = provider._make_cost(usage)
        if pricing:
            cost.compute_cost(pricing)

        if binary:
            return JudgeResult(
                passed=bool(parsed.get("pass", False)),
                reasoning=parsed.get("reasoning", ""),
                judge_cost=cost,
                raw_response=raw,
            )
        else:
            raw_score = parsed.get("score")
            score = None
            if raw_score is not None:
                try:
                    raw_score = int(raw_score)
                    score = (raw_score - 1) / 4.0
                except (ValueError, TypeError):
                    raw_score = None
            return JudgeResult(
                score=score,
                raw_score=raw_score,
                reasoning=parsed.get("reasoning", ""),
                judge_cost=cost,
                raw_response=raw,
            )

    _judge.__doc__ = f"Custom judge: {criteria}"
    return _judge
