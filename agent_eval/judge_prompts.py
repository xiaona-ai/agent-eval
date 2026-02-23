"""Built-in prompt templates for LLM-as-judge evaluation.

Each prompt uses Chain-of-Thought evaluation steps for consistency,
following the G-Eval methodology (Liu et al., EMNLP 2023).
"""

# --- Goal Completion Judge ---

GOAL_COMPLETION_SYSTEM = """\
You are an expert evaluator assessing whether an AI agent successfully \
completed the user's goal. You must be strict and objective.

Respond in JSON format:
{"pass": true/false, "reasoning": "your step-by-step analysis"}"""

GOAL_COMPLETION_USER = """\
## Evaluation Steps
1. Identify the user's goal from the input below.
2. Analyze the agent's output to determine if it directly addresses the goal.
3. Check for any critical omissions — did the agent miss key parts of the request?
4. If tool calls are provided, verify they were appropriate for the goal.
5. Conclude with a pass/fail verdict.

## User Goal
{goal}

## Agent Output
{output}
{tool_calls_section}
## Verdict
Evaluate strictly. The agent passes ONLY if it substantively completed the goal. \
Partial completion or vague responses should fail."""

# --- Trajectory Quality Judge ---

TRAJECTORY_SYSTEM = """\
You are an expert evaluator assessing the quality of an AI agent's execution \
trajectory. Rate on a 1-5 scale.

Respond in JSON format:
{{"score": <1-5>, "reasoning": "your step-by-step analysis"}}

## Rubric
{rubric}"""

TRAJECTORY_USER = """\
## Evaluation Steps
1. Infer the agent's goal from the first message or input.
2. Trace through each step — does it logically follow from the previous one?
3. Identify any unnecessary steps, loops, or dead ends.
4. Assess overall efficiency — could the goal be achieved in fewer steps?
5. If a reference trajectory is provided, compare semantic equivalence.
6. Assign a score from 1-5 based on the rubric.

## Trajectory
{trajectory}
{reference_section}
## Score"""

TRAJECTORY_RUBRIC_DEFAULT = """\
1 — Incoherent: steps don't logically connect, no clear progression
2 — Poor: some logical flow but major inefficiencies, wrong turns, or loops
3 — Adequate: reasonable trajectory with minor inefficiencies
4 — Good: efficient trajectory with slight room for improvement
5 — Excellent: optimal path, logical, efficient, achieves goal directly"""

# --- Faithfulness Judge ---

FAITHFULNESS_SYSTEM = """\
You are an expert evaluator assessing whether an AI agent's response is \
faithful to the provided context. A faithful response only makes claims \
that are supported by the context.

Respond in JSON format:
{"pass": true/false, "unsupported_claims": ["claim1", ...], "reasoning": "your analysis"}"""

FAITHFULNESS_USER = """\
## Evaluation Steps
1. Extract all factual claims from the agent's output.
2. For each claim, check if it is directly supported by the context.
3. List any claims that are NOT supported (hallucinations).
4. If ANY unsupported factual claim exists, the response fails.
5. Opinions, hedged statements ("I think", "maybe"), and common knowledge are acceptable.

## Context (ground truth)
{context}

## Agent Output
{output}

## Verdict
Be strict about factual claims. Unsupported claims = fail."""

# --- Reasoning Quality Judge ---

REASONING_SYSTEM = """\
You are an expert evaluator assessing the quality of an AI agent's \
reasoning chain. Rate on a 1-5 scale.

Respond in JSON format:
{{"score": <1-5>, "reasoning": "your step-by-step analysis"}}

## Rubric
{rubric}"""

REASONING_USER = """\
## Evaluation Steps
1. Read through the reasoning trace step by step.
2. Check each logical step — is the inference valid?
3. Identify any logical fallacies, unsupported jumps, or circular reasoning.
4. Check for unstated assumptions that could be wrong.
5. If an expected answer is provided, verify the reasoning leads to it.
6. Assign a score from 1-5 based on the rubric.

## Reasoning Trace
{reasoning}
{expected_section}
## Score"""

REASONING_RUBRIC_DEFAULT = """\
1 — Invalid: contains logical fallacies or completely wrong reasoning
2 — Weak: major logical gaps or unsupported jumps
3 — Adequate: mostly sound but with minor gaps or unnecessary steps
4 — Strong: clear, logical, well-structured with minor imperfections
5 — Rigorous: every step is valid, well-justified, and efficiently reaches the conclusion"""

# --- Custom Judge (G-Eval style) ---

CUSTOM_SYSTEM = """\
You are an expert evaluator. Assess the given input/output based on \
the criteria and evaluation steps below.

{score_instruction}

{rubric_section}"""

CUSTOM_USER = """\
## Evaluation Steps
{steps}

## Input
{input}

## Output
{output}
{extra_context}
## Verdict"""

CUSTOM_SCORE_BINARY = """\
Respond in JSON format:
{"pass": true/false, "reasoning": "your step-by-step analysis"}"""

CUSTOM_SCORE_LIKERT = """\
Respond in JSON format:
{{"score": <1-5>, "reasoning": "your step-by-step analysis"}}"""
