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
2. Determine the core deliverable — what must the agent provide to satisfy the goal?
3. Check whether the agent's output contains the core deliverable.
4. Check for fabricated, absurd, or clearly implausible claims in the output. \
Even if the agent addresses the goal, responses containing significant \
misinformation or nonsensical content should fail.
5. Minor omissions of non-essential details are acceptable if the main goal is met.
6. If tool calls are provided, verify they were appropriate for the goal.
7. Conclude with a pass/fail verdict.

## Pass Criteria
- The agent provided the information or action the user requested.
- The response is substantive, not vague or evasive.
- Minor formatting differences or extra helpful context do not cause failure.

## Fail Criteria
- The agent did not address the core goal at all.
- Critical information is missing or wrong.
- The response contains fabricated, absurd, or clearly implausible claims \
(e.g., impossible statistics, nonsensical facts). Even if the core question \
is superficially addressed, responses with significant fabricated content fail.
- The response is a refusal, deflection, or off-topic.

## User Goal
{goal}

## Agent Output
{output}
{tool_calls_section}
## Verdict
Focus on whether the core goal was substantively met, not on perfection."""

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
faithful to the provided context.

Classify each factual claim in the output as:
- SUPPORTED: entailed by or consistent with the context
- CONTRADICTED: directly conflicts with or subtly distorts information \
in the context (e.g., changing "budget" to "production budget", or \
converting "quarter-hour" to a specific "15 minutes" when the source \
is ambiguous)
- FABRICATED: introduces specific facts, numbers, or details that are \
completely absent from the context (extrinsic hallucination)
- BENIGN: adds widely-known common knowledge or trivial formatting \
changes that cannot mislead anyone

CONTRADICTED and FABRICATED claims both count as unfaithful.
BENIGN additions and SUPPORTED claims are acceptable.

Be especially alert to:
- Subtle meaning shifts in paraphrasing (intrinsic hallucination)
- Inferences presented as facts when the source is ambiguous
- Specific numbers or names added that aren't in the source

Respond in JSON format:
{"pass": true/false, "unfaithful_claims": ["claim1", ...], "reasoning": "your analysis"}"""

FAITHFULNESS_USER = """\
## Evaluation Steps
1. Extract all factual claims from the agent's output.
2. For each claim, classify it:
   - SUPPORTED: the context entails this claim. Semantically equivalent \
rephrasings that preserve the original meaning are supported.
   - CONTRADICTED: the output subtly changes, narrows, or broadens \
the meaning of information in the context. Examples: adding specificity \
the source doesn't have ("budget" → "production budget"), changing \
hedged language to definitive claims, or presenting ambiguous info as fact.
   - FABRICATED: the output introduces entirely new facts, statistics, \
or details with no basis in the context.
   - BENIGN: universally known facts or trivial formatting that cannot \
mislead (e.g., standard date formatting).
3. The response FAILS if there are any CONTRADICTED or FABRICATED claims.
4. When in doubt between SUPPORTED and CONTRADICTED, check: does the \
rephrasing preserve the EXACT same meaning, or does it add/change \
nuance? If nuance changed, it's CONTRADICTED.

## Context (ground truth)
{context}

## Agent Output
{output}

## Verdict
Be precise about subtle meaning shifts. A good paraphrase preserves meaning; \
a bad one distorts it."""

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

# --- Multi-step Faithfulness Pipeline (thorough mode) ---

CLAIMS_EXTRACTION_SYSTEM = """\
You are a precise claim extractor. Extract all factual claims from the \
given text. Each claim should be a self-contained, atomic statement that \
can be independently verified against a source.

Respond in JSON format:
{"claims": ["claim1", "claim2", ...]}"""

CLAIMS_EXTRACTION_USER = """\
Extract every factual claim from the following text. Include:
- Explicit factual statements (numbers, names, dates, quantities)
- Implicit claims (e.g., "the best" implies a comparison)
- Causal claims (X caused Y, X led to Y)

Do NOT include:
- Opinions explicitly marked as such ("I think...")
- Questions
- Greetings or filler text

Each claim must be self-contained — someone should understand it \
without seeing the original text.

## Text
{output}

## Claims (JSON)"""

CLAIM_VERIFICATION_SYSTEM = """\
You are a precise fact-checker. For each claim, determine whether it is \
supported by, contradicted by, fabricated beyond, or not addressed in \
the provided context.

Respond in JSON format:
{"verdicts": [{"claim": "...", "verdict": "supported|contradicted|fabricated|idk", \
"reason": "..."}]}"""

CLAIM_VERIFICATION_USER = """\
For each claim below, classify it against the provided context:

- **supported**: The context entails or is consistent with this claim. \
Semantically equivalent rephrasings are supported.
- **contradicted**: The context directly conflicts with this claim, or \
the claim subtly distorts information from the context (meaning shifts, \
added specificity the source doesn't have, hedged→definitive).
- **fabricated**: The claim introduces specific facts, numbers, statistics, \
or quantitative details that are completely absent from the context AND \
could mislead the reader. Examples: made-up percentages, invented measurements, \
specific predictions not in the source.
- **idk**: The claim is not addressed by the context, but is either \
common knowledge, a reasonable inference, or a benign addition that \
cannot mislead anyone. Examples: well-known location names, standard \
units, widely-known facts.

IMPORTANT distinctions:
- "idk" = benign gap (common knowledge, location context, trivial additions)
- "fabricated" = dangerous gap (specific numbers, statistics, predictions, \
measurements that look authoritative but have no source)
- "contradicted" = context says X, claim says Y (direct conflict)
- Only "contradicted" and "fabricated" count as unfaithful.

## Context (ground truth)
{context}

## Claims to verify
{claims}

## Verdicts (JSON)"""
