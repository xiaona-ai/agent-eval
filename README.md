# agent-eval 🧪

![version](https://img.shields.io/badge/version-0.4.2-blue)
![deps](https://img.shields.io/badge/dependencies-0-brightgreen)
![python](https://img.shields.io/badge/python-3.8%2B-3776AB)

Lightweight evaluation framework for AI agents. Zero dependencies, local-first, framework-agnostic.

> **pytest for agents, without the baggage.**

Built by [xiaona-ai](https://x.com/ai_xiaona).

## Why?

32% of teams say **quality is the #1 barrier** to deploying agents in production. Existing eval tools often need hosted infrastructure or heavy dependency stacks.

agent-eval gives deterministic assertions *and* LLM-as-judge evaluation — all with zero dependencies.

## Install

```bash
pip install agent-eval-lite
```

## Features

- **Deterministic assertions** for tool use, control flow, output, and latency
- **LLM-as-judge evaluation** (v0.4) — 4 built-in judges + custom G-Eval, zero external deps
- **Cost tracking** — both for agent traces and judge calls themselves
- **Consistency reports** — multi-run comparison with threshold checks
- **Safety assertions** — sensitive data detection and prompt injection leak patterns
- **CLI** — `show`, `diff`, `stats`, `cost`, `consistency`, `judge` with JSON output

## Quick Start

### Deterministic Assertions

```python
from agent_eval import (
    Trace,
    assert_tool_called,
    assert_tool_not_called,
    assert_no_loop,
    assert_max_steps,
    assert_final_answer_contains,
)

trace = Trace.from_messages([
    {"role": "user", "content": "What's the weather in SF?"},
    {"role": "assistant", "tool_calls": [
        {"function": {"name": "get_weather", "arguments": '{"city": "SF"}'}}
    ]},
    {"role": "tool", "name": "get_weather", "content": "80°F and sunny"},
    {"role": "assistant", "content": "It's 80°F and sunny in SF."},
])

assert_tool_called(trace, "get_weather", args={"city": "SF"})
assert_tool_not_called(trace, "delete_database")
assert_no_loop(trace, max_repeats=3)
assert_max_steps(trace, 10)
assert_final_answer_contains(trace, "sunny")
```

### LLM-as-Judge (v0.4.0) 🆕

```python
from agent_eval import JudgeProvider, judge_goal_completion, judge_faithfulness

# Works with ANY OpenAI-compatible API — zero external dependencies
provider = JudgeProvider(
    api_key="your-key",
    base_url="https://api.openai.com/v1",  # or any compatible endpoint
    model="gpt-4o",
)

# Goal completion judge
result = judge_goal_completion(
    provider,
    goal="Find the weather in San Francisco",
    output="It's 80°F and sunny in SF.",
)
print(result.passed)     # True
print(result.reasoning)  # "The agent directly answered..."

# Faithfulness judge (hallucination detection)
result = judge_faithfulness(
    provider,
    context="The API returned: 80°F, sunny, San Francisco",
    output="It's 80°F and sunny in SF.",
)
print(result.passed)             # True
print(result.unsupported_claims) # []

# Every judge call tracks its own cost
print(result.judge_cost.total_tokens)       # 150
print(result.judge_cost.estimated_cost_usd) # 0.00125 (with pricing table)
```

### Custom Judge (G-Eval Style)

```python
from agent_eval import create_custom_judge, Rubric

# Binary (pass/fail)
conciseness_judge = create_custom_judge(
    criteria="Response must be concise and under 100 words",
    binary=True,
)
result = conciseness_judge(provider=provider, input="Summarize X", output="X is Y.")

# Likert (1-5 scale) with custom rubric
helpfulness_judge = create_custom_judge(
    criteria="How helpful is the response?",
    evaluation_steps=[
        "Check if the response addresses the user's question",
        "Check if actionable steps are provided",
        "Evaluate tone and clarity",
    ],
    rubric=[
        Rubric(1, "Not helpful at all"),
        Rubric(3, "Somewhat helpful but missing key info"),
        Rubric(5, "Extremely helpful and actionable"),
    ],
)
result = helpfulness_judge(provider=provider, input="How do I deploy?", output="Run docker push...")
print(result.score)      # 0.75 (normalized 0-1)
print(result.raw_score)  # 4 (original 1-5)
```

### Trajectory Quality Judge

```python
from agent_eval import judge_trajectory

trajectory = [
    {"role": "user", "content": "Book a flight to Tokyo"},
    {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "search_flights"}}]},
    {"role": "tool", "name": "search_flights", "content": "Found 3 flights"},
    {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "book_flight"}}]},
    {"role": "tool", "name": "book_flight", "content": "Booked!"},
    {"role": "assistant", "content": "Your flight to Tokyo is booked!"},
]

result = judge_trajectory(provider, trajectory=trajectory)
print(result.score)      # 0.75-1.0 for efficient trajectories
print(result.raw_score)  # 4 or 5
```

Works with pytest:

```python
def test_weather_agent():
    trace = run_my_agent("What's the weather in SF?")
    assert_tool_called(trace, "get_weather")
    assert_final_answer_contains(trace, "SF")
    assert_max_steps(trace, 5)

def test_agent_quality():
    trace = run_my_agent("What's the weather in SF?")
    provider = JudgeProvider(api_key=os.environ["JUDGE_API_KEY"])
    result = judge_goal_completion(provider, goal="Get SF weather", output=trace.final_response.text_content)
    assert result.success
```

## Assertions

### Tool Call Checks
| Assertion | What it checks |
|-----------|---------------|
| `assert_tool_called(trace, name, args=, min_times=, max_times=)` | Tool was called with optional arg matching |
| `assert_tool_not_called(trace, name)` | Tool was NOT called |
| `assert_tool_call_order(trace, ["a", "b", "c"])` | Tools called in order (subsequence) |
| `assert_tool_call_efficiency(trace, max_redundant=1)` | No excessive duplicate calls |

### Control Flow
| Assertion | What it checks |
|-----------|---------------|
| `assert_no_loop(trace, max_repeats=3)` | No tool called N+ times consecutively |
| `assert_max_steps(trace, N)` | Agent finished within N steps |

### Output Quality
| Assertion | What it checks |
|-----------|---------------|
| `assert_final_answer_contains(trace, text)` | Final response contains text |
| `assert_final_answer_matches(trace, regex)` | Final response matches pattern |
| `assert_no_empty_response(trace)` | No blank assistant messages |
| `assert_no_repetition(trace, threshold=0.85)` | No near-identical consecutive responses |

### Performance
| Assertion | What it checks |
|-----------|---------------|
| `assert_latency(trace, max_seconds=5.0)` | Total latency within bounds |

### Cost and Budget (v0.3)
| Assertion | What it checks |
|-----------|---------------|
| `assert_total_tokens(trace, max_tokens)` | Total usage tokens within budget |
| `assert_total_cost(trace, max_usd, pricing, strict=False)` | Total USD cost within budget |
| `assert_tokens_per_step(trace, max_avg)` | Avg tokens per assistant step |
| `assert_cost_efficiency(trace, max_cost_per_tool_call, pricing)` | Cost per tool call |

### Consistency (v0.3)
| API | What it checks |
|-----|----------------|
| `ConsistencyReport(traces)` | Pairwise consistency metrics |
| `assert_consistency(...)` | Threshold checks for tool/answer/step variance |

### Safety (v0.3)
| Assertion | What it checks |
|-----------|---------------|
| `assert_no_sensitive_data(trace, patterns, roles=None)` | Detects regex matches in message content |
| `assert_no_injection_leak(trace, system_prompt, min_chunk_words=5)` | Detects verbatim system prompt leakage |

### LLM-as-Judge (v0.4) 🆕
| Judge | What it evaluates | Output |
|-------|-------------------|--------|
| `judge_goal_completion(provider, goal, output)` | Did the agent complete the goal? | pass/fail |
| `judge_trajectory(provider, trajectory, reference=)` | Trajectory quality and efficiency | 1-5 score |
| `judge_faithfulness(provider, context, output)` | Is the output grounded in context? | pass/fail + claims |
| `judge_reasoning(provider, reasoning, expected=)` | Reasoning chain quality | 1-5 score |
| `create_custom_judge(criteria, steps=, rubric=, binary=)` | Custom G-Eval criteria | configurable |

**Key differentiators:**
- 🏆 **Zero dependencies** — uses `urllib.request` from stdlib (competitors need openai/httpx/langchain)
- 🏆 **Judge cost tracking** — every judge call reports token usage and estimated cost
- 🏆 **Any provider** — works with any OpenAI-compatible API endpoint
- CoT evaluation steps and rubric anchoring (G-Eval methodology)

## CLI

```bash
# Trace inspection
agent-eval show run.jsonl [--json]
agent-eval diff baseline.jsonl current.jsonl [--json] [--fail-on-regression]
agent-eval stats run.jsonl [--json]

# Cost & consistency
agent-eval cost run.jsonl --max-tokens 5000 --max-usd 0.05 --pricing pricing.json
agent-eval consistency run1.jsonl run2.jsonl run3.jsonl --min-tool-consistency 0.8

# LLM-as-judge (v0.4) 🆕
agent-eval judge run.jsonl --judge-type goal --api-key $KEY --model gpt-4o
agent-eval judge run.jsonl --judge-type trajectory --api-key $KEY --json
agent-eval judge run.jsonl --judge-type faithfulness --context "ground truth text"
agent-eval judge run.jsonl --judge-type custom --criteria "Response is helpful" --binary
```

Set `JUDGE_API_KEY` env var or pass `--api-key`. Use `--base-url` for non-OpenAI providers.

## Trace Format

Standard OpenAI chat messages + optional metadata:

```jsonl
{"role":"user","content":"What's the weather?","timestamp":"2026-02-23T10:00:00Z"}
{"role":"assistant","tool_calls":[{"function":{"name":"get_weather","arguments":"{\"city\":\"SF\"}"}}],"latency_ms":500,"usage":{"prompt_tokens":100,"completion_tokens":50,"model":"gpt-4o"}}
{"role":"tool","name":"get_weather","content":"80°F sunny"}
{"role":"assistant","content":"It's 80°F and sunny.","latency_ms":300}
```

## Architecture

| Layer | What | Dependencies |
|-------|------|-------------|
| **Deterministic** (v0.1-0.3) | Assertions, cost, consistency, safety | Zero |
| **LLM-as-Judge** (v0.4) | Semantic evaluation via any LLM API | Zero (urllib) |
| **Statistical** (planned) | Drift detection, similarity metrics | Zero |

## Design Philosophy

- **Zero dependencies**: stdlib only — even LLM judges use urllib
- **Framework-agnostic**: works with any OpenAI-style trace
- **Deterministic first**: assertions before judges
- **Local-first**: no required data upload
- **Cost-aware**: track both agent and evaluation costs

## Comparison

| | DeepEval | agentevals | judges | **agent-eval** |
|---|---------|------------|--------|----------------|
| Dependencies | 40+ (torch...) | langchain | openai+instructor | **0** |
| Needs API | Yes | Yes | Yes | **Optional** (judge only) |
| Framework lock-in | No | LangChain | No | **No** |
| Fully local | Partial | No | No | **Yes** (deterministic) |
| Judge cost tracking | No | No | No | **Yes** |
| Zero-dep LLM judge | No | No | No | **Yes** (urllib) |

## Benchmark Results

We evaluate our faithfulness judge against **FaithBench** (Bao et al., NAACL 2025) — a human-annotated benchmark of 750 challenging summarization hallucinations where SOTA detectors disagree.

### Faithfulness Judge on FaithBench (100 samples, v0.4.2)

| Judge Model | Accuracy | F1 | Cohen's κ | TP | FP | FN |
|-------------|----------|------|-----------|-----|-----|-----|
| **Claude Sonnet 4.6** | **71%** | **0.688** | **0.424** | 32 | 20 | 9 |
| DeepSeek v3.2 | 67% | 0.459 | 0.260 | 14 | 6 | 27 |
| GPT-4o | 59% | 0.0 | 0.0 | 0 | 0 | 41 |

**Key findings:**
- Cohen's κ = 0.424 (moderate agreement) with Claude Sonnet 4.6 as the judge model
- FaithBench specifically selects the hardest cases where SOTA detectors disagree — 71% accuracy on adversarial samples is competitive
- Our single-call prompt approach achieves this at 1/3 the cost of multi-step pipelines (like DeepEval's claim extraction → verification → aggregation)
- GPT-4o is too strict for faithfulness evaluation (flags everything as unfaithful)

**Prompt methodology:** 4-level NLI-inspired classification (SUPPORTED / CONTRADICTED / FABRICATED / BENIGN), catching both intrinsic hallucinations (subtle meaning distortion) and extrinsic hallucinations (fabricated facts).

> Reproduce: `python benchmarks/run_standard_benchmark.py --dataset faithbench --model claude-sonnet-4-6 --samples 100`

## License

MIT
