# agent-eval 🧪

![version](https://img.shields.io/badge/version-0.3.1-blue)
![deps](https://img.shields.io/badge/dependencies-0-brightgreen)
![python](https://img.shields.io/badge/python-3.8%2B-3776AB)

Lightweight evaluation framework for AI agents. Zero dependencies, local-first, framework-agnostic.

> **pytest for agents, without the baggage.**

Built by [xiaona-ai](https://x.com/ai_xiaona).

## Why?

32% of teams say **quality is the #1 barrier** to deploying agents in production. Existing eval tools often need hosted infrastructure or heavy dependency stacks.

agent-eval gives deterministic, rule-based assertions that run locally with zero API calls and zero dependencies.

## Install

```bash
pip install agent-eval-lite
```

## Features

- Deterministic assertions for tool use, control flow, output, and latency
- Cost and budget assertions from trace usage + custom pricing
- Multi-run consistency reports and threshold checks
- Safety assertions for sensitive data and prompt injection leak patterns
- CLI support for `show`, `diff`, `stats`, `cost`, `consistency` with JSON output where useful

## Quick Start

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

Works with pytest:

```python
def test_weather_agent():
    trace = run_my_agent("What's the weather in SF?")
    assert_tool_called(trace, "get_weather")
    assert_final_answer_contains(trace, "SF")
    assert_max_steps(trace, 5)
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

### Cost and Budget (v0.3.0)
| Assertion | What it checks |
|-----------|---------------|
| `assert_total_tokens(trace, max_tokens)` | Total usage tokens within budget |
| `assert_total_cost(trace, max_usd, pricing, strict=False)` | Total USD cost within budget |
| `assert_tokens_per_step(trace, max_avg)` | Avg tokens per assistant step |
| `assert_cost_efficiency(trace, max_cost_per_tool_call, pricing)` | Cost per tool call |

```python
from agent_eval import Trace, assert_total_tokens, assert_total_cost

pricing = {"gpt-4o": {"input": 2.5, "output": 10.0}}  # $/1M tokens
trace = Trace.from_jsonl("run.jsonl")

assert_total_tokens(trace, max_tokens=5000)
assert_total_cost(trace, max_usd=0.05, pricing=pricing, strict=True)
```

### Consistency (v0.3.0)
| API | What it checks |
|-----|----------------|
| `ConsistencyReport(traces)` | Pairwise consistency metrics |
| `assert_consistency(...)` | Threshold checks for tool/answer/step variance |

```python
from agent_eval import Trace, ConsistencyReport, assert_consistency

traces = [Trace.from_jsonl(p) for p in ["run1.jsonl", "run2.jsonl", "run3.jsonl"]]
report = ConsistencyReport(traces)
print(report.summary())

assert_consistency(
    traces,
    min_tool_consistency=0.8,
    min_answer_consistency=0.7,
    max_step_variance=1.0,
)
```

### Safety (v0.3.0)
| Assertion | What it checks |
|-----------|---------------|
| `assert_no_sensitive_data(trace, patterns, roles=None)` | Detects regex matches in message content |
| `assert_no_injection_leak(trace, system_prompt, min_chunk_words=5)` | Detects verbatim system prompt leakage |

```python
from agent_eval import assert_no_sensitive_data, assert_no_injection_leak

assert_no_sensitive_data(trace, [
    r"\d{3}-\d{2}-\d{4}",        # SSN
    r"\b(?:\d[ -]*?){13,16}\b",  # card-like numbers
])
assert_no_injection_leak(trace, system_prompt, min_chunk_words=5)
```

## CLI

```bash
agent-eval show run.jsonl
agent-eval show run.jsonl --json

agent-eval diff baseline.jsonl current.jsonl
agent-eval diff baseline.jsonl current.jsonl --json --fail-on-regression

agent-eval stats run.jsonl
agent-eval stats run.jsonl --json

agent-eval cost run.jsonl --max-tokens 5000 --max-usd 0.05 --pricing pricing.json --strict
agent-eval consistency run1.jsonl run2.jsonl run3.jsonl --min-tool-consistency 0.8
```

`cost` always prints token/cost summary and exits `1` on failed thresholds.
`consistency` always prints `ConsistencyReport.summary()` and exits `1` on failed thresholds.

## Trace Format

Standard OpenAI chat messages + optional metadata:

```jsonl
{"role":"user","content":"What's the weather?","timestamp":"2026-02-23T10:00:00Z"}
{"role":"assistant","tool_calls":[{"function":{"name":"get_weather","arguments":"{\"city\":\"SF\"}"}}],"latency_ms":500}
{"role":"tool","name":"get_weather","content":"80°F sunny"}
{"role":"assistant","content":"It's 80°F and sunny.","latency_ms":300}
```

Load from file: `Trace.from_jsonl("run.jsonl")`  
Save to file: `trace.to_jsonl("run.jsonl")`

## Three-Layer Architecture

**Layer 1 (available now): Deterministic assertions** for tool behavior, output, latency, cost, consistency, and safety.

**Layer 2 (planned): Statistical metrics** for drift and similarity over time.

**Layer 3 (planned): LLM-as-Judge** for optional semantic scoring.

## Design Philosophy

- **Zero dependencies**: stdlib only
- **Framework-agnostic**: works with any OpenAI-style trace
- **Deterministic first**: assertions before judges
- **Local-first**: no required data upload
- **File-first**: JSONL traces, version-controllable

## Comparison

| | DeepEval | agentevals | **agent-eval** |
|---|---------|------------|----------------|
| Dependencies | 40+ (torch...) | openai, langchain | **0** |
| Needs API | Yes | Yes (OpenAI) | **No** (deterministic layers) |
| Framework lock-in | No | LangChain | **No** |
| Fully local | Partial | Partial | **Yes** |
| Agent-specific | Partial | Yes | **Yes** |

## License

MIT
