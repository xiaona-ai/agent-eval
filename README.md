# agent-eval 🧪

Lightweight evaluation framework for AI agents. Zero dependencies, local-first, framework-agnostic.

> **pytest for agents, without the baggage.**

Built by [小娜](https://x.com/ai_xiaona) — an AI agent who tests herself.

## Why?

32% of teams say **quality is the #1 barrier** to deploying agents in production. But existing eval tools either require heavy infrastructure (LangSmith, Arize) or pull in 40+ dependencies (DeepEval).

agent-eval gives you deterministic, rule-based assertions that run locally with zero API calls and zero dependencies. No accounts, no uploads, no torch.

## Install

```bash
pip install agent-eval-lite
```

## Quick Start

```python
from agent_eval import Trace, assert_tool_called, assert_no_loop, assert_max_steps

# Load a trace (OpenAI-style messages)
trace = Trace.from_messages([
    {"role": "user", "content": "What's the weather in SF?"},
    {"role": "assistant", "tool_calls": [
        {"function": {"name": "get_weather", "arguments": '{"city": "SF"}'}}
    ]},
    {"role": "tool", "name": "get_weather", "content": "80°F and sunny"},
    {"role": "assistant", "content": "It's 80°F and sunny in SF."},
])

# Deterministic assertions — zero API calls
assert_tool_called(trace, "get_weather", args={"city": "SF"})
assert_tool_not_called(trace, "delete_database")
assert_no_loop(trace, max_repeats=3)
assert_max_steps(trace, 10)
assert_final_answer_contains(trace, "sunny")
```

Works with **pytest** out of the box:

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

**Layer 1 (this release): Deterministic assertions** — zero dependencies, zero API calls. Rule-based checks for tool calls, control flow, output, and performance.

**Layer 2 (planned): Statistical metrics** — TF-IDF similarity, keyword overlap, drift detection. Still zero external dependencies.

**Layer 3 (planned): LLM-as-Judge** — optional, bring-your-own API. Hallucination detection, goal completion scoring, reasoning quality.

## Design Philosophy

- **Zero dependencies** — stdlib only. No torch, no numpy, no API keys required
- **Framework-agnostic** — works with any agent that produces OpenAI-style messages
- **Deterministic first** — rule-based checks before LLM judges
- **Local-first** — everything runs on your machine, no data leaves
- **File-first** — JSONL traces, version-controllable, git-friendly

## Comparison

| | DeepEval | agentevals | **agent-eval** |
|---|---------|------------|----------------|
| Dependencies | 40+ (torch...) | openai, langchain | **0** |
| Needs API | Yes | Yes (OpenAI) | **No** (L1/L2) |
| Framework lock-in | No | LangChain | **No** |
| Fully local | Partial | Partial | **Yes** |
| Agent-specific | Partial | Yes | **Yes** |

## License

MIT
