"""Multi-model benchmark for agent-eval-lite judges.

Runs 8 judge evaluations across multiple models/providers to validate quality.
Supports multi-provider parallel execution to avoid rate limits.
"""
import json, os, sys, time, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from agent_eval import (
    Trace, JudgeProvider,
    judge_goal_completion, judge_trajectory,
    judge_faithfulness, judge_reasoning,
)

# === Provider configs ===
PROVIDERS = {
    "wcgio":    {"url": "https://520.wcgio.com/v1",         "key": os.environ.get("WCGIO_KEY", "")},
    "huan666":  {"url": "https://ai.huan666.de/v1",         "key": os.environ.get("HUAN666_KEY", "")},
    "aicenter": {"url": "https://aicenter.hejiu.icu/v1",    "key": os.environ.get("AICENTER_KEY", "")},
    "api-test": {"url": "https://openai.api-test.us.ci/v1", "key": os.environ.get("APITEST_KEY", "")},
}

# Model -> preferred provider (can override with --provider)
MODEL_PROVIDER = {
    "claude-sonnet-4-6":        "aicenter",
    "claude-sonnet-4-5-20250929": "huan666",
    "grok-4.1-fast":            "wcgio",
    "grok-4.1-mini":            "wcgio",
    "deepseek-v3.2":            "huan666",
    "deepseek-chat":            "wcgio",
    "gpt-4o":                   "aicenter",
    "gpt-5.2":                  "wcgio",
    "kimi-k2.5":                "huan666",
    "gemini-2.5-pro":           "aicenter",
    "gemini-2.5-flash":         "aicenter",
    "glm-5":                    "aicenter",
}

DEFAULT_MODELS = ["grok-4.1-fast", "claude-sonnet-4-6", "deepseek-v3.2", "gemini-2.5-flash"]

# === Traces ===

GOOD_TRACE = Trace.from_messages([
    {"role": "user", "content": "What's the weather in San Francisco right now?"},
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_1", "type": "function", "function": {
            "name": "get_weather", "arguments": '{"city": "San Francisco", "units": "fahrenheit"}'
        }}
    ]},
    {"role": "tool", "name": "get_weather", "tool_call_id": "call_1",
     "content": '{"temperature": 68, "condition": "partly cloudy", "humidity": 72, "wind": "12 mph NW"}'},
    {"role": "assistant", "content": "It's currently 68°F and partly cloudy in San Francisco, with 72% humidity and northwest winds at 12 mph."},
])

BAD_TRACE = Trace.from_messages([
    {"role": "user", "content": "What's the weather in San Francisco right now?"},
    {"role": "assistant", "content": "Let me search for that information."},
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_1", "type": "function", "function": {
            "name": "web_search", "arguments": '{"query": "weather"}'
        }}
    ]},
    {"role": "tool", "name": "web_search", "tool_call_id": "call_1",
     "content": "Weather.com - Check your local weather forecast"},
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_2", "type": "function", "function": {
            "name": "web_search", "arguments": '{"query": "San Francisco weather today"}'
        }}
    ]},
    {"role": "tool", "name": "web_search", "tool_call_id": "call_2",
     "content": "San Francisco, CA: 68°F, partly cloudy"},
    {"role": "assistant", "content": "The weather in San Francisco is 68°F and partly cloudy. The UV index is extreme at 11, and there's a 90% chance of thunderstorms this evening. Sea level is rising at 3 inches per hour."},
])

GOOD_REASONING = """
The user asks: what is 15% of 240?
Step 1: Convert 15% to decimal: 15/100 = 0.15
Step 2: Multiply: 0.15 × 240 = 36
Therefore, 15% of 240 is 36.
"""

BAD_REASONING = """
The user asks: what is 15% of 240?
Step 1: 15% means we divide by 15, so 240/15 = 16
Step 2: Actually let me reconsider. 15% is like 1/5 which is 20%.
Step 3: So 20% of 240 is 48.
Step 4: But we want 15%, which is less, so maybe around 40.
Therefore, 15% of 240 is approximately 40.
"""

GOOD_CONTEXT = '{"temperature": 68, "condition": "partly cloudy", "humidity": 72, "wind": "12 mph NW"}'
BAD_CONTEXT = "San Francisco, CA: 68°F, partly cloudy"

TESTS = [
    ("Goal: good trace", judge_goal_completion, True,
     {"goal": "What's the weather in San Francisco?", "output": "It's currently 68°F and partly cloudy in San Francisco, with 72% humidity and northwest winds at 12 mph."}),
    ("Goal: bad trace", judge_goal_completion, False,
     {"goal": "What's the weather in San Francisco?", "output": "The weather in San Francisco is 68°F and partly cloudy. The UV index is extreme at 11, and there's a 90% chance of thunderstorms this evening. Sea level is rising at 3 inches per hour."}),
    ("Trajectory: good", judge_trajectory, True,
     {"trajectory": [m.to_dict() for m in GOOD_TRACE]}),
    ("Trajectory: bad", judge_trajectory, False,
     {"trajectory": [m.to_dict() for m in BAD_TRACE]}),
    ("Faithfulness: good", judge_faithfulness, True,
     {"context": GOOD_CONTEXT, "output": "It's currently 68°F and partly cloudy in San Francisco, with 72% humidity and northwest winds at 12 mph."}),
    ("Faithfulness: bad", judge_faithfulness, False,
     {"context": BAD_CONTEXT, "output": "The weather in San Francisco is 68°F and partly cloudy. The UV index is extreme at 11, and there's a 90% chance of thunderstorms this evening. Sea level is rising at 3 inches per hour."}),
    ("Reasoning: good", judge_reasoning, True,
     {"reasoning": GOOD_REASONING, "expected_answer": "36"}),
    ("Reasoning: bad", judge_reasoning, False,
     {"reasoning": BAD_REASONING, "expected_answer": "36"}),
]

RPM_DELAY = 7.5  # seconds between calls per provider


def get_provider(model, override_provider=None):
    """Get provider config for a model."""
    name = override_provider or MODEL_PROVIDER.get(model)
    if not name or name not in PROVIDERS:
        # Try to find any provider that might have it
        for n, p in PROVIDERS.items():
            if p["key"]:
                return n, p
        raise ValueError(f"No provider configured for {model}")
    p = PROVIDERS[name]
    if not p["key"]:
        raise ValueError(f"No API key for provider {name}")
    return name, p


def run_model(model_name, override_provider=None):
    prov_name, prov = get_provider(model_name, override_provider)
    provider = JudgeProvider(
        api_key=prov["key"], base_url=prov["url"],
        model=model_name, timeout=60,
    )
    results = []
    correct = 0
    total_tokens = 0
    total_time = 0

    print(f"\n{'─'*60}")
    print(f"  Model: {model_name} (via {prov_name})")
    print(f"{'─'*60}")

    for i, (name, judge_fn, expect_good, kwargs) in enumerate(TESTS):
        if i > 0:
            time.sleep(RPM_DELAY)
        try:
            t = time.time()
            result = judge_fn(provider, **kwargs)
            elapsed = time.time() - t
            total_time += elapsed

            match = result.success == expect_good
            if match:
                correct += 1
            icon = "✅" if match else "⚠️"

            score_str = ""
            if result.passed is not None:
                score_str = f"pass={'✓' if result.passed else '✗'}"
            if result.score is not None:
                score_str = f"{result.raw_score}/5 ({result.score:.0%})"

            tokens = result.judge_cost.total_tokens if result.judge_cost else 0
            total_tokens += tokens
            reason_preview = result.reasoning[:80].replace('\n', ' ') if result.reasoning else "(no reasoning)"

            print(f"  {icon} {name:25s} │ {score_str:12s} │ {tokens:4d}tok │ {elapsed:.1f}s")
            print(f"     └─ {reason_preview}...")

            results.append({
                "test": name, "model": model_name, "provider": prov_name,
                "matched": match, "passed": result.passed,
                "score": result.score, "raw_score": result.raw_score,
                "tokens": tokens, "time_s": round(elapsed, 1),
                "reasoning": result.reasoning,
            })
        except Exception as e:
            print(f"  ❌ {name:25s} │ ERROR: {str(e)[:60]}")
            results.append({"test": name, "model": model_name, "provider": prov_name, "error": str(e)[:200]})

    print(f"{'─'*60}")
    print(f"  Score: {correct}/8 │ Tokens: {total_tokens} │ Time: {total_time:.0f}s")
    print(f"{'─'*60}")
    return {
        "model": model_name, "provider": prov_name,
        "correct": correct, "total": 8,
        "tokens": total_tokens, "time_s": round(total_time, 1),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="agent-eval-lite multi-model benchmark")
    parser.add_argument("--models", nargs="+", default=None, help="Models to test")
    parser.add_argument("--provider", default=None, help="Force all models to use this provider")
    parser.add_argument("--parallel", action="store_true", help="Run models in parallel (different providers)")
    parser.add_argument("--rpm-delay", type=float, default=7.5, help="Delay between calls (seconds)")
    args = parser.parse_args()

    global RPM_DELAY
    RPM_DELAY = args.rpm_delay
    models = args.models or DEFAULT_MODELS

    print("=" * 60)
    print("  agent-eval-lite — Multi-Model Benchmark")
    print("=" * 60)

    all_results = []

    if args.parallel:
        # Run models on different providers in parallel
        with ThreadPoolExecutor(max_workers=len(models)) as executor:
            futures = {
                executor.submit(run_model, m, args.provider): m
                for m in models
            }
            for future in as_completed(futures):
                try:
                    all_results.append(future.result())
                except Exception as e:
                    model = futures[future]
                    print(f"  ❌ {model}: {e}")
                    all_results.append({"model": model, "correct": 0, "total": 8, "error": str(e)})
    else:
        for m in models:
            all_results.append(run_model(m, args.provider))

    # Sort by model name for consistent output
    all_results.sort(key=lambda x: x.get("model", ""))

    # Summary table
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Model':25s} │ {'Provider':10s} │ {'Score':5s} │ {'Tokens':7s} │ {'Time':5s}")
    print(f"  {'─'*25}─┼─{'─'*10}─┼─{'─'*5}─┼─{'─'*7}─┼─{'─'*5}")
    for r in all_results:
        print(f"  {r.get('model','?'):25s} │ {r.get('provider','?'):10s} │ {r.get('correct',0)}/{r.get('total',8)}   │ {r.get('tokens',0):7d} │ {r.get('time_s',0):4.0f}s")
    print(f"{'='*60}\n")

    with open("benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print("Saved to benchmark_results.json")


if __name__ == "__main__":
    main()
