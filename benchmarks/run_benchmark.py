"""Multi-model benchmark for agent-eval-lite judges.

Runs 8 judge evaluations across multiple models to validate quality.
"""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from agent_eval import (
    Trace, JudgeProvider,
    judge_goal_completion, judge_trajectory,
    judge_faithfulness, judge_reasoning,
)

API_KEY = os.environ.get("SORAI_API_KEY", "")
BASE_URL = "https://newapi.sorai.me/v1"

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

def run_model(model_name):
    provider = JudgeProvider(api_key=API_KEY, base_url=BASE_URL, model=model_name, timeout=60)
    results = []
    correct = 0
    total_tokens = 0
    total_time = 0

    print(f"\n{'─'*60}")
    print(f"  Model: {model_name}")
    print(f"{'─'*60}")

    for name, judge_fn, expect_good, kwargs in TESTS:
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
                "test": name, "model": model_name, "matched": match,
                "passed": result.passed, "score": result.score, "raw_score": result.raw_score,
                "tokens": tokens, "time_s": round(elapsed, 1),
                "reasoning": result.reasoning,
            })
        except Exception as e:
            print(f"  ❌ {name:25s} │ ERROR: {str(e)[:60]}")
            results.append({"test": name, "model": model_name, "error": str(e)[:200]})

    print(f"{'─'*60}")
    print(f"  Score: {correct}/8 │ Tokens: {total_tokens} │ Time: {total_time:.0f}s")
    print(f"{'─'*60}")
    return {"model": model_name, "correct": correct, "total": 8, "tokens": total_tokens, "time_s": round(total_time, 1), "results": results}


def main():
    if not API_KEY:
        print("❌ Set SORAI_API_KEY"); sys.exit(1)

    models = ["grok-4.1-fast", "gpt-5.2", "kimi-k2.5"]

    print("=" * 60)
    print("  agent-eval-lite — Multi-Model Benchmark")
    print("=" * 60)

    all_results = []
    for m in models:
        all_results.append(run_model(m))

    # Summary table
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Model':20s} │ {'Score':7s} │ {'Tokens':7s} │ {'Time':6s}")
    print(f"  {'─'*20}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*6}")
    for r in all_results:
        print(f"  {r['model']:20s} │ {r['correct']}/{r['total']}     │ {r['tokens']:7d} │ {r['time_s']:5.0f}s")
    print(f"{'='*60}\n")

    with open("benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print("Saved to benchmark_results.json")


if __name__ == "__main__":
    main()
