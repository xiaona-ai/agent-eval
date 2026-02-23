"""Real-world benchmark: run actual LLM judges against crafted traces.

Uses Sorai API (OpenAI-compatible) with real models.
NOT a unit test — makes real API calls and costs money.
"""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_eval import (
    Trace, JudgeProvider,
    judge_goal_completion, judge_trajectory,
    judge_faithfulness, judge_reasoning,
)

API_KEY = os.environ.get("SORAI_API_KEY", "")
BASE_URL = "https://newapi.sorai.me/v1"
MODEL = "grok-4.1-mini"

# ============================================================
# Trace 1: GOOD — Weather agent, efficient and accurate
# ============================================================
GOOD_TRACE = Trace.from_messages([
    {"role": "user", "content": "What's the weather in San Francisco right now?"},
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_1", "type": "function", "function": {
            "name": "get_weather",
            "arguments": '{"city": "San Francisco", "units": "fahrenheit"}'
        }}
    ]},
    {"role": "tool", "name": "get_weather", "tool_call_id": "call_1",
     "content": '{"temperature": 68, "condition": "partly cloudy", "humidity": 72, "wind": "12 mph NW"}'},
    {"role": "assistant", "content": "It's currently 68°F and partly cloudy in San Francisco, with 72% humidity and northwest winds at 12 mph."},
])

# ============================================================
# Trace 2: BAD — Weather agent, hallucinating and inefficient
# ============================================================
BAD_TRACE = Trace.from_messages([
    {"role": "user", "content": "What's the weather in San Francisco right now?"},
    {"role": "assistant", "content": "Let me search for that information."},
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_1", "type": "function", "function": {
            "name": "web_search",
            "arguments": '{"query": "weather"}'
        }}
    ]},
    {"role": "tool", "name": "web_search", "tool_call_id": "call_1",
     "content": "Weather.com - Check your local weather forecast"},
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_2", "type": "function", "function": {
            "name": "web_search",
            "arguments": '{"query": "San Francisco weather today"}'
        }}
    ]},
    {"role": "tool", "name": "web_search", "tool_call_id": "call_2",
     "content": "San Francisco, CA: 68°F, partly cloudy"},
    {"role": "assistant", "content": "The weather in San Francisco is 68°F and partly cloudy. The UV index is extreme at 11, and there's a 90% chance of thunderstorms this evening. Sea level is rising at 3 inches per hour."},
])

# ============================================================
# Trace 3: GOOD — Math reasoning, clean chain
# ============================================================
GOOD_REASONING = """
The user asks: what is 15% of 240?
Step 1: Convert 15% to decimal: 15/100 = 0.15
Step 2: Multiply: 0.15 × 240 = 36
Therefore, 15% of 240 is 36.
"""

# ============================================================
# Trace 4: BAD — Math reasoning, flawed logic
# ============================================================
BAD_REASONING = """
The user asks: what is 15% of 240?
Step 1: 15% means we divide by 15, so 240/15 = 16
Step 2: Actually let me reconsider. 15% is like 1/5 which is 20%.
Step 3: So 20% of 240 is 48.
Step 4: But we want 15%, which is less, so maybe around 40.
Therefore, 15% of 240 is approximately 40.
"""


def run_benchmark():
    if not API_KEY:
        print("❌ Set SORAI_API_KEY to run benchmark")
        sys.exit(1)

    provider = JudgeProvider(
        api_key=API_KEY, base_url=BASE_URL, model=MODEL, timeout=60
    )

    results = []

    def run_judge(name, fn, expected_good, **kwargs):
        try:
            result = fn(provider, **kwargs)
            status = "✅" if (result.success == expected_good) else "⚠️ UNEXPECTED"
            score_str = ""
            if result.passed is not None:
                score_str = f"pass={result.passed}"
            if result.score is not None:
                score_str = f"score={result.score:.2f} (raw={result.raw_score}/5)"
            tokens = result.judge_cost.total_tokens if result.judge_cost else "?"
            print(f"  {status} {name}: {score_str} | tokens={tokens}")
            print(f"    Reasoning: {result.reasoning[:120]}...")
            results.append({
                "name": name, "success_matches_expected": result.success == expected_good,
                "score": result.score, "passed": result.passed, "raw_score": result.raw_score,
                "tokens": result.judge_cost.total_tokens if result.judge_cost else 0,
                "reasoning": result.reasoning,
            })
        except Exception as e:
            print(f"  ❌ {name}: ERROR — {e}")
            results.append({"name": name, "error": str(e)})

    print(f"\n{'='*60}")
    print(f"agent-eval-lite Benchmark — Model: {MODEL}")
    print(f"{'='*60}")

    # --- Goal Completion ---
    print("\n📋 Goal Completion Judge")
    run_judge("good_trace_goal", judge_goal_completion, True,
              goal="What's the weather in San Francisco?",
              output=GOOD_TRACE.final_response.text_content)
    run_judge("bad_trace_goal", judge_goal_completion, False,
              goal="What's the weather in San Francisco?",
              output=BAD_TRACE.final_response.text_content)

    # --- Trajectory ---
    print("\n📋 Trajectory Quality Judge")
    run_judge("good_trace_traj", judge_trajectory, True,
              trajectory=[m.to_dict() for m in GOOD_TRACE])
    run_judge("bad_trace_traj", judge_trajectory, False,
              trajectory=[m.to_dict() for m in BAD_TRACE])

    # --- Faithfulness ---
    print("\n📋 Faithfulness Judge")
    good_context = '{"temperature": 68, "condition": "partly cloudy", "humidity": 72, "wind": "12 mph NW"}'
    bad_context = "San Francisco, CA: 68°F, partly cloudy"
    run_judge("good_trace_faith", judge_faithfulness, True,
              context=good_context, output=GOOD_TRACE.final_response.text_content)
    run_judge("bad_trace_faith", judge_faithfulness, False,
              context=bad_context, output=BAD_TRACE.final_response.text_content)

    # --- Reasoning ---
    print("\n📋 Reasoning Quality Judge")
    run_judge("good_reasoning", judge_reasoning, True,
              reasoning=GOOD_REASONING, expected_answer="36")
    run_judge("bad_reasoning", judge_reasoning, False,
              reasoning=BAD_REASONING, expected_answer="36")

    # --- Summary ---
    print(f"\n{'='*60}")
    correct = sum(1 for r in results if r.get("success_matches_expected"))
    total = len(results)
    total_tokens = sum(r.get("tokens", 0) for r in results)
    print(f"Results: {correct}/{total} matched expectations")
    print(f"Total tokens: {total_tokens}")
    print(f"{'='*60}\n")

    # Save results
    with open("benchmark_results.json", "w") as f:
        json.dump({"model": MODEL, "results": results, "total_tokens": total_tokens}, f, indent=2, ensure_ascii=False)
    print("Results saved to benchmark_results.json")


if __name__ == "__main__":
    run_benchmark()
