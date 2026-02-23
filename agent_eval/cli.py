"""CLI for agent-eval."""
import argparse
import json
import sys
from pathlib import Path

from .trace import Trace
from .diff import diff_traces


def main():
    parser = argparse.ArgumentParser(
        prog="agent-eval",
        description="Lightweight agent evaluation framework",
    )
    sub = parser.add_subparsers(dest="command")

    # trace show
    p_show = sub.add_parser("show", help="Display a trace file")
    p_show.add_argument("file", help="JSONL trace file")
    p_show.add_argument("--json", action="store_true", dest="as_json", help="Output as JSON")

    # diff
    p_diff = sub.add_parser("diff", help="Compare two traces for regressions")
    p_diff.add_argument("baseline", help="Baseline trace (JSONL)")
    p_diff.add_argument("current", help="Current trace (JSONL)")
    p_diff.add_argument("--fail-on-regression", action="store_true", help="Exit 1 if regression detected")

    # stats
    p_stats = sub.add_parser("stats", help="Show trace statistics")
    p_stats.add_argument("file", help="JSONL trace file")

    args = parser.parse_args()

    if args.command == "show":
        trace = Trace.from_jsonl(args.file)
        if args.as_json:
            print(json.dumps([m.to_dict() for m in trace], indent=2, ensure_ascii=False))
        else:
            _print_trace(trace)

    elif args.command == "diff":
        baseline = Trace.from_jsonl(args.baseline)
        current = Trace.from_jsonl(args.current)
        result = diff_traces(baseline, current)
        print(result.summary())
        if args.fail_on_regression and result.is_regression:
            print("\n❌ Regression detected!")
            sys.exit(1)

    elif args.command == "stats":
        trace = Trace.from_jsonl(args.file)
        _print_stats(trace)

    else:
        parser.print_help()


ROLE_ICONS = {
    "user": "👤",
    "assistant": "🤖",
    "tool": "🔧",
    "system": "⚙️",
}


def _print_trace(trace: Trace):
    print(f"Trace: {len(trace)} messages, {trace.step_count} steps")
    print(f"Tools: {trace.all_tool_names or '(none)'}")
    if trace.total_latency_ms:
        print(f"Latency: {trace.total_latency_ms:.0f}ms")
    print("---")
    for m in trace:
        icon = ROLE_ICONS.get(m.role, "?")
        latency = f" ({m.latency_ms:.0f}ms)" if m.latency_ms else ""
        if m.is_tool_call:
            for name in m.tool_names:
                args = m.tool_args(name)
                args_str = json.dumps(args, ensure_ascii=False) if args else ""
                print(f"{icon} {name}({args_str}){latency}")
        elif m.is_tool_response:
            content = (m.content or "")[:100]
            print(f"{icon} [{m.name}] → {content}")
        else:
            content = (m.content or "")[:200]
            print(f"{icon} {content}{latency}")


def _print_stats(trace: Trace):
    print(f"Messages:     {len(trace)}")
    print(f"Steps:        {trace.step_count}")
    print(f"Tool calls:   {len(trace.tool_calls)}")
    print(f"Unique tools: {len(set(trace.all_tool_names))}")
    if trace.all_tool_names:
        print(f"Tools used:   {', '.join(dict.fromkeys(trace.all_tool_names))}")
    if trace.total_latency_ms:
        print(f"Total latency: {trace.total_latency_ms:.0f}ms")
    final = trace.final_response
    if final and final.content:
        print(f"Final answer:  {final.content[:150]}...")


if __name__ == "__main__":
    main()
