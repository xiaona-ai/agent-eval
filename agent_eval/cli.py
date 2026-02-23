"""CLI for agent-eval."""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from .assertions import EvalFailure
from .consistency import ConsistencyReport, assert_consistency
from .cost import _sum_cost, _sum_tokens
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
    p_diff.add_argument("--json", action="store_true", dest="as_json", help="Output as JSON")

    # stats
    p_stats = sub.add_parser("stats", help="Show trace statistics")
    p_stats.add_argument("file", help="JSONL trace file")
    p_stats.add_argument("--json", action="store_true", dest="as_json", help="Output as JSON")

    # cost
    p_cost = sub.add_parser("cost", help="Show token/cost usage and enforce budget assertions")
    p_cost.add_argument("file", help="JSONL trace file")
    p_cost.add_argument("--max-tokens", type=int, help="Maximum allowed total tokens")
    p_cost.add_argument("--max-usd", type=float, help="Maximum allowed total cost in USD")
    p_cost.add_argument("--pricing", help="Path to pricing JSON file")
    p_cost.add_argument("--strict", action="store_true", help="Error on unknown models in pricing")

    # consistency
    p_consistency = sub.add_parser("consistency", help="Evaluate consistency across 2+ traces")
    p_consistency.add_argument("files", nargs="+", help="Trace files (JSONL), at least 2")
    p_consistency.add_argument("--min-tool-consistency", type=float, dest="min_tool_consistency")
    p_consistency.add_argument("--min-answer-consistency", type=float, dest="min_answer_consistency")
    p_consistency.add_argument("--max-step-variance", type=float, dest="max_step_variance")

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
        if args.as_json:
            print(json.dumps(_diff_to_dict(result), indent=2, ensure_ascii=False))
        else:
            print(result.summary())
        if args.fail_on_regression and result.is_regression:
            print("\n❌ Regression detected!")
            sys.exit(1)

    elif args.command == "stats":
        trace = Trace.from_jsonl(args.file)
        if args.as_json:
            print(json.dumps(_stats_dict(trace), indent=2, ensure_ascii=False))
        else:
            _print_stats(trace)

    elif args.command == "cost":
        _run_cost(args)

    elif args.command == "consistency":
        _run_consistency(args)

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
            content = m.text_content[:100]
            print(f"{icon} [{m.name}] → {content}")
        else:
            content = m.text_content[:200]
            print(f"{icon} {content}{latency}")


def _print_stats(trace: Trace):
    stats = _stats_dict(trace)
    print(f"Messages:     {stats['messages']}")
    print(f"Steps:        {stats['steps']}")
    print(f"Tool calls:   {stats['tool_calls']}")
    print(f"Unique tools: {stats['unique_tools']}")
    if stats["tools_used"]:
        print(f"Tools used:   {', '.join(stats['tools_used'])}")
    if stats["total_latency_ms"] is not None:
        print(f"Total latency: {stats['total_latency_ms']:.0f}ms")
    final = stats["final_answer"]
    if final:
        preview = f"{final[:150]}..." if len(final) > 150 else final
        print(f"Final answer:  {preview}")


def _stats_dict(trace: Trace) -> Dict[str, Any]:
    final = trace.final_response
    return {
        "messages": len(trace),
        "steps": trace.step_count,
        "tool_calls": len(trace.tool_calls),
        "unique_tools": len(set(trace.all_tool_names)),
        "tools_used": list(dict.fromkeys(trace.all_tool_names)),
        "total_latency_ms": trace.total_latency_ms,
        "final_answer": final.text_content if final else "",
    }


def _diff_to_dict(result) -> Dict[str, Any]:
    return {
        "has_changes": result.has_changes,
        "is_regression": result.is_regression,
        "step_count_change": list(result.step_count_change) if result.step_count_change else None,
        "tool_changes": result.tool_changes,
        "output_changes": result.output_changes,
        "performance_changes": result.performance_changes,
        "summary": result.summary(),
    }


def _load_pricing(path: str) -> Dict[str, Dict[str, float]]:
    pricing = json.loads(Path(path).read_text())
    if not isinstance(pricing, dict):
        raise ValueError("pricing must be a JSON object of model -> rates")
    return pricing


def _run_cost(args):
    try:
        trace = Trace.from_jsonl(args.file)
    except Exception as exc:
        print(f"❌ Failed to read trace: {exc}", file=sys.stderr)
        sys.exit(1)

    pricing: Optional[Dict[str, Dict[str, float]]] = None
    if args.pricing:
        try:
            pricing = _load_pricing(args.pricing)
        except Exception as exc:
            print(f"❌ Failed to read pricing: {exc}", file=sys.stderr)
            sys.exit(1)

    if args.max_usd is not None and pricing is None:
        print("❌ --max-usd requires --pricing.", file=sys.stderr)
        sys.exit(1)
    if args.strict and pricing is None:
        print("❌ --strict requires --pricing.", file=sys.stderr)
        sys.exit(1)

    try:
        total_tokens = _sum_tokens(trace)
        total_cost = _sum_cost(trace, pricing, strict=args.strict) if pricing is not None else None
    except EvalFailure as exc:
        print(f"❌ {exc}", file=sys.stderr)
        sys.exit(1)

    print("Cost Summary")
    print(f"  Total tokens: {total_tokens}")
    if total_cost is not None:
        print(f"  Total cost:   ${total_cost:.6f}")
    else:
        print("  Total cost:   n/a (no pricing provided)")
    if args.max_tokens is not None:
        print(f"  Max tokens:   {args.max_tokens}")
    if args.max_usd is not None:
        print(f"  Max cost:     ${args.max_usd:.6f}")

    failed = False
    if args.max_tokens is not None and total_tokens > args.max_tokens:
        print(
            f"❌ [total_tokens] Total tokens {total_tokens} exceeds limit of {args.max_tokens}.",
            file=sys.stderr,
        )
        failed = True

    if args.max_usd is not None and total_cost is not None and total_cost > args.max_usd:
        print(
            f"❌ [total_cost] Total cost ${total_cost:.6f} exceeds limit of ${args.max_usd:.6f}.",
            file=sys.stderr,
        )
        failed = True

    if failed:
        sys.exit(1)


def _run_consistency(args):
    if len(args.files) < 2:
        print("❌ consistency requires at least 2 trace files.", file=sys.stderr)
        sys.exit(1)

    traces = []
    for path in args.files:
        try:
            traces.append(Trace.from_jsonl(path))
        except Exception as exc:
            print(f"❌ Failed to read trace {path!r}: {exc}", file=sys.stderr)
            sys.exit(1)

    try:
        report = ConsistencyReport(traces)
    except ValueError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        sys.exit(1)

    print(report.summary())

    try:
        assert_consistency(
            traces,
            min_tool_consistency=args.min_tool_consistency,
            min_answer_consistency=args.min_answer_consistency,
            max_step_variance=args.max_step_variance,
        )
    except EvalFailure as exc:
        print(f"\n❌ {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
