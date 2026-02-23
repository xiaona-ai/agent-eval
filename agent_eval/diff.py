"""Trace comparison and regression detection.

Compare two traces to detect behavioral changes — useful for CI/CD
regression testing after prompt/model changes.
"""
from typing import Dict, List, Optional, Tuple

from .trace import Trace


class TraceDiff:
    """Result of comparing two traces."""

    def __init__(self):
        self.tool_changes: List[dict] = []
        self.output_changes: List[dict] = []
        self.performance_changes: List[dict] = []
        self.step_count_change: Optional[Tuple[int, int]] = None

    @property
    def has_changes(self) -> bool:
        return bool(self.tool_changes or self.output_changes or
                     self.performance_changes or self.step_count_change)

    @property
    def is_regression(self) -> bool:
        """Conservative check: any tool removal or significant perf degradation."""
        for tc in self.tool_changes:
            if tc["type"] == "removed":
                return True
        for pc in self.performance_changes:
            if pc.get("type") == "latency_increase" and pc.get("ratio", 1.0) > 2.0:
                return True
        return False

    def summary(self) -> str:
        lines = []
        if not self.has_changes:
            return "✅ No changes detected."

        if self.step_count_change:
            a, b = self.step_count_change
            icon = "⚠️" if b > a else "✅"
            lines.append(f"{icon} Steps: {a} → {b}")

        for tc in self.tool_changes:
            if tc["type"] == "added":
                lines.append(f"🆕 Tool added: {tc['tool']}")
            elif tc["type"] == "removed":
                lines.append(f"❌ Tool removed: {tc['tool']}")
            elif tc["type"] == "count_changed":
                lines.append(f"🔄 Tool '{tc['tool']}' calls: {tc['before']} → {tc['after']}")
            elif tc["type"] == "order_changed":
                lines.append(f"🔀 Tool call order changed: {tc['before']} → {tc['after']}")

        for oc in self.output_changes:
            if oc["type"] == "final_answer_changed":
                lines.append(f"📝 Final answer changed (similarity: {oc.get('similarity', 0):.0%})")
            elif oc["type"] == "final_answer_missing_in_current":
                lines.append("❌ Final answer missing in current trace.")
            elif oc["type"] == "final_answer_added_in_current":
                lines.append("🆕 Final answer added in current trace.")

        for pc in self.performance_changes:
            if pc["type"] == "latency_increase":
                lines.append(f"🐢 Latency increased: {pc['before']:.0f}ms → {pc['after']:.0f}ms ({pc['ratio']:.1f}x)")
            elif pc["type"] == "latency_decrease":
                lines.append(f"⚡ Latency decreased: {pc['before']:.0f}ms → {pc['after']:.0f}ms")

        return "\n".join(lines)


def diff_traces(baseline: Trace, current: Trace) -> TraceDiff:
    """Compare two traces and return a structured diff."""
    result = TraceDiff()

    # Step count
    if baseline.step_count != current.step_count:
        result.step_count_change = (baseline.step_count, current.step_count)

    # Tool changes
    _diff_tools(baseline, current, result)

    # Output changes
    _diff_output(baseline, current, result)

    # Performance changes
    _diff_performance(baseline, current, result)

    return result


def _diff_tools(baseline: Trace, current: Trace, result: TraceDiff):
    """Compare tool usage between traces."""
    base_tools = baseline.all_tool_names
    curr_tools = current.all_tool_names

    base_set = set(base_tools)
    curr_set = set(curr_tools)

    # Added/removed tools
    for tool in curr_set - base_set:
        result.tool_changes.append({"type": "added", "tool": tool})
    for tool in base_set - curr_set:
        result.tool_changes.append({"type": "removed", "tool": tool})

    # Count changes for shared tools
    for tool in base_set & curr_set:
        bc = base_tools.count(tool)
        cc = curr_tools.count(tool)
        if bc != cc:
            result.tool_changes.append({
                "type": "count_changed", "tool": tool,
                "before": bc, "after": cc,
            })

    # Order changes
    shared_base = [t for t in base_tools if t in curr_set]
    shared_curr = [t for t in curr_tools if t in base_set]
    if shared_base != shared_curr and shared_base and shared_curr:
        result.tool_changes.append({
            "type": "order_changed",
            "before": shared_base, "after": shared_curr,
        })


def _diff_output(baseline: Trace, current: Trace, result: TraceDiff):
    """Compare final outputs."""
    base_final = baseline.final_response
    curr_final = current.final_response
    base_text = base_final.text_content if base_final else ""
    curr_text = curr_final.text_content if curr_final else ""
    has_base = bool(base_text.strip())
    has_curr = bool(curr_text.strip())

    # Both traces have no final answer text: no output diff.
    if not has_base and not has_curr:
        return

    # One side has a final answer and the other does not.
    if has_base != has_curr:
        result.output_changes.append({
            "type": "final_answer_missing_in_current" if has_base else "final_answer_added_in_current",
            "before_preview": base_text[:100],
            "after_preview": curr_text[:100],
        })
        return

    # Both present: compare similarity.
    sim = _jaccard(base_text, curr_text)
    if sim < 0.9:
        result.output_changes.append({
            "type": "final_answer_changed",
            "similarity": sim,
            "before_preview": base_text[:100],
            "after_preview": curr_text[:100],
        })


def _diff_performance(baseline: Trace, current: Trace, result: TraceDiff):
    """Compare latency."""
    base_lat = baseline.total_latency_ms
    curr_lat = current.total_latency_ms

    if base_lat and curr_lat and base_lat > 0:
        ratio = curr_lat / base_lat
        if ratio > 1.5:
            result.performance_changes.append({
                "type": "latency_increase",
                "before": base_lat, "after": curr_lat, "ratio": ratio,
            })
        elif ratio < 0.7:
            result.performance_changes.append({
                "type": "latency_decrease",
                "before": base_lat, "after": curr_lat, "ratio": ratio,
            })


def _jaccard(a: str, b: str) -> float:
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)
