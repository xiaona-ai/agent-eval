"""Trace and Message data structures for agent evaluation.

A Trace is a sequence of Messages representing an agent's execution.
Messages follow the OpenAI chat format (role/content/tool_calls/name).
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class Message:
    """A single message in an agent trace."""

    __slots__ = ("role", "content", "name", "tool_calls", "tool_call_id",
                 "timestamp", "latency_ms", "metadata")

    def __init__(self, data: dict):
        self.role: str = data.get("role", "")
        self.content: Optional[str] = data.get("content")
        self.name: Optional[str] = data.get("name")  # tool name for role=tool
        self.tool_calls: Optional[List[dict]] = data.get("tool_calls")
        self.tool_call_id: Optional[str] = data.get("tool_call_id")
        self.timestamp: Optional[str] = data.get("timestamp")
        self.latency_ms: Optional[float] = data.get("latency_ms")
        self.metadata: Dict[str, Any] = data.get("metadata", {})

    @property
    def is_assistant(self) -> bool:
        return self.role == "assistant"

    @property
    def is_tool_call(self) -> bool:
        return self.is_assistant and bool(self.tool_calls)

    @property
    def is_tool_response(self) -> bool:
        return self.role == "tool"

    @property
    def is_user(self) -> bool:
        return self.role == "user"

    @property
    def tool_names(self) -> List[str]:
        """Extract tool names from tool_calls."""
        if not self.tool_calls:
            return []
        names = []
        for tc in self.tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "") if isinstance(fn, dict) else ""
            if name:
                names.append(name)
        return names

    def tool_args(self, tool_name: str) -> Optional[dict]:
        """Get arguments for a specific tool call."""
        if not self.tool_calls:
            return None
        for tc in self.tool_calls:
            fn = tc.get("function", {})
            if isinstance(fn, dict) and fn.get("name") == tool_name:
                args = fn.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        return json.loads(args)
                    except json.JSONDecodeError:
                        return {"_raw": args}
                return args
        return None

    def to_dict(self) -> dict:
        d = {"role": self.role}
        if self.content is not None:
            d["content"] = self.content
        if self.name is not None:
            d["name"] = self.name
        if self.tool_calls is not None:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.timestamp is not None:
            d["timestamp"] = self.timestamp
        if self.latency_ms is not None:
            d["latency_ms"] = self.latency_ms
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    def __repr__(self) -> str:
        content_preview = ""
        if self.content:
            content_preview = self.content[:50] + ("..." if len(self.content) > 50 else "")
        elif self.tool_calls:
            content_preview = f"tool_calls={self.tool_names}"
        return f"Message(role={self.role!r}, {content_preview})"


class Trace:
    """An ordered sequence of messages from an agent execution.

    Can be loaded from JSONL files, lists of dicts, or constructed manually.
    """

    def __init__(self, messages: Optional[List[Union[dict, Message]]] = None,
                 trace_id: Optional[str] = None, metadata: Optional[dict] = None):
        self.messages: List[Message] = []
        self.trace_id = trace_id
        self.metadata = metadata or {}
        if messages:
            for m in messages:
                if isinstance(m, Message):
                    self.messages.append(m)
                else:
                    self.messages.append(Message(m))

    @classmethod
    def from_jsonl(cls, path: Union[str, Path], **kwargs) -> "Trace":
        """Load a trace from a JSONL file (one message per line)."""
        p = Path(path)
        messages = []
        for line in p.read_text().splitlines():
            line = line.strip()
            if line:
                messages.append(json.loads(line))
        return cls(messages, **kwargs)

    @classmethod
    def from_messages(cls, messages: List[dict], **kwargs) -> "Trace":
        """Create a trace from a list of OpenAI-style message dicts."""
        return cls(messages, **kwargs)

    def to_jsonl(self, path: Union[str, Path]):
        """Save trace to a JSONL file."""
        p = Path(path)
        with open(p, "w") as f:
            for m in self.messages:
                f.write(json.dumps(m.to_dict(), ensure_ascii=False) + "\n")

    @property
    def assistant_messages(self) -> List[Message]:
        return [m for m in self.messages if m.is_assistant]

    @property
    def tool_calls(self) -> List[Message]:
        return [m for m in self.messages if m.is_tool_call]

    @property
    def tool_responses(self) -> List[Message]:
        return [m for m in self.messages if m.is_tool_response]

    @property
    def all_tool_names(self) -> List[str]:
        """All tool names called in order."""
        names = []
        for m in self.tool_calls:
            names.extend(m.tool_names)
        return names

    @property
    def final_response(self) -> Optional[Message]:
        """The last assistant message (non-tool-call)."""
        for m in reversed(self.messages):
            if m.is_assistant and not m.is_tool_call:
                return m
        return None

    @property
    def step_count(self) -> int:
        """Number of assistant messages (actions taken)."""
        return len(self.assistant_messages)

    @property
    def total_latency_ms(self) -> Optional[float]:
        """Sum of all latency_ms values."""
        latencies = [m.latency_ms for m in self.messages if m.latency_ms is not None]
        return sum(latencies) if latencies else None

    def __len__(self) -> int:
        return len(self.messages)

    def __iter__(self):
        return iter(self.messages)

    def __getitem__(self, idx):
        return self.messages[idx]

    def __repr__(self) -> str:
        return f"Trace(messages={len(self.messages)}, tools={self.all_tool_names})"
