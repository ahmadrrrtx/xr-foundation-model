"""XR-native, model-neutral protocol records for agent execution."""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable


class DecisionType(str, Enum):
    ANSWER = "answer"
    TOOL_CALL = "tool_call"
    ASK_USER = "ask_user"
    REFUSE = "refuse"
    VERIFY = "verify"
    COMPLETE = "complete"


class AgentPhase(str, Enum):
    REQUEST = "request"
    CONTEXT = "context"
    DECIDE = "decide"
    AUTHORIZE = "authorize"
    EXECUTE = "execute"
    OBSERVE = "observe"
    VERIFY = "verify"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True)
class Message:
    role: str
    content: str
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] | None = None
    permissions: tuple[str, ...] = ()
    risk: str = "low"
    timeout_seconds: float = 10.0
    requires_confirmation: bool = False
    mutating: bool = False
    environment: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["permissions"] = list(self.permissions)
        return result


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    call_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.call_id, "name": self.name, "arguments": self.arguments}


@dataclass
class ToolResult:
    call_id: str
    name: str
    ok: bool
    output: Any = None
    error: str | None = None
    denied: bool = False
    duration_ms: float = 0.0

    def as_observation(self) -> str:
        payload = {"tool": self.name, "ok": self.ok, "output": self.output}
        if self.error:
            payload["error"] = self.error
        if self.denied:
            payload["denied"] = True
        return json.dumps(payload, ensure_ascii=False, default=str)


@dataclass(frozen=True)
class Decision:
    type: DecisionType
    rationale: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    answer: str | None = None
    verification_target: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "rationale": self.rationale,
            "tool_calls": [c.to_dict() for c in self.tool_calls],
            "answer": self.answer,
            "verification_target": self.verification_target,
        }


@dataclass(frozen=True)
class AgentEvent:
    phase: AgentPhase
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {"phase": self.phase.value, "message": self.message, "data": self.data, "timestamp": self.timestamp}


@dataclass
class AgentState:
    request: str
    phase: AgentPhase = AgentPhase.REQUEST
    messages: list[Message] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    observations: list[ToolResult] = field(default_factory=list)
    events: list[AgentEvent] = field(default_factory=list)
    step: int = 0
    final_answer: str | None = None
    failed: bool = False

    def emit(self, phase: AgentPhase, message: str, **data: Any) -> None:
        self.phase = phase
        self.events.append(AgentEvent(phase, message, data))


ToolFunction = Callable[[dict[str, Any]], Any]
