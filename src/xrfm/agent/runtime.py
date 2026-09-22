"""XR bounded agent execution state machine."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterator

from .backends import ModelBackend
from .execution import ToolExecutor
from .protocols import AgentEvent, AgentPhase, AgentState, DecisionType, Message


@dataclass(frozen=True)
class RuntimeLimits:
    max_steps: int = 8
    max_wall_seconds: float = 60.0
    max_tool_calls: int = 12


class AgentRuntime:
    """Model-neutral REQUEST→DECIDE→EXECUTE→OBSERVE→VERIFY loop."""

    def __init__(self, backend: ModelBackend, executor: ToolExecutor, limits: RuntimeLimits | None = None) -> None:
        self.backend = backend
        self.executor = executor
        self.limits = limits or RuntimeLimits()

    def run(self, request: str, on_event: Callable[[AgentEvent], None] | None = None) -> AgentState:
        state = AgentState(request=request, messages=[Message("user", request)])
        started = time.monotonic()

        def emit(phase: AgentPhase, message: str, **data: object) -> None:
            state.emit(phase, message, **data)
            if on_event:
                on_event(state.events[-1])

        emit(AgentPhase.REQUEST, "Request accepted")
        tool_signatures: set[str] = set()
        while state.step < self.limits.max_steps and time.monotonic() - started < self.limits.max_wall_seconds:
            state.step += 1
            emit(AgentPhase.CONTEXT, "Context assembled", step=state.step, message_count=len(state.messages))
            decision = self.backend.chat(state.messages, self.executor.registry.specs())
            state.decisions.append(decision)
            emit(AgentPhase.DECIDE, "Decision produced", decision=decision.to_dict())
            if decision.type in (DecisionType.ANSWER, DecisionType.COMPLETE):
                state.final_answer = decision.answer or ""
                emit(AgentPhase.VERIFY, "Final answer accepted", answer_length=len(state.final_answer))
                emit(AgentPhase.COMPLETE, "Execution complete")
                return state
            if decision.type == DecisionType.REFUSE:
                state.final_answer = decision.answer or "I cannot perform that request."
                emit(AgentPhase.COMPLETE, "Request refused")
                return state
            if decision.type == DecisionType.ASK_USER:
                state.final_answer = decision.answer or "I need more information from you."
                emit(AgentPhase.COMPLETE, "Clarification required")
                return state
            if decision.type != DecisionType.TOOL_CALL or not decision.tool_calls:
                state.failed = True
                state.final_answer = "The model did not produce a valid terminal decision."
                emit(AgentPhase.FAILED, "Invalid decision")
                return state
            if len(state.observations) + len(decision.tool_calls) > self.limits.max_tool_calls:
                state.failed = True
                state.final_answer = "The tool-call budget was exceeded."
                emit(AgentPhase.FAILED, "Tool-call budget exceeded")
                return state
            for call in decision.tool_calls:
                signature = call.name + ":" + repr(sorted(call.arguments.items()))
                if signature in tool_signatures:
                    state.failed = True
                    state.final_answer = "The runtime stopped a repeated tool call."
                    emit(AgentPhase.FAILED, "Repeated tool call detected", tool=call.name)
                    return state
                tool_signatures.add(signature)
                emit(AgentPhase.AUTHORIZE, "Authorizing tool call", tool=call.name, call_id=call.call_id)
                emit(AgentPhase.EXECUTE, "Executing tool call", tool=call.name, call_id=call.call_id)
                result = self.executor.execute(call)
                state.observations.append(result)
                emit(AgentPhase.OBSERVE, "Tool observation received", tool=call.name, ok=result.ok, denied=result.denied)
                state.messages.append(Message("assistant", "", tool_calls=[call.to_dict()]))
                state.messages.append(Message("tool", result.as_observation(), name=call.name, tool_call_id=call.call_id))
            emit(AgentPhase.VERIFY, "Observation available for verification", observation_count=len(state.observations))
        state.failed = True
        state.final_answer = "The agent stopped because its execution budget was exhausted."
        emit(AgentPhase.FAILED, "Runtime budget exhausted", steps=state.step)
        return state

    def stream(self, request: str) -> Iterator[AgentEvent]:
        events: list[AgentEvent] = []
        self.run(request, events.append)
        yield from events
