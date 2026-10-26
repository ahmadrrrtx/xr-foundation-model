"""Streaming, bounded REQUEST→DECIDE→EXECUTE→OBSERVE→VERIFY runtime."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass

from .backend_execution import BackendCancelledError, BackendExecutor, BackendTimeoutError
from .backends import ModelBackend
from .execution import ToolExecutor
from .protocols import AgentEvent, AgentPhase, AgentState, Decision, DecisionType, Message


@dataclass(frozen=True)
class RuntimeLimits:
    max_steps: int = 8
    max_wall_seconds: float = 60.0
    max_tool_calls: int = 12

    def __post_init__(self) -> None:
        for name, minimum in (("max_steps", 1), ("max_tool_calls", 0)):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{name} must be an integer greater than or equal to {minimum}")
        wall_seconds = self.max_wall_seconds
        if (
            not isinstance(wall_seconds, (int, float))
            or isinstance(wall_seconds, bool)
            or not math.isfinite(wall_seconds)
            or wall_seconds <= 0
        ):
            raise ValueError("max_wall_seconds must be a finite positive number")


class AgentRuntime:
    """Model-neutral runtime with per-run state and incremental events."""

    def __init__(
        self,
        backend: ModelBackend,
        executor: ToolExecutor,
        limits: RuntimeLimits | None = None,
        backend_executor: BackendExecutor | None = None,
    ) -> None:
        self.backend = backend
        self.executor = executor
        self.limits = limits or RuntimeLimits()
        self.backend_executor = backend_executor or BackendExecutor()

    def _execute(
        self, request: str, cancel_event: threading.Event | None = None
    ) -> Generator[AgentEvent, None, AgentState]:
        state = AgentState(request=request, messages=[Message("user", request)])
        started = time.monotonic()

        def emit(phase: AgentPhase, message: str, **data: object) -> AgentEvent:
            state.emit(phase, message, **data)
            return state.events[-1]

        def stop_reason() -> str | None:
            if cancel_event is not None and cancel_event.is_set():
                return "cancelled"
            if time.monotonic() - started >= self.limits.max_wall_seconds:
                return "wall-clock budget exhausted"
            return None

        def failed(reason: str, **data: object) -> tuple[AgentState, AgentEvent]:
            state.failed = True
            state.final_answer = reason
            return state, emit(AgentPhase.FAILED, reason, **data)

        yield emit(AgentPhase.REQUEST, "Request accepted")
        tool_signatures: set[str] = set()
        while state.step < self.limits.max_steps:
            reason = stop_reason()
            if reason is not None:
                state, event = failed(f"The agent stopped: {reason}.", steps=state.step)
                yield event
                return state
            state.step += 1
            yield emit(
                AgentPhase.CONTEXT,
                "Context assembled",
                step=state.step,
                message_count=len(state.messages),
                schema_version=self.executor.registry.schema_version,
            )
            specs = self.executor.registry.specs()
            reason = stop_reason()
            if reason is not None:
                state, event = failed(f"The agent stopped: {reason}.", steps=state.step)
                yield event
                return state
            remaining = self.limits.max_wall_seconds - (time.monotonic() - started)
            try:
                decision = self.backend_executor.execute(
                    self.backend,
                    state.messages,
                    specs,
                    timeout=remaining,
                    cancel_event=cancel_event,
                )
            except BackendCancelledError:
                state, event = failed("The agent stopped: cancelled.", steps=state.step)
                yield event
                return state
            except BackendTimeoutError:
                state, event = failed("The agent stopped: wall-clock budget exhausted.", steps=state.step)
                yield event
                return state
            except Exception as exc:  # provider failures are contained and fail closed
                state, event = failed(f"Backend decision failed ({type(exc).__name__}).")
                yield event
                return state
            reason = stop_reason()
            if reason is not None:
                state, event = failed(f"The agent stopped: {reason}.", steps=state.step)
                yield event
                return state
            if not isinstance(decision, Decision):
                state, event = failed("The backend returned an invalid decision object.")
                yield event
                return state
            state.decisions.append(decision)
            yield emit(AgentPhase.DECIDE, "Decision produced", decision=decision.to_dict())
            if decision.type in (DecisionType.ANSWER, DecisionType.COMPLETE):
                state.final_answer = decision.answer or ""
                yield emit(AgentPhase.VERIFY, "Final answer accepted", answer_length=len(state.final_answer))
                yield emit(AgentPhase.COMPLETE, "Execution complete")
                return state
            if decision.type == DecisionType.REFUSE:
                state.final_answer = decision.answer or "I cannot perform that request."
                yield emit(AgentPhase.COMPLETE, "Request refused")
                return state
            if decision.type == DecisionType.ASK_USER:
                state.final_answer = decision.answer or "I need more information from you."
                yield emit(AgentPhase.COMPLETE, "Clarification required")
                return state
            if decision.type != DecisionType.TOOL_CALL or not decision.tool_calls:
                state, event = failed("The model did not produce a valid terminal decision.")
                yield event
                return state
            if len(state.observations) + len(decision.tool_calls) > self.limits.max_tool_calls:
                state, event = failed("The tool-call budget was exceeded.")
                yield event
                return state
            for call in decision.tool_calls:
                reason = stop_reason()
                if reason is not None:
                    state, event = failed(f"The agent stopped: {reason}.", steps=state.step)
                    yield event
                    return state
                signature = call.name + ":" + repr(sorted(call.arguments.items()))
                if signature in tool_signatures:
                    state, event = failed("The runtime stopped a repeated tool call.", tool=call.name)
                    yield event
                    return state
                tool_signatures.add(signature)
                yield emit(AgentPhase.AUTHORIZE, "Authorizing tool call", tool=call.name, call_id=call.call_id)
                yield emit(AgentPhase.EXECUTE, "Executing tool call", tool=call.name, call_id=call.call_id)
                result = self.executor.execute(call, cancel_event=cancel_event)
                if cancel_event is not None and cancel_event.is_set():
                    state, event = failed("The agent stopped: cancelled.", tool=call.name)
                    yield event
                    return state
                state.observations.append(result)
                yield emit(
                    AgentPhase.OBSERVE, "Tool observation received", tool=call.name, ok=result.ok, denied=result.denied
                )
                state.messages.append(Message("assistant", "", tool_calls=[call.to_dict()]))
                state.messages.append(
                    Message("tool", result.as_observation(), name=call.name, tool_call_id=call.call_id)
                )
            yield emit(
                AgentPhase.VERIFY, "Observation available for verification", observation_count=len(state.observations)
            )

        state, event = failed("The agent stopped because its execution budget was exhausted.", steps=state.step)
        yield event
        return state

    def run(
        self,
        request: str,
        on_event: Callable[[AgentEvent], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> AgentState:
        """Consume the streaming state machine and return its final state."""
        events = self._execute(request, cancel_event=cancel_event)
        try:
            while True:
                try:
                    event = next(events)
                except StopIteration as completed:
                    state = completed.value
                    if not isinstance(state, AgentState):  # defensive generator contract
                        raise RuntimeError("agent execution ended without a final state")
                    return state
                if on_event is not None:
                    on_event(event)
        finally:
            events.close()

    def stream(self, request: str, cancel_event: threading.Event | None = None) -> Iterator[AgentEvent]:
        """Return live events; no full-run buffering is performed."""
        return self._execute(request, cancel_event=cancel_event)
