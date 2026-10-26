"""Hard-bounded backend execution and model-neutral decision validation."""

from __future__ import annotations

import json
import multiprocessing as mp
import threading
import time
from typing import Any

from .backends import ModelBackend, ScriptedBackend
from .protocols import Decision, DecisionType, Message, ToolCall, ToolSpec
from .tools import JSONLimits, ToolError, enforce_json_bounds

_DECISION_LIMITS = JSONLimits(max_bytes=1_000_000, max_items=5_000, max_depth=32, max_string_bytes=1_000_000)


class BackendExecutionError(RuntimeError):
    """A backend could not produce a trusted decision within its boundary."""


class BackendTimeoutError(BackendExecutionError):
    """The parent-enforced backend wall deadline expired."""


class BackendCancelledError(BackendExecutionError):
    """The parent cancellation event interrupted backend execution."""


def _normalize_decision(decision: Decision, max_bytes: int) -> dict[str, Any]:
    if not isinstance(decision, Decision):
        raise ValueError("backend returned a non-Decision object")
    encoded = json.dumps(decision.to_dict(), ensure_ascii=False, allow_nan=False).encode("utf-8")
    if len(encoded) > max_bytes:
        raise ValueError(f"backend decision exceeds the {max_bytes}-byte bound")
    normalized = json.loads(encoded)
    if not isinstance(normalized, dict):
        raise ValueError("backend decision must be a JSON object")
    return normalized


def _check_capabilities(backend: ModelBackend, tools: list[ToolSpec]) -> None:
    capabilities = backend.get_model_info()
    if tools and not (capabilities.structured_output or capabilities.native_tool_calling):
        raise ValueError("backend does not declare a structured tool-decision capability")


def _backend_process_main(
    connection: Any,
    backend: ModelBackend,
    messages: list[Message],
    tools: list[ToolSpec],
    max_bytes: int,
) -> None:
    try:
        _check_capabilities(backend, tools)
        decision = backend.chat(messages, tools)
        connection.send({"status": "ok", "decision": _normalize_decision(decision, max_bytes)})
    except Exception as exc:
        try:
            connection.send({"status": "error", "kind": type(exc).__name__, "message": str(exc)[:500]})
        except Exception:
            pass
    finally:
        connection.close()


def _parse_decision(payload: Any, tools: list[ToolSpec]) -> Decision:
    try:
        enforce_json_bounds(payload, limits=_DECISION_LIMITS, path="decision")
    except ToolError as exc:
        raise ValueError(str(exc)) from exc
    if not isinstance(payload, dict):
        raise ValueError("backend decision must be an object")
    allowed_keys = {"type", "rationale", "tool_calls", "answer", "verification_target"}
    if set(payload) - allowed_keys:
        raise ValueError("backend decision contains unknown fields")
    try:
        decision_type = DecisionType(payload["type"])
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError("backend decision has an invalid type") from exc
    rationale = payload.get("rationale", "")
    answer = payload.get("answer")
    verification_target = payload.get("verification_target")
    if not isinstance(rationale, str):
        raise ValueError("decision rationale must be a string")
    if answer is not None and not isinstance(answer, str):
        raise ValueError("decision answer must be a string or null")
    if verification_target is not None and not isinstance(verification_target, str):
        raise ValueError("decision verification_target must be a string or null")

    raw_calls = payload.get("tool_calls", [])
    if not isinstance(raw_calls, list) or len(raw_calls) > 100:
        raise ValueError("decision tool_calls must be a bounded array")
    if raw_calls and decision_type != DecisionType.TOOL_CALL:
        raise ValueError("decision type contradicts its tool_calls")
    if decision_type == DecisionType.TOOL_CALL and not raw_calls:
        raise ValueError("tool_call decision contains no calls")
    allowed_names = {tool.name for tool in tools}
    calls: list[ToolCall] = []
    for raw_call in raw_calls:
        if not isinstance(raw_call, dict) or set(raw_call) != {"id", "name", "arguments"}:
            raise ValueError("each tool call must contain exactly id, name, and arguments")
        call_id = raw_call["id"]
        name = raw_call["name"]
        arguments = raw_call["arguments"]
        if not isinstance(call_id, str) or not call_id:
            raise ValueError("tool-call id must be a non-empty string")
        if not isinstance(name, str) or name not in allowed_names:
            raise ValueError("backend requested an unregistered tool")
        if not isinstance(arguments, dict):
            raise ValueError("tool-call arguments must be an object")
        try:
            enforce_json_bounds(arguments, path="tool-call arguments")
        except ToolError as exc:
            raise ValueError(str(exc)) from exc
        calls.append(ToolCall(name=name, arguments=arguments, call_id=call_id))
    return Decision(
        type=decision_type,
        rationale=rationale,
        tool_calls=tuple(calls),
        answer=answer,
        verification_target=verification_target,
    )


class BackendExecutor:
    """Execute external/custom backends with a parent-enforced wall deadline.

    The exact built-in ScriptedBackend is finite, deterministic, and performs
    no I/O, so it runs inline. Every subclass or other backend is isolated in
    a spawn child and must be picklable. There is no unbounded thread fallback.
    """

    def __init__(self, max_decision_bytes: int = 1_000_000) -> None:
        if max_decision_bytes <= 0 or max_decision_bytes > 1_000_000:
            raise ValueError("max_decision_bytes must be in [1, 1000000]")
        self.max_decision_bytes = max_decision_bytes

    @staticmethod
    def _kill_process(process: Any) -> None:
        if process.is_alive():
            process.terminate()
            process.join(timeout=0.2)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(timeout=0.2)

    def _invoke(
        self,
        backend: ModelBackend,
        messages: list[Message],
        tools: list[ToolSpec],
        timeout: float,
        cancel_event: threading.Event | None,
    ) -> dict[str, Any]:
        methods = mp.get_all_start_methods()
        context: Any = mp.get_context("spawn" if "spawn" in methods else methods[0])
        read_connection, write_connection = context.Pipe(duplex=False)
        process = context.Process(
            target=_backend_process_main,
            args=(write_connection, backend, messages, tools, self.max_decision_bytes),
        )
        process.daemon = True
        started = time.monotonic()
        try:
            process.start()
            write_connection.close()
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    self._kill_process(process)
                    raise BackendCancelledError("backend cancelled")
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    self._kill_process(process)
                    raise BackendTimeoutError("backend wall-clock deadline exceeded")
                if read_connection.poll(min(0.02, remaining)):
                    try:
                        message = read_connection.recv()
                    except EOFError as exc:
                        raise BackendExecutionError("backend process closed without a decision") from exc
                    self._kill_process(process)
                    if message.get("status") == "ok":
                        decision = message["decision"]
                        if not isinstance(decision, dict):
                            raise BackendExecutionError("backend process returned an invalid decision envelope")
                        return decision
                    kind = message.get("kind", "backend error")
                    if kind in {"ValueError", "TypeError", "KeyError", "ToolError"}:
                        raise BackendExecutionError(f"invalid backend decision: {message.get('message', kind)}")
                    raise BackendExecutionError(f"backend failed: {kind}")
                if not process.is_alive():
                    if read_connection.poll(0):
                        continue
                    raise BackendExecutionError("backend process exited without returning a decision")
        except (BackendTimeoutError, BackendCancelledError, BackendExecutionError):
            raise
        except Exception as exc:
            raise BackendExecutionError(f"backend process failed: {type(exc).__name__}") from exc
        finally:
            self._kill_process(process)
            read_connection.close()
            try:
                write_connection.close()
            except OSError:
                pass

    def execute(
        self,
        backend: ModelBackend,
        messages: list[Message],
        tools: list[ToolSpec],
        timeout: float,
        cancel_event: threading.Event | None = None,
    ) -> Decision:
        if timeout <= 0:
            raise BackendTimeoutError("backend wall-clock deadline exhausted")
        if cancel_event is not None and cancel_event.is_set():
            raise BackendCancelledError("backend cancelled before execution")
        if type(backend) is ScriptedBackend:
            _check_capabilities(backend, tools)
            payload = _normalize_decision(backend.chat(messages, tools), self.max_decision_bytes)
        else:
            payload = self._invoke(backend, messages, tools, timeout, cancel_event)
        return _parse_decision(payload, tools)
