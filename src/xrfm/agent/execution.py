"""Validated, authorized, process-bounded tool execution."""

from __future__ import annotations

import json
import multiprocessing as mp
import threading
import time
from collections.abc import Callable
from typing import Any

from .protocols import ToolCall, ToolResult
from .safety import AuditLog, PermissionPolicy, sanitize_observation
from .tools import JSONLimits, ToolError, ToolRegistry, _validate_schema


def _tool_process_main(
    connection: Any, function: Callable[[dict[str, Any]], Any], arguments: dict[str, Any], limits: JSONLimits
) -> None:
    """Run in a child; only bounded JSON-compatible observations cross the pipe."""
    try:
        output = function(arguments)
        encoded = json.dumps(output, ensure_ascii=False, allow_nan=False, default=str).encode("utf-8")
        if len(encoded) > limits.max_bytes:
            connection.send(
                {"status": "error", "kind": "output", "message": "tool output exceeds total JSON byte bound"}
            )
        else:
            connection.send({"status": "ok", "output": json.loads(encoded)})
    except Exception as exc:  # the parent applies the public error policy
        try:
            connection.send({"status": "error", "kind": type(exc).__name__, "message": str(exc)[:500]})
        except Exception:
            pass
    finally:
        connection.close()


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        policy: PermissionPolicy | None = None,
        audit: AuditLog | None = None,
        confirm: Callable[[Any], bool] | None = None,
        limits: JSONLimits | None = None,
    ) -> None:
        self.registry = registry
        self.policy = policy or PermissionPolicy()
        self.audit = audit or AuditLog()
        self.confirm = confirm
        self.limits = limits or JSONLimits()

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
        function: Callable[[dict[str, Any]], Any],
        arguments: dict[str, Any],
        timeout: float,
        cancel_event: threading.Event | None,
    ) -> Any:
        methods = mp.get_all_start_methods()
        # spawn avoids forking a multi-threaded ASGI process (which can
        # deadlock the child). Registered callables must therefore be
        # picklable; startup/pickling failures are explicit ToolErrors.
        context: Any = mp.get_context("spawn" if "spawn" in methods else methods[0])
        read_connection, write_connection = context.Pipe(duplex=False)
        process = context.Process(target=_tool_process_main, args=(write_connection, function, arguments, self.limits))
        process.daemon = True
        started = time.monotonic()
        try:
            process.start()
            write_connection.close()
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    self._kill_process(process)
                    raise ToolError("tool cancelled")
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    self._kill_process(process)
                    raise ToolError("tool timed out")
                if read_connection.poll(min(0.02, remaining)):
                    try:
                        message = read_connection.recv()
                    except EOFError as exc:
                        raise ToolError("tool process closed without a result") from exc
                    self._kill_process(process)
                    if message.get("status") == "ok":
                        return message["output"]
                    kind = message.get("kind", "tool error")
                    text = message.get("message", "tool failed")
                    if kind in {"ToolError", "ValueError", "KeyError", "TypeError", "output"}:
                        raise ToolError(text)
                    raise ToolError(f"tool failed: {kind}")
                if not process.is_alive():
                    if read_connection.poll(0):
                        continue
                    raise ToolError("tool process exited without returning a result")
        except Exception as exc:
            # This intentionally also catches process/pipe startup failures;
            # there is no silent fallback to an unkillable worker thread.
            if isinstance(exc, ToolError):
                raise
            raise ToolError(f"tool process failed: {type(exc).__name__}") from exc
        finally:
            self._kill_process(process)
            read_connection.close()
            try:
                write_connection.close()
            except OSError:
                pass

    def execute(self, call: ToolCall, cancel_event: threading.Event | None = None) -> ToolResult:
        started = time.perf_counter()
        result: ToolResult
        try:
            registered = self.registry.get(call.name)
            _validate_schema(registered.spec.input_schema, call.arguments, limits=self.limits)
            auth = self.policy.authorize(registered.spec)
            self.audit.record(
                "authorization", call.name, call_id=call.call_id, allowed=auth.allowed, reason=auth.reason
            )
            if not auth.allowed:
                result = ToolResult(call.call_id, call.name, False, error=auth.reason, denied=True)
            elif auth.requires_confirmation and (self.confirm is None or not self.confirm(registered.spec)):
                result = ToolResult(call.call_id, call.name, False, error="user confirmation required", denied=True)
            else:
                output = self._invoke(
                    registered.function, call.arguments, registered.spec.timeout_seconds, cancel_event
                )
                if registered.spec.output_schema is not None:
                    _validate_schema(registered.spec.output_schema, output, path="output", limits=self.limits)
                safe = sanitize_observation(output)
                result = ToolResult(call.call_id, call.name, True, output=safe)
        except (ToolError, KeyError, ValueError, TypeError) as exc:
            self.audit.record("execution_error", call.name, call_id=call.call_id, error=str(exc))
            result = ToolResult(call.call_id, call.name, False, error=str(exc))
        except Exception as exc:  # containment boundary: tool failures never crash the agent
            self.audit.record("unexpected_tool_error", call.name, call_id=call.call_id, error=type(exc).__name__)
            result = ToolResult(call.call_id, call.name, False, error=f"tool failed: {type(exc).__name__}")
        result.duration_ms = (time.perf_counter() - started) * 1000
        self.audit.record(
            "execution_result",
            call.name,
            call_id=call.call_id,
            ok=result.ok,
            denied=result.denied,
            duration_ms=round(result.duration_ms, 3),
            error=result.error,
        )
        return result
