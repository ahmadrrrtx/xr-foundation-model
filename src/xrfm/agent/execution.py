"""Validated, authorized, bounded tool execution."""
from __future__ import annotations

import concurrent.futures
import time
from typing import Any, Callable

from .protocols import ToolCall, ToolResult
from .safety import AuditLog, PermissionPolicy, sanitize_observation
from .tools import ToolError, ToolRegistry, _validate_schema


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, policy: PermissionPolicy | None = None, audit: AuditLog | None = None, confirm: Callable[[Any], bool] | None = None) -> None:
        self.registry = registry
        self.policy = policy or PermissionPolicy()
        self.audit = audit or AuditLog()
        self.confirm = confirm

    def execute(self, call: ToolCall) -> ToolResult:
        started = time.perf_counter()
        try:
            registered = self.registry.get(call.name)
            _validate_schema(registered.spec.input_schema, call.arguments)
            auth = self.policy.authorize(registered.spec)
            self.audit.record("authorization", call.name, allowed=auth.allowed, reason=auth.reason)
            if not auth.allowed:
                return ToolResult(call.call_id, call.name, False, error=auth.reason, denied=True)
            if auth.requires_confirmation and (self.confirm is None or not self.confirm(registered.spec)):
                return ToolResult(call.call_id, call.name, False, error="user confirmation required", denied=True)
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(registered.function, call.arguments)
                output = future.result(timeout=registered.spec.timeout_seconds)
            safe = sanitize_observation(output)
            result = ToolResult(call.call_id, call.name, True, output=safe)
            self.audit.record("executed", call.name, ok=True)
            return result
        except (ToolError, KeyError, ValueError, TypeError) as exc:
            self.audit.record("execution_error", call.name, error=str(exc))
            return ToolResult(call.call_id, call.name, False, error=str(exc))
        except concurrent.futures.TimeoutError:
            self.audit.record("timeout", call.name)
            return ToolResult(call.call_id, call.name, False, error="tool timed out")
        except Exception as exc:  # containment boundary: tool failures never crash the agent
            self.audit.record("unexpected_tool_error", call.name, error=type(exc).__name__)
            return ToolResult(call.call_id, call.name, False, error=f"tool failed: {type(exc).__name__}")
