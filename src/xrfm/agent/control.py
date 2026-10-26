"""Thread-safe, application-scoped agent kill and permission controls."""

from __future__ import annotations

import threading
from typing import Any

from .safety import PermissionPolicy


class AgentControl:
    """Own operational controls, never request messages, observations, or audits.

    Registration and kill/revocation decisions share one lock so a run cannot
    slip between a killed-state check and cancellation registration. Event
    notifications happen outside the lock. Revocation intentionally cancels all
    active runs: this is conservative and avoids a policy-update race.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._killed = False
        self._revoked_permissions: set[str] = set()
        self._revoked_tools: set[str] = set()
        self._active_runs: dict[int, threading.Event] = {}
        self._next_registration = 0

    @staticmethod
    def _name(value: str | None, label: str) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip() or len(value.strip()) > 128:
            raise ValueError(f"{label} must be a non-empty string of at most 128 characters")
        return value.strip()

    def _snapshot_locked(self) -> dict[str, Any]:
        return {
            "killed": self._killed,
            "revoked_permissions": sorted(self._revoked_permissions),
            "revoked_tools": sorted(self._revoked_tools),
            "active_runs": len(self._active_runs),
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_locked()

    def assert_active(self) -> None:
        with self._lock:
            if self._killed:
                raise RuntimeError("agent kill switch is engaged")

    def register(self, cancel_event: threading.Event) -> int:
        """Atomically reject a killed runtime or register its cancellation."""
        with self._lock:
            if self._killed:
                raise RuntimeError("agent kill switch is engaged")
            self._next_registration += 1
            registration = self._next_registration
            self._active_runs[registration] = cancel_event
            return registration

    def unregister(self, registration: int) -> None:
        with self._lock:
            self._active_runs.pop(registration, None)

    def apply_policy(self, policy: PermissionPolicy) -> None:
        """Apply the current revocation snapshot to a fresh request policy."""
        with self._lock:
            permissions = set(self._revoked_permissions)
            tools = set(self._revoked_tools)
        policy.allowed_permissions.difference_update(permissions)
        policy.denied_tools.update(tools)

    def kill(self) -> dict[str, Any]:
        with self._lock:
            self._killed = True
            cancellations = list(self._active_runs.values())
        for cancel_event in cancellations:
            cancel_event.set()
        return self.snapshot()

    def resume(self) -> dict[str, Any]:
        """Clear only the kill switch; permission revocations remain intact."""
        with self._lock:
            self._killed = False
        return self.snapshot()

    def revoke(self, permission: str | None = None, tool: str | None = None) -> dict[str, Any]:
        permission = self._name(permission, "permission")
        tool = self._name(tool, "tool")
        if permission is None and tool is None:
            raise ValueError("at least one permission or tool must be supplied")
        with self._lock:
            if permission is not None:
                self._revoked_permissions.add(permission)
            if tool is not None:
                self._revoked_tools.add(tool)
            cancellations = list(self._active_runs.values())
        for cancel_event in cancellations:
            cancel_event.set()
        return self.snapshot()

    def close(self) -> None:
        """Stop active work during application shutdown."""
        self.kill()
