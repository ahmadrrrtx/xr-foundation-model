"""Authenticated, per-request XR agent endpoints."""

from __future__ import annotations

import asyncio
import hmac
import json
import os
import threading
from collections.abc import AsyncIterator, Iterator
from contextlib import suppress
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import StreamingResponse

from api.agent_config import env_bool
from api.schemas import AgentRequest, AgentResponse, AgentRevokeRequest
from api.state import get_state
from xrfm.agent import (
    AgentControl,
    AgentEvent,
    AgentRuntime,
    AuditLog,
    ModelBackend,
    OpenAICompatibleBackend,
    PermissionPolicy,
    SafeFilesystemTools,
    ScriptedBackend,
    ToolExecutor,
    ToolRegistry,
)

router = APIRouter()
# Retained only as a compatibility attribute for older tests/integrations.
# It is never used as a cache or as request state.
_runtime: AgentRuntime | None = None


def _check_api_key(provided: str | None) -> None:
    if not env_bool("XRFM_AGENT_ENABLED", False):
        raise HTTPException(status_code=503, detail="Agent endpoint is disabled")
    expected = os.getenv("XRFM_AGENT_API_KEY")
    local_demo = env_bool("XRFM_AGENT_ALLOW_LOCAL_DEMO", False)
    if expected:
        if not provided or not hmac.compare_digest(provided, expected):
            raise HTTPException(status_code=401, detail="Invalid agent API key")
    elif not local_demo:
        raise HTTPException(status_code=503, detail="Agent authentication is not configured")


def _build_runtime() -> AgentRuntime:
    local_demo = env_bool("XRFM_AGENT_ALLOW_LOCAL_DEMO", False)
    root_value = os.getenv("XRFM_AGENT_ROOT")
    if not root_value and not local_demo:
        raise RuntimeError("XRFM_AGENT_ROOT is required outside local demo mode")
    root = Path(root_value or os.getcwd()).expanduser().resolve()
    registry = ToolRegistry()
    SafeFilesystemTools(root).register(registry)
    policy = PermissionPolicy(allowed_permissions={"filesystem.read"})
    executor = ToolExecutor(registry, policy=policy, audit=AuditLog())
    base_url = os.getenv("XRFM_AGENT_BASE_URL")
    model = os.getenv("XRFM_AGENT_MODEL", "xrfm-agent")
    backend: ModelBackend
    if base_url:
        backend = OpenAICompatibleBackend(
            base_url,
            model,
            api_key=os.getenv("XRFM_AGENT_MODEL_API_KEY"),
            native_tool_calling=env_bool("XRFM_AGENT_NATIVE_TOOL_CALLING", False),
            structured_output=env_bool("XRFM_AGENT_STRUCTURED_OUTPUT", False),
            reasoning=env_bool("XRFM_AGENT_REASONING", False),
        )
    elif local_demo:
        backend = ScriptedBackend()
    else:
        raise RuntimeError("scripted backend is only available with XRFM_AGENT_ALLOW_LOCAL_DEMO=true")
    return AgentRuntime(backend, executor)


def _get_runtime() -> AgentRuntime:
    """Build fresh backend/executor/audit state for every request."""
    return _build_runtime()


def _get_control(request: Request) -> AgentControl:
    control = get_state(request).agent_control
    if control is None:
        raise HTTPException(status_code=503, detail="Agent control plane is not ready")
    return control


def _configured_runtime(control: AgentControl) -> AgentRuntime:
    runtime = _get_runtime()
    control.apply_policy(runtime.executor.policy)
    return runtime


async def _watch_disconnect(request: Request, cancel_event: threading.Event) -> None:
    try:
        while not cancel_event.is_set():
            message = await request.receive()
            if message.get("type") == "http.disconnect":
                cancel_event.set()
                return
    except asyncio.CancelledError:
        raise
    except Exception:
        # A broken request channel must not leave execution running.
        cancel_event.set()


def _next_event(stream: Iterator[AgentEvent]) -> AgentEvent | None:
    try:
        return next(stream)
    except StopIteration:
        return None


async def _close_watcher(task: asyncio.Task[None]) -> None:
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


def _run(
    request: AgentRequest,
    cancel_event: threading.Event | None = None,
    control: AgentControl | None = None,
) -> AgentResponse:
    runtime = _configured_runtime(control) if control is not None else _get_runtime()
    state = runtime.run(request.request, cancel_event=cancel_event)
    return AgentResponse(
        request=state.request,
        answer=state.final_answer or "",
        failed=state.failed,
        steps=state.step,
        events=[event.to_dict() for event in state.events],
        observations=[observation.as_observation() for observation in state.observations],
    )


@router.post("/v1/agent/run", response_model=AgentResponse)
async def run_agent(
    request: AgentRequest,
    http_request: Request,
    x_xrfm_agent_key: str | None = Header(default=None),
) -> AgentResponse:
    _check_api_key(x_xrfm_agent_key)
    control = _get_control(http_request)
    cancel_event = threading.Event()
    try:
        registration = control.register(cancel_event)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    watcher = asyncio.create_task(_watch_disconnect(http_request, cancel_event))
    try:
        return await asyncio.to_thread(_run, request, cancel_event, control)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        cancel_event.set()
        control.unregister(registration)
        await _close_watcher(watcher)


@router.post("/v1/agent/stream")
async def stream_agent(
    request: AgentRequest,
    http_request: Request,
    x_xrfm_agent_key: str | None = Header(default=None),
) -> StreamingResponse:
    _check_api_key(x_xrfm_agent_key)
    control = _get_control(http_request)
    try:
        control.assert_active()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    async def events() -> AsyncIterator[str]:
        cancel_event = threading.Event()
        registration: int | None = None
        watcher: asyncio.Task[None] | None = None
        stream: Iterator[AgentEvent] | None = None
        try:
            registration = control.register(cancel_event)
            watcher = asyncio.create_task(_watch_disconnect(http_request, cancel_event))
            runtime = _configured_runtime(control)
            stream = runtime.stream(request.request, cancel_event=cancel_event)
            while not cancel_event.is_set():
                event = await asyncio.to_thread(_next_event, stream)
                if event is None:
                    yield "event: done\ndata: [DONE]\n\n"
                    return
                if cancel_event.is_set():
                    return
                yield f"event: {event.phase.value}\ndata: {json.dumps(event.to_dict(), ensure_ascii=False)}\n\n"
        except RuntimeError as exc:
            yield f"event: failed\ndata: {json.dumps({'error': str(exc)}, ensure_ascii=False)}\n\n"
        finally:
            cancel_event.set()
            if stream is not None:
                close = getattr(stream, "close", None)
                if close:
                    close()
            if registration is not None:
                control.unregister(registration)
            if watcher is not None:
                await _close_watcher(watcher)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/v1/agent/control/status")
def agent_control_status(
    request: Request,
    x_xrfm_agent_key: str | None = Header(default=None),
) -> dict[str, Any]:
    _check_api_key(x_xrfm_agent_key)
    return _get_control(request).snapshot()


@router.post("/v1/agent/control/kill")
def kill_agent(
    request: Request,
    x_xrfm_agent_key: str | None = Header(default=None),
) -> dict[str, Any]:
    _check_api_key(x_xrfm_agent_key)
    return _get_control(request).kill()


@router.post("/v1/agent/control/resume")
def resume_agent(
    request: Request,
    x_xrfm_agent_key: str | None = Header(default=None),
) -> dict[str, Any]:
    _check_api_key(x_xrfm_agent_key)
    return _get_control(request).resume()


@router.post("/v1/agent/control/revoke")
def revoke_agent(
    revocation: AgentRevokeRequest,
    request: Request,
    x_xrfm_agent_key: str | None = Header(default=None),
) -> dict[str, Any]:
    _check_api_key(x_xrfm_agent_key)
    try:
        return _get_control(request).revoke(permission=revocation.permission, tool=revocation.tool)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
