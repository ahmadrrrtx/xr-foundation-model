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

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import StreamingResponse

from api.agent_config import env_bool
from api.schemas import AgentRequest, AgentResponse
from xrfm.agent import (
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


def _run(request: AgentRequest, cancel_event: threading.Event | None = None) -> AgentResponse:
    state = _get_runtime().run(request.request, cancel_event=cancel_event)
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
    cancel_event = threading.Event()
    watcher = asyncio.create_task(_watch_disconnect(http_request, cancel_event))
    try:
        return await asyncio.to_thread(_run, request, cancel_event)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        cancel_event.set()
        await _close_watcher(watcher)


@router.post("/v1/agent/stream")
async def stream_agent(
    request: AgentRequest,
    http_request: Request,
    x_xrfm_agent_key: str | None = Header(default=None),
) -> StreamingResponse:
    _check_api_key(x_xrfm_agent_key)

    async def events() -> AsyncIterator[str]:
        cancel_event = threading.Event()
        watcher = asyncio.create_task(_watch_disconnect(http_request, cancel_event))
        stream: Iterator[AgentEvent] | None = None
        try:
            runtime = _get_runtime()
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
            await _close_watcher(watcher)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
