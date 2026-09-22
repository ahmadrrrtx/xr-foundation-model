"""XR Intelligence Runtime API endpoints."""
from __future__ import annotations

import asyncio
import hmac
import json
import os
from pathlib import Path
from typing import Any, Iterator

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import StreamingResponse

from api.schemas import AgentEventResponse, AgentRequest, AgentResponse
from xrfm.agent import (
    AgentRuntime,
    OpenAICompatibleBackend,
    PermissionPolicy,
    SafeFilesystemTools,
    ScriptedBackend,
    ToolExecutor,
    ToolRegistry,
)

router = APIRouter()
_runtime: AgentRuntime | None = None


def _check_api_key(provided: str | None) -> None:
    expected = os.getenv("XRFM_AGENT_API_KEY")
    if expected and not provided or expected and not hmac.compare_digest(provided or "", expected):
        raise HTTPException(status_code=401, detail="Invalid agent API key")


def _build_runtime() -> AgentRuntime:
    root = Path(os.getenv("XRFM_AGENT_ROOT", os.getcwd())).expanduser().resolve()
    registry = ToolRegistry()
    SafeFilesystemTools(root).register(registry)
    policy = PermissionPolicy(allowed_permissions={"filesystem.read"})
    executor = ToolExecutor(registry, policy=policy)
    base_url = os.getenv("XRFM_AGENT_BASE_URL")
    model = os.getenv("XRFM_AGENT_MODEL", "xrfm-agent")
    if base_url:
        backend = OpenAICompatibleBackend(base_url, model, api_key=os.getenv("XRFM_AGENT_MODEL_API_KEY"))
    else:
        backend = ScriptedBackend()
    return AgentRuntime(backend, executor)


def _get_runtime() -> AgentRuntime:
    global _runtime
    if _runtime is None:
        _runtime = _build_runtime()
    return _runtime


def _run(request: AgentRequest) -> AgentResponse:
    state = _get_runtime().run(request.request)
    return AgentResponse(
        request=state.request,
        answer=state.final_answer or "",
        failed=state.failed,
        steps=state.step,
        events=[event.to_dict() for event in state.events],
        observations=[observation.as_observation() for observation in state.observations],
    )


@router.post("/v1/agent/run", response_model=AgentResponse)
async def run_agent(request: AgentRequest, x_xrfm_agent_key: str | None = Header(default=None)) -> AgentResponse:
    """Run a bounded agent task with structured tool events in the response."""
    _check_api_key(x_xrfm_agent_key)
    return await asyncio.to_thread(_run, request)


@router.post("/v1/agent/stream")
async def stream_agent(
    request: AgentRequest,
    http_request: Request,
    x_xrfm_agent_key: str | None = Header(default=None),
) -> StreamingResponse:
    """Stream typed XR events as Server-Sent Events."""
    _check_api_key(x_xrfm_agent_key)

    async def events() -> Iterator[str]:
        runtime = _get_runtime()
        for event in runtime.stream(request.request):
            if await http_request.is_disconnected():
                break
            yield f"event: {event.phase.value}\ndata: {json.dumps(event.to_dict(), ensure_ascii=False)}\n\n"
            await asyncio.sleep(0)
        yield "event: done\ndata: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
