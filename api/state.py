"""Per-application API lifecycle state (never module-global request state)."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException, Request

if TYPE_CHECKING:
    from xrfm.agent.control import AgentControl


@dataclass
class APIState:
    model: Any = None
    engine: Any = None
    tokenizer: Any = None
    model_loaded: bool = False
    startup_error: str | None = None
    checkpoint_path: str | None = None
    model_id: str = "xrfm-default"
    started_at: float = 0.0
    generation_lock: threading.Lock = field(default_factory=threading.Lock)
    agent_control: AgentControl | None = None


def get_state(request: Request) -> APIState:
    state = getattr(request.app.state, "xrfm", None)
    if state is None:
        state = APIState()
        request.app.state.xrfm = state
    return state


def require_ready(request: Request) -> APIState:
    state = get_state(request)
    if not state.model_loaded or state.model is None or state.engine is None or state.tokenizer is None:
        raise HTTPException(status_code=503, detail=state.startup_error or "Model/tokenizer/engine are not ready")
    return state
