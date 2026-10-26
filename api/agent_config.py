"""Fail-closed agent deployment configuration shared by lifespan and routes."""

from __future__ import annotations

import os


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def validate_agent_configuration() -> None:
    """Reject public agent configurations that would silently become a demo."""
    if not env_bool("XRFM_AGENT_ENABLED", False):
        return
    local_demo = env_bool("XRFM_AGENT_ALLOW_LOCAL_DEMO", False)
    if not local_demo and not os.getenv("XRFM_AGENT_API_KEY"):
        raise RuntimeError("XRFM agent is enabled publicly but XRFM_AGENT_API_KEY is missing")
    if not local_demo and not os.getenv("XRFM_AGENT_BASE_URL"):
        raise RuntimeError(
            "XRFM agent requires XRFM_AGENT_BASE_URL; scripted demo mode needs XRFM_AGENT_ALLOW_LOCAL_DEMO=true"
        )
    if not local_demo and not os.getenv("XRFM_AGENT_ROOT"):
        raise RuntimeError("XRFM agent requires an explicit XRFM_AGENT_ROOT outside local demo mode")
    if not local_demo and not (
        env_bool("XRFM_AGENT_NATIVE_TOOL_CALLING", False) or env_bool("XRFM_AGENT_STRUCTURED_OUTPUT", False)
    ):
        raise RuntimeError(
            "public agent provider must explicitly declare XRFM_AGENT_NATIVE_TOOL_CALLING or "
            "XRFM_AGENT_STRUCTURED_OUTPUT"
        )
