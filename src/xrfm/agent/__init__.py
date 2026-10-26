"""XR Intelligence Runtime: model-agnostic, bounded, tool-using execution."""

from .backend_execution import BackendCancelledError, BackendExecutionError, BackendExecutor, BackendTimeoutError
from .backends import BackendInfo, ModelBackend, OpenAICompatibleBackend, ScriptedBackend
from .evaluation import AgentBenchmarkCase, AgentBenchmarkResult, reference_cases, run_benchmark
from .execution import ToolExecutor
from .protocols import (
    AgentEvent,
    AgentPhase,
    AgentState,
    Decision,
    DecisionType,
    Message,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from .runtime import AgentRuntime, RuntimeLimits
from .safety import AuditLog, PermissionPolicy, sanitize_observation
from .tools import SafeFilesystemTools, ToolError, ToolRegistry

__all__ = [
    "AgentBenchmarkCase",
    "AgentBenchmarkResult",
    "AgentEvent",
    "AgentPhase",
    "AgentRuntime",
    "AgentState",
    "AuditLog",
    "BackendInfo",
    "BackendCancelledError",
    "BackendExecutionError",
    "BackendExecutor",
    "BackendTimeoutError",
    "Decision",
    "DecisionType",
    "Message",
    "ModelBackend",
    "OpenAICompatibleBackend",
    "PermissionPolicy",
    "RuntimeLimits",
    "SafeFilesystemTools",
    "ScriptedBackend",
    "ToolCall",
    "ToolError",
    "ToolExecutor",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "reference_cases",
    "run_benchmark",
    "sanitize_observation",
]
