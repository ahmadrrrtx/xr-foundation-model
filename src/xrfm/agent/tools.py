"""Tool registry and safe built-in filesystem inspection tools."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .protocols import ToolFunction, ToolSpec


class ToolError(RuntimeError):
    pass


def _validate_schema(schema: dict[str, Any], value: Any, path: str = "arguments") -> None:
    """Small dependency-free JSON Schema subset used at the execution boundary."""
    typ = schema.get("type")
    if typ == "object":
        if not isinstance(value, dict):
            raise ToolError(f"{path} must be an object")
        required = schema.get("required", [])
        missing = [key for key in required if key not in value]
        if missing:
            raise ToolError(f"{path} missing required field(s): {', '.join(missing)}")
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)
        if additional is False:
            unknown = set(value) - set(properties)
            if unknown:
                raise ToolError(f"{path} has unknown field(s): {', '.join(sorted(unknown))}")
        for key, subschema in properties.items():
            if key in value:
                _validate_schema(subschema, value[key], f"{path}.{key}")
    elif typ == "string" and not isinstance(value, str):
        raise ToolError(f"{path} must be a string")
    elif typ == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
        raise ToolError(f"{path} must be an integer")
    elif typ == "number" and (not isinstance(value, (int, float)) or isinstance(value, bool)):
        raise ToolError(f"{path} must be a number")
    elif typ == "boolean" and not isinstance(value, bool):
        raise ToolError(f"{path} must be a boolean")
    if "enum" in schema and value not in schema["enum"]:
        raise ToolError(f"{path} must be one of {schema['enum']!r}")
    if isinstance(value, str) and "maxLength" in schema and len(value) > schema["maxLength"]:
        raise ToolError(f"{path} exceeds maxLength")
    if isinstance(value, (int, float)) and "maximum" in schema and value > schema["maximum"]:
        raise ToolError(f"{path} exceeds maximum")


class RegisteredTool:
    def __init__(self, spec: ToolSpec, function: ToolFunction) -> None:
        self.spec = spec
        self.function = function


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, spec: ToolSpec, function: ToolFunction) -> None:
        if spec.name in self._tools:
            raise ValueError(f"tool already registered: {spec.name}")
        self._tools[spec.name] = RegisteredTool(spec, function)

    def get(self, name: str) -> RegisteredTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolError(f"unavailable tool: {name}") from exc

    def specs(self) -> list[ToolSpec]:
        return [tool.spec for tool in self._tools.values()]

    def descriptions(self) -> list[dict[str, Any]]:
        return [spec.to_dict() for spec in self.specs()]


class SafeFilesystemTools:
    """Read-only tools confined to one root directory."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root).expanduser().resolve()

    def _path(self, relative: str) -> Path:
        candidate = (self.root / relative).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ToolError("path escapes the configured filesystem root")
        return candidate

    def register(self, registry: ToolRegistry) -> None:
        registry.register(
            ToolSpec("list_files", "List files under the approved project root.", {"type": "object", "properties": {"relative": {"type": "string"}, "max_items": {"type": "integer", "maximum": 500}}, "additionalProperties": False}, permissions=("filesystem.read",)),
            self.list_files,
        )
        registry.register(
            ToolSpec("largest_file", "Find the largest regular file under the approved project root.", {"type": "object", "properties": {"relative": {"type": "string"}}, "additionalProperties": False}, permissions=("filesystem.read",)),
            self.largest_file,
        )
        registry.register(
            ToolSpec("read_file", "Read a bounded UTF-8 text file under the approved project root.", {"type": "object", "properties": {"relative": {"type": "string"}, "max_chars": {"type": "integer", "maximum": 100000}}, "required": ["relative"], "additionalProperties": False}, permissions=("filesystem.read",)),
            self.read_file,
        )

    def list_files(self, args: dict[str, Any]) -> dict[str, Any]:
        base = self._path(args.get("relative", "."))
        if not base.is_dir():
            raise ToolError("relative path is not a directory")
        limit = int(args.get("max_items", 100))
        files = [str(path.relative_to(self.root)) for path in sorted(base.rglob("*")) if path.is_file()][:limit]
        return {"root": str(self.root), "files": files, "count": len(files)}

    def largest_file(self, args: dict[str, Any]) -> dict[str, Any]:
        base = self._path(args.get("relative", "."))
        candidates = [p for p in base.rglob("*") if p.is_file() and ".git" not in p.parts]
        if not candidates:
            raise ToolError("no regular files found")
        largest = max(candidates, key=lambda p: p.stat().st_size)
        return {"relative": str(largest.relative_to(self.root)), "bytes": largest.stat().st_size, "suffix": largest.suffix}

    def read_file(self, args: dict[str, Any]) -> dict[str, Any]:
        path = self._path(args["relative"])
        if not path.is_file():
            raise ToolError("not a regular file")
        max_chars = int(args.get("max_chars", 12000))
        text = path.read_text(encoding="utf-8", errors="replace")[:max_chars]
        return {"relative": str(path.relative_to(self.root)), "text": text, "truncated": path.stat().st_size > max_chars}
