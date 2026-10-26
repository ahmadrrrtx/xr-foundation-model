"""Tool registry, Draft 2020-12 validation, and bounded filesystem tools."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema

from .protocols import ToolFunction, ToolSpec


class ToolError(RuntimeError):
    pass


@dataclass(frozen=True)
class JSONLimits:
    max_bytes: int = 65_536
    max_items: int = 1_000
    max_depth: int = 32
    max_string_bytes: int = 16_384

    def __post_init__(self) -> None:
        bounds = (
            ("max_bytes", 1, 1_000_000),
            ("max_items", 1, 10_000),
            ("max_depth", 0, 64),
            ("max_string_bytes", 1, 1_000_000),
        )
        for name, minimum, maximum in bounds:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")


_SCHEMA_URI = "https://json-schema.org/draft/2020-12/schema"
_REGISTRY_LIMITS = JSONLimits(max_bytes=1_000_000, max_items=10_000, max_depth=32, max_string_bytes=100_000)


def enforce_json_bounds(value: Any, limits: JSONLimits = JSONLimits(), path: str = "value") -> None:
    """Reject oversized/deep/non-JSON values before schema or model use."""
    try:
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise ToolError(f"{path} must be finite JSON data") from exc
    if len(encoded) > limits.max_bytes:
        raise ToolError(f"{path} exceeds total JSON byte bound ({limits.max_bytes})")

    count = 0
    pending: list[tuple[Any, int, str]] = [(value, 0, path)]
    while pending:
        current, depth, current_path = pending.pop()
        if depth > limits.max_depth:
            raise ToolError(f"{current_path} exceeds maximum nesting depth ({limits.max_depth})")
        if isinstance(current, dict):
            count += len(current)
            for key, item in current.items():
                if not isinstance(key, str):
                    raise ToolError(f"{current_path} has a non-string object key")
                pending.append((item, depth + 1, f"{current_path}.{key}"))
        elif isinstance(current, (list, tuple)):
            count += len(current)
            for index, item in enumerate(current):
                pending.append((item, depth + 1, f"{current_path}[{index}]"))
        elif isinstance(current, str):
            if len(current.encode("utf-8")) > limits.max_string_bytes:
                raise ToolError(f"{current_path} exceeds string byte bound ({limits.max_string_bytes})")
        if count > limits.max_items:
            raise ToolError(f"{path} exceeds total item bound ({limits.max_items})")


def _check_schema_definition(schema: dict[str, Any], path: str) -> None:
    if not isinstance(schema, dict):
        raise ToolError(f"{path} must be a JSON Schema object")
    enforce_json_bounds(schema, path=path)
    try:
        jsonschema.Draft202012Validator.check_schema(schema)
    except jsonschema.SchemaError as exc:
        raise ToolError(f"{path} is not a valid Draft 2020-12 schema: {exc.message}") from exc


def _validate_schema(
    schema: dict[str, Any],
    value: Any,
    path: str = "arguments",
    limits: JSONLimits = JSONLimits(),
) -> None:
    """Validate bounded JSON data with the complete Draft 2020-12 vocabulary."""
    _check_schema_definition(schema, path=f"{path} schema")
    enforce_json_bounds(value, limits=limits, path=path)
    validator = jsonschema.Draft202012Validator(schema)
    try:
        validator.validate(value)
    except jsonschema.ValidationError as exc:
        location = ".".join(str(part) for part in exc.absolute_path)
        suffix = f".{location}" if location else ""
        raise ToolError(f"{path}{suffix}: {exc.message}") from exc


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
        _check_schema_definition(spec.input_schema, f"tool {spec.name} input_schema")
        if spec.output_schema is not None:
            _check_schema_definition(spec.output_schema, f"tool {spec.name} output_schema")
        timeout = spec.timeout_seconds
        if (
            not isinstance(timeout, (int, float))
            or isinstance(timeout, bool)
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("tool timeout_seconds must be a finite positive number")
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

    def schema_manifest(self) -> dict[str, Any]:
        """Return a bounded manifest of schema/spec metadata, sorted by tool name."""
        tools = sorted(self.descriptions(), key=lambda item: item["name"])
        payload: dict[str, Any] = {"schema_draft": "2020-12", "schema_uri": _SCHEMA_URI, "tools": tools}
        enforce_json_bounds(payload, limits=_REGISTRY_LIMITS, path="tool registry")
        canonical = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        digest = hashlib.sha256(canonical).hexdigest()
        return {**payload, "version": f"sha256:{digest}"}

    @property
    def schema_version(self) -> str:
        """Content-address schema/spec metadata; callable code is not hashed."""
        return str(self.schema_manifest()["version"])


class SafeFilesystemTools:
    """Read-only, bounded tools confined to one root and denied secret files."""

    MAX_SCAN_FILES = 10_000
    MAX_SCAN_DIRECTORIES = 1_000
    MAX_READ_CHARS = 100_000
    _SENSITIVE_NAMES = {
        ".env",
        ".netrc",
        ".git-credentials",
        "id_rsa",
        "id_ed25519",
        "credentials",
        "credentials.json",
        "secrets",
        "secrets.json",
    }
    _SENSITIVE_DIRS = {".git", ".ssh", ".aws", ".config/gcloud"}

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root).expanduser().resolve()

    @classmethod
    def _sensitive(cls, path: Path) -> bool:
        lowered = [part.lower() for part in path.parts]
        if any(part in {".git", ".ssh", ".aws"} for part in lowered):
            return True
        name = lowered[-1] if lowered else ""
        return name in cls._SENSITIVE_NAMES or name.endswith((".pem", ".key", ".p12", ".pfx"))

    def _path(self, relative: str) -> Path:
        if not isinstance(relative, str) or "\x00" in relative:
            raise ToolError("relative path must be a valid string")
        candidate = (self.root / relative).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ToolError("path escapes the configured filesystem root")
        relative_path = candidate.relative_to(self.root)
        if self._sensitive(relative_path):
            raise ToolError("access to sensitive files or credential directories is forbidden")
        return candidate

    @staticmethod
    def _limit(value: Any, name: str, default: int, maximum: int, minimum: int = 0) -> int:
        if value is None:
            return default
        if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
            raise ToolError(f"{name} must be an integer in [{minimum}, {maximum}]")
        return value

    def _walk_files(self, base: Path, max_files: int, max_directories: int) -> Iterator[Path]:
        """Stream a bounded depth-first walk; never materialize a whole tree."""
        pending = [base]
        directories = 1
        files = 0
        if directories > max_directories:
            raise ToolError(f"directory scan exceeds bound ({max_directories})")
        while pending:
            directory = pending.pop()
            try:
                entries = os.scandir(directory)
            except OSError as exc:
                raise ToolError(f"cannot scan directory: {type(exc).__name__}") from exc
            with entries:
                for entry in entries:
                    path = Path(entry.path)
                    relative = path.relative_to(self.root)
                    if self._sensitive(relative):
                        continue
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            directories += 1
                            if directories > max_directories:
                                raise ToolError(f"directory scan exceeds bound ({max_directories})")
                            pending.append(path)
                        elif entry.is_file(follow_symlinks=False):
                            files += 1
                            if files > max_files:
                                raise ToolError(f"file scan exceeds bound ({max_files})")
                            yield path
                    except OSError as exc:
                        raise ToolError(f"cannot inspect filesystem entry: {type(exc).__name__}") from exc

    def register(self, registry: ToolRegistry) -> None:
        registry.register(
            ToolSpec(
                "list_files",
                "List files under the approved project root.",
                {
                    "type": "object",
                    "properties": {
                        "relative": {"type": "string", "maxLength": 4096},
                        "max_items": {"type": "integer", "minimum": 0, "maximum": 500},
                    },
                    "additionalProperties": False,
                },
                permissions=("filesystem.read",),
            ),
            self.list_files,
        )
        registry.register(
            ToolSpec(
                "largest_file",
                "Find the largest regular file under the approved project root.",
                {
                    "type": "object",
                    "properties": {"relative": {"type": "string", "maxLength": 4096}},
                    "additionalProperties": False,
                },
                permissions=("filesystem.read",),
            ),
            self.largest_file,
        )
        registry.register(
            ToolSpec(
                "read_file",
                "Read a bounded UTF-8 text file under the approved project root.",
                {
                    "type": "object",
                    "properties": {
                        "relative": {"type": "string", "maxLength": 4096},
                        "max_chars": {"type": "integer", "minimum": 0, "maximum": self.MAX_READ_CHARS},
                    },
                    "required": ["relative"],
                    "additionalProperties": False,
                },
                permissions=("filesystem.read",),
            ),
            self.read_file,
        )

    def list_files(self, args: dict[str, Any]) -> dict[str, Any]:
        base = self._path(args.get("relative", "."))
        if not base.is_dir():
            raise ToolError("relative path is not a directory")
        limit = self._limit(args.get("max_items"), "max_items", 100, 500)
        files: list[str] = []
        truncated = False
        iterator = self._walk_files(base, self.MAX_SCAN_FILES, self.MAX_SCAN_DIRECTORIES)
        for path in iterator:
            if len(files) >= limit:
                truncated = True
                break
            files.append(str(path.relative_to(self.root)))
        return {"root": str(self.root), "files": sorted(files), "count": len(files), "truncated": truncated}

    def largest_file(self, args: dict[str, Any]) -> dict[str, Any]:
        base = self._path(args.get("relative", "."))
        if not base.is_dir():
            raise ToolError("relative path is not a directory")
        largest: Path | None = None
        largest_size = -1
        for path in self._walk_files(base, self.MAX_SCAN_FILES, self.MAX_SCAN_DIRECTORIES):
            size = path.stat().st_size
            if size > largest_size:
                largest, largest_size = path, size
        if largest is None:
            raise ToolError("no regular files found")
        return {"relative": str(largest.relative_to(self.root)), "bytes": largest_size, "suffix": largest.suffix}

    def read_file(self, args: dict[str, Any]) -> dict[str, Any]:
        path = self._path(args["relative"])
        if not path.is_file():
            raise ToolError("not a regular file")
        max_chars = self._limit(args.get("max_chars"), "max_chars", 12_000, self.MAX_READ_CHARS)
        # UTF-8 uses at most four bytes per scalar; the small overshoot lets us
        # detect truncation without ever reading the whole file.
        max_bytes = max_chars * 4 + 1
        with path.open("rb") as handle:
            raw = handle.read(max_bytes)
        decoded = raw.decode("utf-8", errors="replace")
        text = decoded[:max_chars]
        truncated = path.stat().st_size > len(raw) or len(decoded) > max_chars
        return {"relative": str(path.relative_to(self.root)), "text": text, "truncated": truncated}
