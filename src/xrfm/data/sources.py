"""
Source registry for XRFM.

Machine-readable source registry, authoritative description of where training data comes from.

Location: configs/data/sources.yaml (or architecture-consistent location)

Each source describes:
name, uri, type, license, license_url, language, description, expected_size, enabled, transform
Where relevant, track download checksum, version, snapshot date, dataset revision, citation, redistribution restrictions.

Must NOT hard-code source URLs throughout Python files.
"""

from __future__ import annotations

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from pathlib import Path


@dataclass
class SourceConfig:
    name: str
    uri: str
    type: str  # text, jsonl, parquet, hf_dataset, etc.
    license: str
    license_url: str = ""
    language: str = "en"
    description: str = ""
    expected_size: str = ""  # e.g. "1GB", "10M docs"
    enabled: bool = True
    transform: str = ""  # optional transform name
    # Provenance
    version: str = "v1"
    checksum: str = ""
    snapshot_date: str = ""
    dataset_revision: str = ""
    citation: str = ""
    redistribution: str = ""  # e.g. "allowed", "restricted", "recipe-only"
    # Extra
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SourceConfig:
        allowed = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in data.items() if k in allowed}
        # Handle extra
        extra = {k: v for k, v in data.items() if k not in allowed}
        if extra:
            filtered["extra"] = extra
        return cls(**filtered)

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict

        d = asdict(self)
        # Merge extra
        extra = d.pop("extra", {})
        d.update(extra)
        return d


class SourceRegistry:
    """Registry of all data sources."""

    def __init__(self, sources: List[SourceConfig] | None = None):
        self.sources: Dict[str, SourceConfig] = {}
        if sources:
            for s in sources:
                self.sources[s.name] = s

    @classmethod
    def from_yaml(cls, path: str) -> SourceRegistry:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        sources = []
        if isinstance(data, dict):
            # Expect top-level "sources" list or dict
            if "sources" in data:
                raw = data["sources"]
                if isinstance(raw, list):
                    for item in raw:
                        sources.append(SourceConfig.from_dict(item))
                elif isinstance(raw, dict):
                    for name, item in raw.items():
                        if isinstance(item, dict):
                            item = dict(item)
                            item["name"] = name
                            sources.append(SourceConfig.from_dict(item))
            else:
                # Direct dict of name -> config
                for name, item in data.items():
                    if isinstance(item, dict):
                        item = dict(item)
                        item["name"] = name
                        sources.append(SourceConfig.from_dict(item))
        elif isinstance(data, list):
            for item in data:
                sources.append(SourceConfig.from_dict(item))

        return cls(sources)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SourceRegistry:
        sources = []
        for name, cfg in data.items():
            if isinstance(cfg, dict):
                cfg = dict(cfg)
                cfg["name"] = name
                sources.append(SourceConfig.from_dict(cfg))
        return cls(sources)

    def get(self, name: str) -> Optional[SourceConfig]:
        return self.sources.get(name)

    def list_enabled(self) -> List[SourceConfig]:
        return [s for s in self.sources.values() if s.enabled]

    def list_all(self) -> List[SourceConfig]:
        return list(self.sources.values())

    def to_yaml(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = {"sources": [s.to_dict() for s in self.sources.values()]}
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)

    def to_dict(self) -> Dict[str, Any]:
        return {name: src.to_dict() for name, src in self.sources.items()}


def load_registry(path: str | None = None) -> SourceRegistry:
    """Load registry from default location if path not provided."""
    if path is None:
        # Try multiple locations
        candidates = [
            "configs/data/sources.yaml",
            "config/data/sources.yaml",
            "src/xrfm/config/sources.yaml",
        ]
        for cand in candidates:
            if os.path.isfile(cand):
                path = cand
                break
        if path is None:
            # Return empty registry
            return SourceRegistry()

    return SourceRegistry.from_yaml(path)
