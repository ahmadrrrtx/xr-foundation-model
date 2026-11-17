"""
Data source registry and strict SPDX license governance for XRFM.

Implements the dual-track governance controls mandated by
``docs/research/XRFM_RESEARCH_REVIEW.md`` (§4–6):

- **Tier A** (``tier_a``): Commercial-permissive SPDX licenses allowed for weight
  training on both Track A (``XRFM Native``) and Track B (``XRFM CoderSec``).
- **Tier B** (``tier_b``): Permissive-with-attribution SPDX licenses allowed for
  weight training with automated ``NOTICE.md`` attribution tracking.
- **RAG-Only** (``rag_only``): ShareAlike / copyleft reference corpora (e.g.,
  OWASP Cheat Sheets ``CC-BY-SA-4.0``) permitted ONLY in an isolated RAG vector
  index and strictly blocked from pretraining/SFT/DPO weight shards.
- **Eval-Only** (``eval_only``): Non-commercial or copyleft evaluation suites
  (e.g., ``Belebele`` ``CC-BY-NC-4.0``, ``MultiPL-E`` ``BSD-3 + GPL``) strictly
  blocked from training shards and registered in the contamination gate.
- **Hold** (``hold``): Quarantined or unverified items (e.g., ``unknown``/``varies``
  licenses or ``Qwen2.5-Coder-3B`` research-only license) blocked from training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import yaml

TIER_A_SPDX_ALLOWLIST: frozenset[str] = frozenset(
    {
        "Apache-2.0",
        "MIT",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "CC0-1.0",
        "ISC",
        "PSF-2.0",
        "Unlicense",
        "0BSD",
    }
)

TIER_B_SPDX_ALLOWLIST: frozenset[str] = frozenset(
    {
        "ODC-By-1.0",
        "CC-BY-4.0",
        "CDLA-Permissive-2.0",
    }
)

TRAINABLE_SPDX_ALLOWLIST: frozenset[str] = TIER_A_SPDX_ALLOWLIST | TIER_B_SPDX_ALLOWLIST

VALID_LICENSE_TIERS: frozenset[str] = frozenset({"tier_a", "tier_b", "rag_only", "eval_only", "hold"})

VALID_ALLOWED_USES: frozenset[str] = frozenset(
    {
        "train_track_a",
        "train_track_b",
        "rag_retrieval",
        "evaluation",
        "hold",
    }
)

FORBIDDEN_TRAINING_SUBSTRINGS: tuple[str, ...] = (
    "unknown",
    "varies",
    "gpl",
    "agpl",
    "lgpl",
    "cc-by-nc",
    "cc-by-nd",
    "cc-by-sa",
    "research-only",
    "qwen-research",
)


class LicenseComplianceError(ValueError):
    """Raised when a data source violates SPDX or track governance rules."""


@dataclass
class SourceConfig:
    """Configuration for a single governed data source."""

    name: str
    type: str  # web | code | math | reasoning | multilingual | security | instruction | eval | custom
    uri: str  # local path, HuggingFace repo, or URL
    license: str = "unknown"
    license_url: str = ""
    license_tier: str = "tier_a"  # tier_a | tier_b | rag_only | eval_only | hold
    allowed_uses: list[str] = field(default_factory=lambda: ["train_track_a", "train_track_b"])
    version: str = "v1"
    revision: str = ""  # Pinned commit SHA / immutable release tag for remote sources
    language: str = "en"  # primary language or "multi"
    languages: list[str] = field(default_factory=list)
    enabled: bool = True
    weight: float = 1.0  # mixture weight (normalized later)
    checksum: str = ""  # expected SHA256 if known
    format: str = "jsonl"  # jsonl | txt | parquet | hf
    text_field: str = "text"  # field containing text in jsonl/parquet
    attribution: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Source name is required")
        if self.weight < 0:
            raise ValueError(f"Source weight must be >= 0, got {self.weight}")
        if self.license_tier not in VALID_LICENSE_TIERS:
            raise LicenseComplianceError(
                f"Source '{self.name}' has invalid license_tier='{self.license_tier}'. "
                f"Must be one of {sorted(VALID_LICENSE_TIERS)}."
            )
        unknown_uses = [u for u in self.allowed_uses if u not in VALID_ALLOWED_USES]
        if unknown_uses:
            raise LicenseComplianceError(
                f"Source '{self.name}' has invalid allowed_uses={unknown_uses}. "
                f"Valid uses: {sorted(VALID_ALLOWED_USES)}."
            )

    def validate_for_weight_training(self, track: str = "track_a", require_pinned_revision: bool = False) -> None:
        """Verify that this source is legally and architecturally cleared for weight training."""
        use_tag = f"train_{track}" if not track.startswith("train_") else track
        if use_tag not in ("train_track_a", "train_track_b"):
            raise ValueError(f"Unknown training track '{track}'. Expected 'track_a' or 'track_b'.")

        lic_norm = (self.license or "").strip()
        lic_lower = lic_norm.lower()

        if self.license_tier not in ("tier_a", "tier_b"):
            raise LicenseComplianceError(
                f"Source '{self.name}' has license_tier='{self.license_tier}' (license='{lic_norm}') "
                f"and is STRICTLY BLOCKED from weight training on {track}."
            )

        if use_tag not in self.allowed_uses:
            raise LicenseComplianceError(
                f"Source '{self.name}' allowed_uses={self.allowed_uses} does not permit '{use_tag}'."
            )

        if not lic_norm or any(bad in lic_lower for bad in FORBIDDEN_TRAINING_SUBSTRINGS):
            raise LicenseComplianceError(
                f"Source '{self.name}' has forbidden/unverified license='{lic_norm}' for weight training."
            )

        if lic_norm not in TRAINABLE_SPDX_ALLOWLIST:
            raise LicenseComplianceError(
                f"Source '{self.name}' license='{lic_norm}' is not in TRAINABLE_SPDX_ALLOWLIST "
                f"({sorted(TRAINABLE_SPDX_ALLOWLIST)})."
            )

        if self.license_tier == "tier_a" and lic_norm not in TIER_A_SPDX_ALLOWLIST:
            raise LicenseComplianceError(
                f"Source '{self.name}' claims tier_a but license '{lic_norm}' belongs outside Tier A."
            )

        if require_pinned_revision and self.uri.startswith(("hf://", "https://huggingface.co/")):
            if not self.revision or self.revision.lower() in ("main", "master", "latest", "unknown"):
                raise LicenseComplianceError(
                    f"Remote Hugging Face source '{self.name}' ({self.uri}) must pin an immutable "
                    f"commit SHA or release revision, got revision='{self.revision}'."
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "uri": self.uri,
            "license": self.license,
            "license_url": self.license_url,
            "license_tier": self.license_tier,
            "allowed_uses": list(self.allowed_uses),
            "version": self.version,
            "revision": self.revision,
            "language": self.language,
            "languages": list(self.languages),
            "enabled": self.enabled,
            "weight": self.weight,
            "checksum": self.checksum,
            "format": self.format,
            "text_field": self.text_field,
            "attribution": self.attribution,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SourceConfig:
        known = {
            "name",
            "type",
            "uri",
            "license",
            "license_url",
            "license_tier",
            "allowed_uses",
            "version",
            "revision",
            "language",
            "languages",
            "enabled",
            "weight",
            "checksum",
            "format",
            "text_field",
            "attribution",
            "metadata",
        }
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)


@dataclass
class SourceRegistry:
    """Registry of governed data sources with SPDX and track isolation enforcement."""

    sources: dict[str, SourceConfig] = field(default_factory=dict)
    registry_version: str = "2.0"

    def register(self, config: SourceConfig) -> None:
        self.sources[config.name] = config

    def get(self, name: str) -> SourceConfig | None:
        return self.sources.get(name)

    def list_enabled(self) -> list[SourceConfig]:
        return [s for s in self.sources.values() if s.enabled]

    def list_by_type(self, source_type: str) -> list[SourceConfig]:
        return [s for s in self.sources.values() if s.type == source_type and s.enabled]

    def list_by_use(self, allowed_use: str) -> list[SourceConfig]:
        return [s for s in self.sources.values() if s.enabled and allowed_use in s.allowed_uses]

    def assert_source_trainable(
        self,
        name: str,
        track: str = "track_a",
        require_pinned_revision: bool = False,
    ) -> SourceConfig:
        src = self.sources.get(name)
        if src is None:
            raise LicenseComplianceError(f"Unregistered data source '{name}' cannot be used for training.")
        src.validate_for_weight_training(track=track, require_pinned_revision=require_pinned_revision)
        return src

    def list_trainable_sources(
        self,
        track: str = "track_a",
        require_pinned_revision: bool = False,
    ) -> list[SourceConfig]:
        """Return enabled sources explicitly cleared for weight training on ``track``."""
        use_tag = f"train_{track}" if not track.startswith("train_") else track
        trainable: list[SourceConfig] = []
        for src in self.sources.values():
            if not src.enabled:
                continue
            if src.license_tier in ("rag_only", "eval_only") and use_tag not in src.allowed_uses:
                continue
            if src.license_tier == "hold" and use_tag not in src.allowed_uses:
                raise LicenseComplianceError(
                    f"Enabled source '{src.name}' is in license_tier='hold' (license='{src.license}') "
                    f"and must either be disabled or cleared before training validation."
                )
            if use_tag in src.allowed_uses or src.license_tier in ("tier_a", "tier_b"):
                if use_tag in src.allowed_uses:
                    src.validate_for_weight_training(
                        track=track,
                        require_pinned_revision=require_pinned_revision,
                    )
                    trainable.append(src)
        return trainable

    def validate_for_training(
        self,
        track: str = "track_a",
        require_pinned_revision: bool = True,
    ) -> list[SourceConfig]:
        """Audit all enabled sources and return verified trainable sources for ``track``."""
        return self.list_trainable_sources(track=track, require_pinned_revision=require_pinned_revision)

    def normalized_weights(self, track: str | None = None) -> dict[str, float]:
        """Return normalized mixture weights for enabled (or track-trainable) sources."""
        if track is not None:
            enabled = self.list_trainable_sources(track=track, require_pinned_revision=False)
        else:
            enabled = self.list_enabled()
        total = sum(s.weight for s in enabled)
        if total <= 0:
            return {s.name: 1.0 / max(len(enabled), 1) for s in enabled}
        return {s.name: s.weight / total for s in enabled}

    def to_dict(self) -> dict[str, Any]:
        return {
            "registry_version": self.registry_version,
            "sources": {k: v.to_dict() for k, v in self.sources.items()},
        }

    def save_yaml(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, allow_unicode=True)

    @classmethod
    def from_yaml(cls, path: str) -> SourceRegistry:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        reg = cls(registry_version=str(data.get("registry_version", "2.0")))
        for name, cfg in data.get("sources", {}).items():
            if "name" not in cfg:
                cfg["name"] = name
            reg.register(SourceConfig.from_dict(cfg))
        return reg

    @classmethod
    def default_local_registry(cls) -> SourceRegistry:
        """Create default registry pointing to local test/golden datasets."""
        reg = cls()
        reg.register(
            SourceConfig(
                name="golden_test",
                type="web",
                uri="tests/data/golden/documents.jsonl",
                license="MIT",
                license_tier="tier_a",
                allowed_uses=["train_track_a", "train_track_b"],
                version="v1",
                revision="local-v1",
                language="en",
                enabled=True,
                weight=1.0,
                format="jsonl",
            )
        )
        return reg
