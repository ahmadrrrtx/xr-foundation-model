"""
Data mixing for XRFM.

Explicit, versioned, reproducible mixture experiments.

Config conceptually:
mixture:
  web: 0.45
  books: 0.15
  code: 0.15
  ...

For every generated corpus, record source, weight, sampled token count, actual token count.

References OLMo's public documentation treating mixture as explicit experiment.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

from xrfm.data.schema import Document


@dataclass
class MixtureComponent:
    source: str
    weight: float
    # Optional overrides
    max_docs: Optional[int] = None
    max_tokens: Optional[int] = None


@dataclass
class MixtureConfig:
    components: List[MixtureComponent]
    seed: int = 42
    # How to sample: token-weighted or doc-weighted
    sampling_strategy: str = "token"  # token | doc | proportional

    def __post_init__(self):
        total = sum(c.weight for c in self.components)
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Mixture weights must sum to ~1.0, got {total}")
        for c in self.components:
            if c.weight < 0 or c.weight > 1:
                raise ValueError(f"Weight must be in [0,1], got {c.weight} for {c.source}")

    @classmethod
    def from_dict(cls, data: Dict[str, float], seed: int = 42) -> MixtureConfig:
        comps = [MixtureComponent(source=k, weight=v) for k, v in data.items()]
        return cls(components=comps, seed=seed)

    def to_dict(self) -> Dict[str, float]:
        return {c.source: c.weight for c in self.components}


@dataclass
class MixtureStats:
    source: str
    weight: float
    sampled_docs: int
    actual_docs: int
    sampled_tokens: int
    actual_tokens: int


def mix_documents(
    docs_by_source: Dict[str, List[Document]],
    config: MixtureConfig,
    tokenizer=None,
    target_tokens: Optional[int] = None,
    target_docs: Optional[int] = None,
) -> Tuple[List[Document], List[MixtureStats]]:
    """
    Mix documents according to weights.

    If target_tokens specified, sample to approximate token count per weight.
    If target_docs specified, sample doc count per weight.
    Otherwise, returns all docs weighted proportionally? For Phase 1, we implement simple proportional sampling.

    Token counting requires tokenizer; if not provided, uses word count approximation.

    Deterministic via seed.
    """
    rng = random.Random(config.seed)

    # Precompute token counts if needed
    token_counts_by_source: Dict[str, List[int]] = {}
    total_tokens_by_source: Dict[str, int] = {}

    for source, docs in docs_by_source.items():
        if tokenizer:
            counts = [len(tokenizer.encode(d.text)) for d in docs]
        else:
            counts = [len(d.text.split()) for d in docs]  # approximate
        token_counts_by_source[source] = counts
        total_tokens_by_source[source] = sum(counts)

    mixed: List[Document] = []
    stats: List[MixtureStats] = []

    if target_tokens:
        # Sample tokens per source according to weight
        for comp in config.components:
            source = comp.source
            docs = docs_by_source.get(source, [])
            if not docs:
                stats.append(
                    MixtureStats(
                        source=source,
                        weight=comp.weight,
                        sampled_docs=0,
                        actual_docs=0,
                        sampled_tokens=0,
                        actual_tokens=0,
                    )
                )
                continue

            desired_tokens = int(target_tokens * comp.weight)
            # Sample docs until we reach desired tokens (shuffled deterministically)
            indices = list(range(len(docs)))
            rng.shuffle(indices)

            sampled_docs = []
            sampled_tokens = 0
            for idx in indices:
                if sampled_tokens >= desired_tokens:
                    break
                if comp.max_docs and len(sampled_docs) >= comp.max_docs:
                    break
                if comp.max_tokens and sampled_tokens >= comp.max_tokens:
                    break
                sampled_docs.append(docs[idx])
                sampled_tokens += token_counts_by_source[source][idx]

            mixed.extend(sampled_docs)
            actual_tokens = sum(token_counts_by_source[source][docs.index(d)] for d in sampled_docs) if sampled_docs else 0
            # Actually compute from counts
            actual_tokens = sampled_tokens

            stats.append(
                MixtureStats(
                    source=source,
                    weight=comp.weight,
                    sampled_docs=len(sampled_docs),
                    actual_docs=len(sampled_docs),
                    sampled_tokens=desired_tokens,
                    actual_tokens=actual_tokens,
                )
            )

        # Shuffle final mixed corpus deterministically
        rng.shuffle(mixed)

    elif target_docs:
        for comp in config.components:
            source = comp.source
            docs = docs_by_source.get(source, [])
            if not docs:
                stats.append(
                    MixtureStats(
                        source=source,
                        weight=comp.weight,
                        sampled_docs=0,
                        actual_docs=0,
                        sampled_tokens=0,
                        actual_tokens=0,
                    )
                )
                continue

            desired_docs = int(target_docs * comp.weight)
            indices = list(range(len(docs)))
            rng.shuffle(indices)
            selected = [docs[i] for i in indices[:desired_docs]]

            mixed.extend(selected)

            sampled_tokens = sum(token_counts_by_source[source][i] for i in indices[:desired_docs])
            stats.append(
                MixtureStats(
                    source=source,
                    weight=comp.weight,
                    sampled_docs=desired_docs,
                    actual_docs=len(selected),
                    sampled_tokens=sampled_tokens,
                    actual_tokens=sampled_tokens,
                )
            )

        rng.shuffle(mixed)

    else:
        # No target, just concatenate all weighted? Or return all docs but with stats?
        # For simplicity, return all docs from requested sources, shuffled, with weight stats
        for comp in config.components:
            source = comp.source
            docs = docs_by_source.get(source, [])
            mixed.extend(docs)
            total_tok = total_tokens_by_source.get(source, 0)
            stats.append(
                MixtureStats(
                    source=source,
                    weight=comp.weight,
                    sampled_docs=len(docs),
                    actual_docs=len(docs),
                    sampled_tokens=total_tok,
                    actual_tokens=total_tok,
                )
            )
        rng.shuffle(mixed)

    return mixed, stats
