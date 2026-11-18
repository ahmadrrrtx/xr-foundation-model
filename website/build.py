#!/usr/bin/env python3
"""Render the static XRFM website from shared template and page fragments.

No external dependencies. Run from anywhere:
    python website/build.py
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
SITE = "https://ahmadrrrtx.github.io/xr-foundation-model/"

PAGES = {
    "index.html": (
        "Home",
        "Open multilingual foundation-model research with an original decoder-only architecture and a separately attributed defensive-security track.",
        "home",
        "home.html",
    ),
    "models/index.html": (
        "Models",
        "Published XRFM Native 25M and 85M checkpoints, plus the publication status of the CoderSec track.",
        "models",
        "models.html",
    ),
    "models/native-25m.html": (
        "XRFM Native 25M",
        "A 25.2M-parameter decoder-only transformer trained from random initialization on 20M-token multilingual shards.",
        "models",
        "native-25m.html",
    ),
    "models/native-85m.html": (
        "XRFM Native 85M",
        "An 86.4M-parameter decoder-only transformer scaling probe trained from random initialization.",
        "models",
        "native-85m.html",
    ),
    "models/codersec.html": (
        "XRFM CoderSec",
        "Track B lineage, QLoRA methodology and the current checkpoint publication status.",
        "models",
        "codersec.html",
    ),
    "architecture/index.html": (
        "Architecture",
        "Decoder-only transformer, RoPE, RMSNorm, SwiGLU, tokenizer and dual-track design.",
        "architecture",
        "architecture.html",
    ),
    "datasets/index.html": (
        "Datasets",
        "Native pretraining shards and the CoderSec SFT dataset: counts, schema, licenses and known limitations.",
        "datasets",
        "datasets.html",
    ),
    "evaluation/index.html": (
        "Evaluation",
        "Per-language BPB, scaling probes, CoderSec evaluation and explicit measurement limitations.",
        "evaluation",
        "evaluation.html",
    ),
    "docs/index.html": (
        "Documentation",
        "Installation, quick start, tokenizer, training, inference, datasets, evaluation and reproduction guidance.",
        "docs",
        "docs.html",
    ),
    "research/index.html": (
        "Research",
        "Research reports, tokenizer tournament, experiment registry and technical decisions.",
        "research",
        "research.html",
    ),
    "releases/index.html": (
        "Releases",
        "Version history, release notes and artifact revisions for XRFM.",
        "releases",
        "releases.html",
    ),
    "404.html": ("Page not found", "The requested XRFM page could not be found.", "home", "notfound.html"),
}


def render() -> list[str]:
    template = (ROOT / "template.html").read_text(encoding="utf-8")
    built: list[str] = []
    for output, (title, description, nav_key, fragment) in PAGES.items():
        depth = output.count("/")
        site_root = "./" if depth == 0 else "../" * depth
        body = (ROOT / "pages" / fragment).read_text(encoding="utf-8").strip()
        # Absolute project links in page fragments are rewritten to a relative
        # site root; full external URLs (including og:image) must stay intact.
        body = body.replace('="/xr-foundation-model/', f'="{site_root}')
        body = body.replace("{{ROOT}}", site_root)
        html = template
        html = html.replace("{{ROOT}}", site_root)
        html = html.replace("{{TITLE}}", title)
        html = html.replace("{{DESCRIPTION}}", description)
        html = html.replace("{{BODY}}", body)
        canonical = SITE + output
        html = html.replace("{{CANONICAL}}", canonical)
        html = html.replace(f'data-nav="{nav_key}"', f'data-nav="{nav_key}" aria-current="page"')
        target = ROOT / output
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(html + "\n", encoding="utf-8")
        built.append(str(target.relative_to(ROOT)))
    return built


if __name__ == "__main__":
    for name in render():
        print("built", name)
