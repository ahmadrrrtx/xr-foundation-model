#!/usr/bin/env python3
"""Validate the generated static XRFM website using only the standard library."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent
PREFIX = "/xr-foundation-model/"
REQUIRED_META = ("description", "og:title", "og:description", "og:url", "twitter:card")


def main() -> int:
    errors: list[str] = []
    pages = sorted(p for p in ROOT.rglob("*.html") if p.name != "template.html" and "pages" not in p.parts)
    for page in pages:
        html = page.read_text(encoding="utf-8")
        rel = "/" + str(page.relative_to(ROOT)).replace("\\", "/")
        if "<title>" not in html or "{{" in html:
            errors.append(f"{rel}: unresolved template or missing title")
        for meta in REQUIRED_META:
            key = f'name="{meta}"' if meta == "description" else f'property="{meta}"'
            if meta == "twitter:card":
                key = 'name="twitter:card"'
            if key not in html:
                errors.append(f"{rel}: missing meta {meta}")
        for attribute, value in re.findall(r'(href|src)="([^"]+)"', html):
            if value.startswith(("http://", "https://", "mailto:", "#", "data:")):
                continue
            parsed = urlparse(value)
            path = unquote(parsed.path)
            if path.startswith(PREFIX):
                path = path[len(PREFIX) :]
            elif not path.startswith("/"):
                path = str(page.parent.relative_to(ROOT) / path)
            target = ROOT / path.lstrip("/")
            if path.endswith("/"):
                target = target / "index.html"
            if not target.exists():
                errors.append(f"{rel}: broken internal {attribute} {value}")
        for bad in ("TODO", "Lorem ipsum", "PLACEHOLDER CONTENT"):
            if bad in html:
                errors.append(f"{rel}: placeholder text '{bad}'")
    sitemap = (ROOT / "sitemap.xml").read_text(encoding="utf-8")
    for loc in re.findall(r"<loc>(.*?)</loc>", sitemap):
        path = urlparse(loc).path.replace(PREFIX, "", 1)
        target = ROOT / (path + "index.html" if path.endswith("/") else path)
        if not target.exists():
            errors.append(f"sitemap: missing {loc}")
    if not (ROOT / "404.html").exists():
        errors.append("missing 404.html")
    if errors:
        print("WEBSITE VALIDATION FAILED")
        for error in errors:
            print(" -", error)
        return 1
    print(f"WEBSITE VALIDATION PASSED: {len(pages)} HTML pages, all internal links resolved")
    return 0


if __name__ == "__main__":
    sys.exit(main())
