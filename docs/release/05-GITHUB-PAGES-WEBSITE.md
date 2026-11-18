# Phase Five — GitHub Pages Website

**Status:** Assets created. Deployment via GitHub Actions pending CI green.

## Website URL

https://ahmadrrrtx.github.io/xr-foundation-model/

## Files Created

| File | Purpose |
|---|---|
| `docs/index.html` | Homepage — hero, models, results, quick start, footer |
| `docs/404.html` | 404 page with navigation back to home |
| `docs/sitemap.xml` | XML sitemap for search engines |
| `docs/robots.txt` | Robots.txt with sitemap reference |
| `docs/favicon.svg` | Site favicon (inline SVG) |
| `.github/workflows/pages.yml` | GitHub Pages deployment workflow |

## Website Features

- **Static HTML/CSS/JS** — no heavy frameworks
- **SEO** — meta description, Open Graph tags, Twitter card, canonical URL, robots.txt, sitemap.xml
- **Accessibility** — skip-to-content link, semantic HTML, ARIA labels on table, lang attribute
- **Responsive** — mobile-friendly grid layout
- **Favicon** — inline SVG favicon
- **Dual-track models** — Track A (Native 25M/85M) and Track B (CoderSec) clearly separated
- **Evaluation results** — red-teaming table with PASS status
- **Quick start** — copy-paste commands for installation and usage
- **Model card links** — Native and CoderSec model cards linked

## Deployment

Configured via `.github/workflows/pages.yml`:
- Triggers on push to `main` and `release/*`
- Deploys from `./docs` directory
- Minimal permissions: `contents: read`, `pages: write`, `id-token: write`
- Environment: `github-pages`

**Note:** GitHub Pages must be enabled in repository settings (Source: GitHub Actions). Deployment will occur after CI is green and the branch is merged to main.

## Verification Checklist

- [x] Static HTML (no React/Vue/etc.)
- [x] SEO meta tags (title, description, og:*, twitter:*)
- [x] robots.txt with sitemap reference
- [x] sitemap.xml with all pages
- [x] favicon.svg
- [x] 404.html
- [ ] GitHub Pages enabled in repo settings
- [ ] Public URL verified (https://ahmadrrrtx.github.io/xr-foundation-model/)
