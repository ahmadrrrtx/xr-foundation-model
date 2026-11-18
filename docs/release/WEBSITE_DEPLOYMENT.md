# Official Website Deployment Report

**Target URL:** https://ahmadrrrtx.github.io/xr-foundation-model/
**Status:** Deployed from `main` and publicly verified after the visual redesign.
**Deployed main commit:** `06fa1099eddb31b64fbd4a6f38f378c2e11e32e2` (merge of PR #7)

## 1. Implementation

The old web UI and Vercel configuration were not restored. The site is a dependency-free static build under `website/`.

- `website/build.py` renders 12 HTML pages from `template.html` and `pages/*.html`.
- `website/assets/css/site.css` implements a dark, dimensional research-first design: gradient/grid field, sticky translucent header, soft layered shadows, 3D-hover cards, responsive navigation, reduced-motion and print fallbacks.
- `website/assets/js/site.js` is progressive enhancement only; the site works without JavaScript.
- `website/assets/img/` contains the original SVG mark, social-preview image and three original generated research artifacts: a layered neural transformer, a crystalline CoderSec shield and a multilingual tokenizer lattice. WebP derivatives are used for web delivery; the original PNGs are kept as local source artifacts.
- `website/validate.py` checks titles, metadata, internal links, local asset references, sitemap entries and the 404 page.
- `sitemap.xml`, `robots.txt` and `.nojekyll` are included.

Pages: home, models index, Native 25M, Native 85M, CoderSec, architecture, datasets, evaluation, documentation, research, releases and a 404 page.

## 2. Visual redesign

The redesign was researched against Mistral, Hugging Face and Qwen official sites. The resulting pattern is a confident hero, research-first navigation, dense model/dataset cards and original dimensional artifacts instead of copied external imagery.

The homepage now leads with the verifiable architecture stack and dual-track separation, then presents three artifact cards, a measured BPB table and reproducibility/engineering links. Models, CoderSec, Architecture, Datasets and Evaluation pages use track-appropriate artifacts while preserving all previously reviewed factual wording. No new capabilities, affiliations, metrics or publication claims were added.

## 3. Content accuracy checks

- Native models are described as randomly initialized; CoderSec as Qwen3-1.7B + QLoRA with base revision `70d244cc`.
- CoderSec is explicitly marked **not published**; no invented Hugging Face identifier is linked.
- Native shards: 8 train shards are confirmed present; the 623-row viewer row is explained as validation-only.
- CoderSec SFT: 24,100 actual rows, measured ChatML and placeholder defects, 11,993 duplicates.
- BPB is defined and explicitly denied as evidence of coding or security capability.
- Red-team sample sizes (8, 7, 4) are disclosed as preliminary.
- License information is artifact-specific; no blanket license assigned.
- No model weight, dataset, tokenizer, release tag or GitHub release was changed by the redesign.

## 4. Deployment workflow

`.github/workflows/pages.yml` has two jobs:

1. `build` — checkout, render, validate, configure Pages, upload artifact with hidden files.
2. `deploy` — depends on build, uses `actions/deploy-pages@v5` with `pages: write` and `id-token: write`.

Triggers: push to `main` or `release/*`, pull requests for validation only, and manual dispatch. Minimal permissions are declared. The `github-pages` environment policy allows deployment only from `main`, so the redesign was merged before publication.

## 5. Verification performed

Local, on `main` at `06fa109`:

- `python3 website/build.py` → rebuilt all 12 pages.
- `python3 website/validate.py` → `WEBSITE VALIDATION PASSED: 12 HTML pages, all internal links resolved`.
- `ruff check .`, `ruff format --check .`, `black --check .` → clean.

Remote:

- PR #7 CI run `44411289580`: success on the feature commit.
- PR #7 Pages run `44411289660`: build success.
- Post-merge CI run `44412448500`: success on `main`.
- Post-merge Pages run `44412448559`: build and deployment success.
- Public URL checks with cache-busting queries returned HTTP 200 for `/`, `/models/`, `/models/native-25m.html`, `/models/native-85m.html`, `/models/codersec.html`, `/architecture/`, `/datasets/`, `/evaluation/`, `/docs/`, `/research/`, `/releases/`, `/404.html`, `/sitemap.xml`, `/robots.txt`, `/assets/css/site.css` and all three WebP artifacts.
- The live homepage contains the new hero headline, the WebP artifact references and the updated CSS marker.

## 6. Acceptance checklist

- [x] Static site implemented with no heavy framework or paid service
- [x] All required pages present
- [x] SEO metadata, canonical URLs, Open Graph, Twitter cards, favicon, sitemap, robots, 404
- [x] Semantic HTML, skip link, keyboard-navigable mobile menu, responsive CSS
- [x] Original local 3D artifacts, no external hotlinks or copied visuals
- [x] Dark dimensional research-first design with smooth shadows and motion fallbacks
- [x] Factual XRFM claims, Track A/B separation, limitations and disclosures preserved
- [x] Internal links and local assets validated
- [x] Build workflow validates before deployment
- [x] GitHub Pages enabled manually by the repository owner
- [x] Deployment performed from `main`
- [x] Public URL returns HTTP 200 for all checked pages and assets
