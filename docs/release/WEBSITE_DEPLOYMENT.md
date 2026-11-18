# Official Website Deployment Report

**Target URL:** https://ahmadrrrtx.github.io/xr-foundation-model/
**Status:** Site source complete and validated; public deployment blocked by a one-time repository Pages setting.

## 1. Implementation

The old web UI and Vercel configuration were not restored. The new site is a dependency-free static site under `website/`:

- `website/build.py` renders 12 HTML pages from `template.html` and `pages/*.html`.
- `website/assets/css/site.css` provides responsive layout, accessible contrast and mobile navigation.
- `website/assets/js/site.js` is progressive enhancement only; the site works without JavaScript.
- `website/assets/img/` contains an original SVG mark and social-preview image.
- `website/validate.py` checks titles, metadata, internal links, sitemap entries and the 404 page.
- `sitemap.xml`, `robots.txt` and `.nojekyll` are included.

Pages: home, models index, Native 25M, Native 85M, CoderSec, architecture, datasets, evaluation, documentation, research, releases and a 404 page.

## 2. Content accuracy checks

- Native models are described as random-initialized; CoderSec as Qwen3-1.7B + QLoRA.
- CoderSec is explicitly marked **not published**; no invented Hugging Face identifier is linked.
- Native shards: 8 train shards are confirmed present; the 623-row viewer row is explained as validation-only.
- CoderSec SFT: 24,100 actual rows, measured ChatML and placeholder defects, 11,993 duplicates.
- BPB is defined and explicitly denied as evidence of coding or security capability.
- Red-team sample sizes (8, 7, 4) are disclosed as preliminary.
- License information is artifact-specific; no blanket license assigned.

## 3. Deployment workflow

`.github/workflows/pages.yml` now has two jobs:

1. `build` — checkout, render, validate, configure Pages, upload artifact with hidden files.
2. `deploy` — depends on build, uses `actions/deploy-pages@v5` with `pages: write` and `id-token: write`.

Triggers: push to `main` or `release/*`, pull requests for validation only, and manual dispatch. Minimal permissions are declared.

## 4. Verification performed

Local: `python website/build.py` followed by `python website/validate.py` → `WEBSITE VALIDATION PASSED: 12 HTML pages, all internal links resolved`.

Remote: Pages run #7 build job rendered and validated the site successfully, but `configure-pages` failed at site creation. Log evidence:

```
Get Pages site failed. Error: Not Found
Create Pages site failed. Error: Resource not accessible by integration
```

This is a repository setting/token permission issue, not a site defect. The fine-grained GitHub PAT also receives HTTP 403 from `PUT /repos/{owner}/{repo}/pages`.

## 5. Required manual step

In the GitHub repository:

1. Open **Settings → Pages**.
2. Set **Source** to **GitHub Actions**.
3. Save.

After that, re-run the latest GitHub Pages workflow (or push to the release branch). The build job will validate the site and the deploy job should publish it. The public URL must then be checked with an HTTP 200 request; until then, deployment is **not claimed as complete**.

## 6. Acceptance checklist

- [x] Static site implemented with no heavy framework or paid service
- [x] All required pages present
- [x] SEO metadata, canonical URLs, Open Graph, Twitter cards, favicon, sitemap, robots, 404
- [x] Semantic HTML, skip link, keyboard-navigable mobile menu, responsive CSS
- [x] No external runtime scripts or fonts
- [x] Internal links validated
- [x] Model/dataset links point to verified public repositories
- [x] Build workflow validates before deployment
- [ ] GitHub Pages enabled (manual)
- [ ] Public URL returns HTTP 200 (pending)
