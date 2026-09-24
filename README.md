# Mijail Littin: academic website

Source of **[mijaillittinj.github.io](https://mijaillittinj.github.io/)**: research, publications, teaching, interactive learning modules and software on computational methods for inverse problems and scientific machine learning.

A static site (Astro 7 + TypeScript), typeset like a LaTeX article in Computer Modern. The interactive figures (forward and inverse problems, physics-informed neural networks, teaching demos) are small dependency-free TypeScript modules that run in the browser.

## Quick start

```bash
npm ci
npm run dev            # http://localhost:4321
npm run build          # static site in dist/
npm run preview        # serve dist/
npx astro check        # type check
for f in tests/*.test.ts; do node "$f"; done   # numerical tests
```

Requires Node ≥ 22.

## Structure

```
src/
  data/          publications, teaching, learn, software, site (single source of truth)
  i18n/          UI strings (en/es), translated routes, URL helpers
  layouts/       Base, LearnModule
  components/
    apps/        tabs of the home-page figure (one forward/inverse problem per field)
    teaching/    interactive teaching demos
    viz/         learning-module labs and shared figure components
    slides/      HTML presentation components
    pages/       bilingual page bodies
    ui/          header, footer, bibliography entries
  lib/           numerics (linear algebra, SAT, PINNs, optimisers, plotting, …)
  pages/         routes (English at /, Spanish under /es/)
public/          exported data, media, CV (PDF)
tests/           numerical tests, run in CI
scripts/         sync-publications.mjs (weekly update from ORCID and Crossref)
```

## Deployment

A push to `main` runs `.github/workflows/deploy.yml`: tests, type check, publication sync, build and deployment to GitHub Pages. The workflow also runs weekly so that new publications appear automatically.

- Repository variable `SITE_URL`: the site address (e.g. `https://mijaillittinj.github.io`).
- Repository variable `BASE_PATH`: only for a project repository (e.g. `/website/`).

## Licence

- **Code:** MIT.
- **Content:** CC BY 4.0, unless stated otherwise. See [LICENSE](LICENSE).
