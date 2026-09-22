# Deploy FieldLens dashboard

Pilot disclaimer: FieldLens is a research pilot. Results are trends from small runs on a data subset, not benchmark numbers.

The dashboard lives in `dashboard/` and builds to a **static export** (`out/`). Production domain placeholder: **DOMAIN_TBD**.

## Prerequisites

- Node.js 20+ locally for `npm install` and `npm run build`
- GitHub repo connected to Vercel (or another static host)
- Exported data under `dashboard/public/data/` from Phase 8 training export

## Vercel project setup

1. Import the FieldLens GitHub repository in Vercel.
2. Set **Root Directory** to `dashboard`.
3. Framework preset: **Next.js** (Vercel detects App Router).
4. Build command: `npm run build` (default).
5. Output: Next.js static export writes to `out/`. Vercel serves static export output automatically when `output: "export"` is set in `next.config.ts`.
6. Install command: `npm install`.
7. Node version: 20.x in Project Settings if the default is older.

No serverless functions are required. Environment variables are optional unless you add analytics later.

## Static export notes

- `next.config.ts` sets `output: "export"` and `images.unoptimized: true` for static hosting.
- Client pages load `/data/results.json` and `/data/gallery/index.json` at runtime from the same origin.
- After each training export, copy `dashboard/public/data/` onto the host or build machine (that folder is gitignored because gallery images are large). Then rebuild and redeploy.
- Screenshots: with `npm run dev` running, install Playwright and run `node scripts/screenshot.mjs http://localhost:3000`. PNGs land in `docs/screenshots/`.

Local verification:

```bash
cd dashboard
npm install
npm run build
npx serve out
```

## Custom domain (DOMAIN_TBD)

When the production hostname is chosen:

1. In Vercel Project Settings, open **Domains** and add `DOMAIN_TBD` (replace with the real FQDN).
2. Vercel shows DNS records. Typical setup:
   - **Apex** (`example.com`): A record to Vercel anycast IP, or CNAME flattening per registrar docs.
   - **WWW** (`www.example.com`): CNAME to `cname.vercel-dns.com`.
3. Enable automatic HTTPS (default on Vercel).
4. Set the primary domain in Vercel and add a redirect from the alternate host if both apex and www are used.
5. Update footer or README links only after the real domain is live (keep `DOMAIN_TBD` in docs until then).

## GitHub link

Footer GitHub URL placeholder: `https://github.com/PLACEHOLDER/FieldLens`. Replace `PLACEHOLDER` with the org or user name before public launch.

## ONNX model (optional Phase 10)

If `public/models/fieldlens.onnx` is committed or uploaded as a static asset, the `/try` route loads it in the browser. Large models may affect deploy size and first-load time; consider hosting the ONNX on the same origin under `public/models/` for same-origin fetch rules.

## Checklist before go-live

- [ ] `results.json` filled or still honestly `pending_runs` with nulls
- [ ] Gallery index and assets comply with Agriculture-Vision redistribution terms
- [ ] GitHub and domain placeholders updated
- [ ] `npm run build` succeeds in CI or locally
- [ ] Lighthouse spot check on mobile for contrast and tap targets
