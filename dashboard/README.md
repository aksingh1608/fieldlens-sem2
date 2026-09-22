# FieldLens dashboard

Pilot disclaimer: FieldLens is a research pilot. Results are trends from small runs on a data subset, not benchmark numbers.

Next.js App Router site (TypeScript, Tailwind CSS) configured for **static export**. All runtime data comes from `public/data` (and optional `public/models/fieldlens.onnx` on `/try`).

## Stack

- Next.js 15 App Router, `output: "export"` in `next.config.ts`
- Tailwind CSS (light/dark via class on `html`)
- Recharts for results charts
- lucide-react icons (theme toggle only)
- onnxruntime-web on `/try` when the ONNX file exists

## Local development

```bash
cd dashboard
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Data contract

| Path | Purpose |
|------|---------|
| `public/data/results.json` | Metrics, curves, robustness, efficiency (`status: pending_runs` until export) |
| `public/data/gallery/index.json` | Gallery tile list with RGB, NIR, GT, and prediction asset URLs |
| `public/models/fieldlens.onnx` | Optional in-browser model (Phase 10) |

The UI reads these files with `fetch` from the browser. It does not call a backend API and does not invent metrics when fields are null.

Type definitions: `lib/types/results.ts`, `lib/types/gallery.ts`.

## Production build

```bash
npm run build
```

Static files are written to `out/`. Serve `out/` with any static host (Vercel, S3, nginx).

## Pages

| Route | Role |
|-------|------|
| `/` | Overview, research questions, runs, dataset note |
| `/explorer` | Tile gallery, RGB/NIR, overlays, class toggles, tile alert |
| `/compare` | GT plus three runs, per-tile IoU from `results.json` |
| `/results` | Tables and charts from `results.json` |
| `/method` | Architecture SVG, split placeholder, metrics, citations |
| `/terms` | Terms and dataset redistribution note |
| `/privacy` | No personal data unless analytics added later |
| `/try` | onnxruntime-web demo when model file is present |

## Deploy

See [../docs/deploy.md](../docs/deploy.md) for Vercel project setup, static export settings, and custom domain DNS for `DOMAIN_TBD`.

## Accessibility

- Semantic headings and table captions
- Keyboard-focusable controls, visible focus rings (accent color)
- Theme toggle with text label
- Images use descriptive `alt` text when assets exist
