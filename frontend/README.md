# ChaturVyuha — frontend

UI for the TIR super-resolution + colorization demo (BAH-2026 PS10). It consumes
the FastAPI backend documented in [`../docs/api.md`](../docs/api.md).

There are **two frontends**, both served by the same Vite dev server:

| Route | What | Stack |
|-------|------|-------|
| **`/`** (landing) | The **ChaturVyuha standalone site** — cinematic hero + before/after slider + a working upload that drives the real backend. | A single self-contained HTML bundle in `public/chaturvyuha-site/`, enhanced by an appended script that wires the upload to the API. |
| **`/app/`** | The React upload/results demo. | React 18 + TypeScript + Tailwind (`src/`). |

`/` redirects to `public/chaturvyuha-site/ChaturVyuha (standalone).html` (see
`index.html`); the React demo entry is `app/index.html`.

## Run

```bash
npm install
npm run dev        # http://localhost:5173  (landing = standalone site; /app/ = React demo)
```

The backend must be running at `API_BASE` (default `http://localhost:8000`) and
its CORS is locked to `http://localhost:5173` — use **`localhost`**, not
`127.0.0.1`, or the browser will block the responses. `API_BASE` is set in two
places that must agree: `src/api.ts` (React demo) and the `API_BASE` variable in
the `TIR-BACKEND-WIRING` script at the bottom of
`public/chaturvyuha-site/ChaturVyuha (standalone).html` (landing).

> The standalone bundle inlines its own assets but still loads React + fonts from
> a CDN at runtime, so it needs internet to render.

```bash
npm run build      # tsc --noEmit type-check + vite production build (both entries)
npm run preview     # serve the built dist/
```

## Structure

| File | Responsibility |
|------|----------------|
| `index.html` | site root — redirects `/` to the standalone landing page |
| `app/index.html` | Vite entry for the React demo (served at `/app/`) |
| `vite.config.ts` | two-entry build (`/` landing + `/app/` React demo), dev server on :5173 |
| `public/chaturvyuha-site/ChaturVyuha (standalone).html` | the landing page — a self-contained bundle; the appended `TIR-BACKEND-WIRING` script connects its upload → `/infer` → poll `/jobs` → previews + Kelvin metrics + GeoTIFF downloads, and a `/health` preflight |
| `public/chaturvyuha-site/support.js` | DC runtime kept alongside the bundle (generated; do not edit) |
| `public/frontend-index.html` | a standalone copy of the redirect splash (also points at the landing) |
| `src/api.ts` | `API_BASE` constant, response types, `postInfer` / `getJob` / `getHealth`, `asset()` URL helper |
| `src/App.tsx` | job lifecycle: probe `/health` on mount, upload → poll `/jobs` every 1.5 s → queued/running/done/failed; React state only, timers cleared on unmount |
| `src/components/Hero.tsx` | hero section, headings, blurbs, upload button, status/error banners, "checkpoints not ready" warning |
| `src/components/BeforeAfterSlider.tsx` | draggable before/after slider (mouse **and** touch) |
| `src/components/Nav.tsx` | top nav (logo, center pill, upload, hamburger) |
| `src/components/ResultsDashboard.tsx` | 4-panel audit grid (residual emphasized) + metrics cards (Kelvin emphasized, `null → "—"`) + GeoTIFF downloads |
| `src/placeholders.ts` | inline SVG placeholder previews so the hero looks alive before a job runs |

## Changing the backend URL

The backend URL lives in two places (one per frontend) — keep them in sync:

- React demo — the constant in `src/api.ts`:
  ```ts
  export const API_BASE = "http://localhost:8000";
  ```
- Standalone landing — the `API_BASE` variable in the `TIR-BACKEND-WIRING`
  script near the end of `public/chaturvyuha-site/ChaturVyuha (standalone).html`:
  ```js
  var API_BASE = 'http://localhost:8000';
  ```

All artifact URLs returned by the backend are relative (`/results/…`) and are
resolved against `API_BASE` (via `asset()` in both frontends), so swapping this
value repoints each frontend.

## Notes

- No persistence: state lives in React only (no `localStorage` / `sessionStorage`).
- Fonts (Inter + Playfair Display) load from Google Fonts via `src/index.css`;
  they fall back to system fonts if offline.
- Respects `prefers-reduced-motion` (hero animations disabled).
