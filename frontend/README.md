# ChaturVyuha — frontend

UI for the TIR super-resolution + colorization demo (BAH-2026 PS10). It consumes
the FastAPI backend documented in [`../docs/api.md`](../docs/api.md).

There are **two frontends**, both served by the same Vite dev server:

| Route | What | Stack |
|-------|------|-------|
| **`/`** (landing) | The 3D ChaturVyuha site — three.js starfield, animated hero, and a working upload that drives the real backend. | Static HTML in `public/chaturvyuha-site/` + the self-bootstrapping `support.js` runtime (pulls React/three from a CDN at runtime). |
| **`/app/`** | The original React upload/results demo. | React 18 + TypeScript + Tailwind (`src/`). |

`/` redirects to `public/chaturvyuha-site/ChaturVyuha 3D.dc.html` (see
`index.html`); the React demo entry is `app/index.html`.

## Run

```bash
npm install
npm run dev        # http://localhost:5173  (landing = 3D site; /app/ = React demo)
```

The backend must be running at `API_BASE` (default `http://localhost:8000`) and
its CORS is locked to `http://localhost:5173` — use **`localhost`**, not
`127.0.0.1`, or the browser will block the responses. `API_BASE` is set in two
places that must agree: `src/api.ts` (React demo) and the `API_BASE` field of the
`Component` class inside `public/chaturvyuha-site/ChaturVyuha 3D.dc.html` (3D
landing).

> The 3D site loads React/three/Google-Fonts from CDNs at runtime, so it needs
> internet. `ChaturVyuha (standalone).html` in the same folder is a fully
> self-contained (offline) snapshot if you ever need one.

```bash
npm run build      # tsc --noEmit type-check + vite production build (both entries)
npm run preview     # serve the built dist/
```

## Structure

| File | Responsibility |
|------|----------------|
| `index.html` | site root — redirects `/` to the 3D landing page |
| `app/index.html` | Vite entry for the React demo (served at `/app/`) |
| `vite.config.ts` | two-entry build (`/` landing + `/app/` React demo), dev server on :5173 |
| `public/chaturvyuha-site/ChaturVyuha 3D.dc.html` | the 3D landing page; its embedded `Component` (DCLogic) class holds `API_BASE` and the wired upload → `/infer` → poll `/jobs` → render previews + Kelvin metrics + GeoTIFF downloads |
| `public/chaturvyuha-site/support.js` | the DC runtime that bootstraps the `<x-dc>` page (loads React from a CDN, mounts the component) — generated; do not edit |
| `public/chaturvyuha-site/ChaturVyuha.dc.html` | non-3D variant of the landing (kept) |
| `public/chaturvyuha-site/ChaturVyuha (standalone).html` | fully self-contained offline snapshot (kept) |
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
- 3D landing — the `API_BASE` field of the `Component` class in
  `public/chaturvyuha-site/ChaturVyuha 3D.dc.html`:
  ```js
  API_BASE = 'http://localhost:8000';
  ```

All artifact URLs returned by the backend are relative (`/results/…`) and are
resolved against `API_BASE` (via `asset()` in React, `_asset()` in the 3D page),
so swapping this value repoints each frontend.

## Notes

- No persistence: state lives in React only (no `localStorage` / `sessionStorage`).
- Fonts (Inter + Playfair Display) load from Google Fonts via `src/index.css`;
  they fall back to system fonts if offline.
- Respects `prefers-reduced-motion` (hero animations disabled).
