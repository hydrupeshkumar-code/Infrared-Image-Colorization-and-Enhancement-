# ChaturVyuha — frontend

The **ChaturVyuha standalone site** — the landing UI for the TIR
super-resolution + colorization demo (BAH-2026 PS10). It consumes the FastAPI
backend documented in [`../docs/api.md`](../docs/api.md).

It's a single self-contained HTML bundle (`public/chaturvyuha-site/ChaturVyuha
(standalone).html`): a cinematic hero + before/after slider + a working upload
that drives the real backend. Vite is used only as a thin static server — `/`
redirects to that bundle.

## Run

```bash
npm install
npm run dev        # http://localhost:5173
```

The backend must be running at `API_BASE` (default `http://localhost:8000`) and
its CORS is locked to `http://localhost:5173` — use **`localhost`**, not
`127.0.0.1`, or the browser will block the responses.

> The bundle inlines its own assets but still loads React + fonts from a CDN at
> runtime, so it needs internet to render.

```bash
npm run build      # vite production build (emits index.html + copies public/)
npm run preview    # serve the built dist/
```

## Structure

| File | Responsibility |
|------|----------------|
| `index.html` | site root — redirects `/` to the standalone landing page |
| `vite.config.ts` | thin static-server config, dev server on :5173 |
| `public/chaturvyuha-site/ChaturVyuha (standalone).html` | the landing page — a self-contained bundle; the appended `TIR-BACKEND-WIRING` script connects its upload → `/infer` → poll `/jobs` → input/SR/RGB/residual previews + Kelvin metrics + GeoTIFF downloads, with a `/health` preflight |
| `public/chaturvyuha-site/support.js` | DC runtime kept alongside the bundle (generated; do not edit) |
| `public/favicon.svg` | favicon |

## Changing the backend URL

Edit the `API_BASE` variable in the `TIR-BACKEND-WIRING` script near the end of
`public/chaturvyuha-site/ChaturVyuha (standalone).html`:

```js
var API_BASE = 'http://localhost:8000';
```

All artifact URLs returned by the backend are relative (`/results/…`) and are
resolved against `API_BASE` (via the script's `asset()` helper), so swapping this
one value repoints the whole site.
