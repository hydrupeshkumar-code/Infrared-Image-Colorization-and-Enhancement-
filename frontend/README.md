# ChaturVyuha — frontend

Dark-themed React 18 + TypeScript + Vite + Tailwind UI for the TIR
super-resolution + colorization demo (BAH-2026 PS10). It consumes the FastAPI
backend documented in [`../docs/api.md`](../docs/api.md).

## Run

```bash
npm install
npm run dev        # http://localhost:5173
```

The backend must be running at the URL in `src/api.ts` (`API_BASE`, default
`http://localhost:8000`) and its CORS is locked to `http://localhost:5173` — use
**`localhost`**, not `127.0.0.1`, or the browser will block the responses.

```bash
npm run build      # tsc --noEmit type-check + vite production build
npm run preview     # serve the built dist/
```

## Structure

| File | Responsibility |
|------|----------------|
| `src/api.ts` | `API_BASE` constant, response types, `postInfer` / `getJob`, `asset()` URL helper |
| `src/App.tsx` | job lifecycle: upload → poll `/jobs` every 1.5 s → queued/running/done/failed; React state only, timers cleared on unmount |
| `src/components/Hero.tsx` | hero section, headings, blurbs, upload button, status/error banners |
| `src/components/BeforeAfterSlider.tsx` | draggable before/after slider (mouse **and** touch) |
| `src/components/Nav.tsx` | top nav (logo, center pill, upload, hamburger) |
| `src/components/ResultsDashboard.tsx` | 4-panel audit grid (residual emphasized) + metrics cards (Kelvin emphasized, `null → "—"`) + GeoTIFF downloads |
| `src/placeholders.ts` | inline SVG placeholder previews so the hero looks alive before a job runs |

## Changing the backend URL

Edit the single constant in `src/api.ts`:

```ts
export const API_BASE = "http://localhost:8000";
```

All artifact URLs returned by the backend are relative (`/results/…`) and are
resolved against `API_BASE` via `asset()`, so swapping this one value
repoints the whole app.

## Notes

- No persistence: state lives in React only (no `localStorage` / `sessionStorage`).
- Fonts (Inter + Playfair Display) load from Google Fonts via `src/index.css`;
  they fall back to system fonts if offline.
- Respects `prefers-reduced-motion` (hero animations disabled).
