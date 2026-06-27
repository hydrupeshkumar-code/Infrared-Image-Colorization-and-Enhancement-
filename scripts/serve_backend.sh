#!/usr/bin/env bash
# Bring up a *working* backend from a fresh clone: if no checkpoints exist yet,
# run the synthetic smoke pipeline to produce them, then start uvicorn.
set -euo pipefail
cd "$(dirname "$0")/.."

SR_CKPT=checkpoints/sr/best.pth
COL_CKPT=checkpoints/colorize/best.pth

if [[ ! -f "$SR_CKPT" || ! -f "$COL_CKPT" ]]; then
  echo "[serve] no checkpoints found — bootstrapping via scripts/run_smoke.sh"
  bash scripts/run_smoke.sh
fi

echo "[serve] starting API on http://localhost:8000  (docs at /docs)"
exec uvicorn app:app --host 0.0.0.0 --port 8000
