#!/usr/bin/env bash
# End-to-end smoke test on the synthetic sample (no Landsat download needed).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[1/5] prepare data"
tir-prepare-data   --config configs/data.yaml
echo "[2/5] train SR (smoke)"
tir-train-sr       --config configs/sr.yaml       --max-steps 20
echo "[3/5] train colorize (smoke)"
tir-train-colorize --config configs/colorize.yaml --max-steps 20
echo "[4/5] infer"
tir-infer --input data/interim/scene_00/lr_tir_200m.tif --out-dir out/scene_00
echo "[5/5] evaluate"
tir-evaluate --config configs/infer.yaml
echo "smoke test complete -> out/eval/metrics.json"
