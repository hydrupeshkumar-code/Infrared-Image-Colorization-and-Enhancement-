# Convenience targets for the TIR SR + colorization project (BAH-2026 PS10).
# Backend runs at :8000, frontend dev server at :5173 (CORS is locked to that origin).

.PHONY: help install install-api data smoke train-sr train-colorize test \
        backend serve frontend demo evaluate download clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## Editable install of the Python package
	pip install -e .

install-api:  ## Install with the FastAPI extras
	pip install -e ".[api,dev]"

data:  ## Build the patch dataset (synthetic sample if data/raw is empty)
	tir-prepare-data --config configs/data.yaml

train-sr:  ## Train the super-resolution stage
	tir-train-sr --config configs/sr.yaml

train-colorize:  ## Train the colorization stage
	tir-train-colorize --config configs/colorize.yaml

smoke:  ## End-to-end smoke run on the synthetic sample (produces checkpoints)
	bash scripts/run_smoke.sh

evaluate:  ## Evaluate on the held-out test split (metrics + panels)
	tir-evaluate --config configs/infer.yaml

test:  ## Run the Python test suite (pipeline + API)
	pytest

backend:  ## Run the FastAPI backend (needs trained checkpoints)
	uvicorn app:app --reload

serve:  ## Bring up a working backend from scratch (auto-bootstraps checkpoints)
	bash scripts/serve_backend.sh

download:  ## Fetch Landsat-9 scenes (optional; needs USGS M2M credentials)
	python scripts/download_landsat.py

frontend:  ## Run the React dev server (separate terminal)
	cd frontend && npm install && npm run dev

demo: ## Reminder of the two commands needed for the full web demo
	@echo "Terminal 1:  make backend   # http://localhost:8000  (run 'make smoke' once first for checkpoints)"
	@echo "Terminal 2:  make frontend  # http://localhost:5173"

clean:  ## Remove generated artifacts (keeps checkpoints)
	rm -rf out data/interim/* data/processed/* data/raw/scene_* frontend/dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
