"""Entry point so the API runs with: uvicorn app:app --reload

The implementation lives in the ``tir.api`` package; this thin module just
exposes the FastAPI instance at the import path uvicorn expects.
"""
from tir.api.server import app

__all__ = ["app"]
