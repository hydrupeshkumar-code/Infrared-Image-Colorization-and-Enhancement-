"""Lightweight logger + CSV metric writer (no heavy deps required)."""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Mapping


def get_logger(name: str = "tir", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


class CSVMetricLogger:
    """Append metric rows to a CSV file, writing the header on first use."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._header_written = self.path.exists() and self.path.stat().st_size > 0
        self._fieldnames: list[str] | None = None

    def log(self, row: Mapping[str, object]) -> None:
        row = dict(row)
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
        with open(self.path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self._fieldnames)
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow({k: row.get(k) for k in self._fieldnames})
