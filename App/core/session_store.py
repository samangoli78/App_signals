"""JSON persistence for the delta annotation list."""

from __future__ import annotations

import json
from typing import Any

import numpy as np


class SessionStore:
    @staticmethod
    def _convert(obj: Any):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    def save_delta(self, path: str, delta: list[Any]) -> None:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(delta, file, default=self._convert)

    def load_delta(self, path: str) -> list[Any]:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
