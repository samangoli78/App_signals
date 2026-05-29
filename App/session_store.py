from __future__ import annotations
# Future annotations defer hint evaluation to runtime.

# Import order note:
# stdlib (`json`, `typing`) before third-party (`numpy`).
import json
from typing import Any

import numpy as np
from .base_components import BaseSessionStore


class SessionStore(BaseSessionStore):
    @staticmethod
    def _convert(obj: Any):
        # Serialization adapter:
        # JSON cannot store NumPy objects directly, so we convert to Python-native types.
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    def save_delta(self, path: str, delta: list[Any]) -> None:
        # Type hints:
        # `path: str` expects file path text, `delta: list[Any]` allows mixed nested data.
        # `-> None` documents that this method returns nothing.
        # Persistence concept:
        # write runtime state to disk so work can be reloaded later.
        with open(path, "w", encoding="utf-8") as file:
            json.dump(delta, file, default=self._convert)

    def load_delta(self, path: str) -> list[Any]:
        # File context manager (`with open(...)`) guarantees the file is closed safely.
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)


class SyncingSessionStore(SessionStore):
    """Child implementation reserved for future remote sync behavior."""

    def after_save(self, path: str, delta: list[Any]) -> None:
        return None

    def after_load(self, path: str, delta: list[Any]) -> None:
        return None

    def save_delta(self, path: str, delta: list[Any]) -> None:
        super().save_delta(path=path, delta=delta)
        self.after_save(path=path, delta=delta)

    def load_delta(self, path: str) -> list[Any]:
        delta = super().load_delta(path=path)
        self.after_load(path=path, delta=delta)
        return delta
