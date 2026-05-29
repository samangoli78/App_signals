from __future__ import annotations
# Future annotations are helpful when types may reference not-yet-loaded names.

# Import order note:
# stdlib typing/dataclasses first, then third-party dataframe types.
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class MapSection:
    # Dataclass concept:
    # concise model class that auto-generates init/repr/equality for structured data.
    """Container for one map point metadata and associated signal dataframe."""

    points_df: pd.DataFrame
    file_name: str
    signals_df: pd.DataFrame
    # Typed fields document expected shapes and improve IDE/static-checker help.

    def __getitem__(self, index: int) -> Any:
        # Adapter concept:
        # exposes tuple-style indexing so old code still works while using a named model.
        if index == 0:
            return self.points_df
        if index == 1:
            return self.file_name
        if index == 2:
            return self.signals_df
        raise IndexError(index)

    def __iter__(self):
        # Generator behavior:
        # `yield` emits values one-by-one, enabling unpacking and iteration.
        yield self.points_df
        yield self.file_name
        yield self.signals_df


@dataclass
class DeltaEntry:
    # Typed record for one analyzed point and its extracted metrics.
    point_number: Any
    label_color: str
    metrics: dict[str, Any]
