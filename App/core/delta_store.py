"""Per-point delta annotations, index maps, and table sync."""

from __future__ import annotations

import traceback
from typing import Any

import numpy as np
import pandas as pd

from .session_store import SessionStore


class DeltaStore:
    """Flat delta list plus (section, point) index maps built from carto data."""

    def __init__(self, carto, session_store: SessionStore | None = None) -> None:
        self.carto = carto
        self.session_store = session_store or SessionStore()
        self.delta: list[Any] = []
        self.to_index: list[list[int]] = []
        self.to_i_j: list[list[int]] = []
        self.labels_memory: list[list[Any]] = []
        self._table_rows: list[Any] = []

    def build_index_map(self, *, hide_labels: bool = True) -> None:
        self.to_index = []
        self.to_i_j = []
        self.labels_memory = []
        self._table_rows = []
        ind = 0
        for i, section in enumerate(self.carto.cont):
            self.to_index.append([])
            self.labels_memory.append([])
            for j, dat in enumerate(section[0].values):
                self.labels_memory[i].append(self.carto.cont[i][0].loc[j, "label_color"])
                if hide_labels:
                    self.carto.cont[i][0].loc[j, "label_color"] = ""
                self.to_index[i].append(ind)
                self.to_i_j.append([i, j])
                self._table_rows.append(dat)
                ind += 1

    def allocate_entries(self) -> None:
        self.delta = [0] * len(self.to_i_j)

    def build_table_dataframe(self) -> pd.DataFrame:
        all_columns = self.carto.cont[0][0].columns
        table = pd.DataFrame(self._table_rows, columns=all_columns)
        n = len(self.to_i_j)
        if "label_color" in table.columns:
            loc = int(table.columns.get_loc("label_color")) + 1
            table.insert(loc, "original_label", np.full(n, "", dtype=object))
        else:
            table["original_label"] = np.full(n, "", dtype=object)
        table = pd.concat([table, pd.DataFrame(np.zeros(n), columns=["Coment"])], axis=1)
        table = pd.concat(
            [table, pd.DataFrame(np.full(n, "", dtype=object), columns=["Prediction"])],
            axis=1,
        )
        table = pd.concat([table, pd.DataFrame(np.zeros(n), columns=["delta"])], axis=1)
        return table

    def original_labels_flat(self) -> list:
        labels = []
        for section in self.labels_memory:
            labels.extend(section)
        return labels

    def label_from_entry(self, index: int, delt) -> str:
        try:
            i, j = self.to_i_j[index]
            point_number = self.carto.cont[i][0].loc[j, "point number"]
            print(delt[0], delt[1], point_number)
            if point_number == delt[0]:
                self.carto.cont[i][0].loc[j, "label_color"] = delt[1]
                return delt[1]
            print(f"mismatch{point_number} {delt[0]}")
            return "mismatch"
        except Exception as e:
            print(e)
            traceback.print_exc()
            return ""

    def summary_from_entry(self, delt) -> list | str:
        try:
            summary = ", ".join(
                [
                    f"{key}: {', '.join([str(ii) for ii in value])}"
                    for key, value in delt[2].items()
                    if "voltage" not in key
                ]
            )
            return [summary]
        except Exception:
            traceback.print_exc()
            return ""

    def text_for_table(self, delt) -> str:
        summary = self.summary_from_entry(delt)
        if isinstance(summary, list):
            return ", ".join(map(str, summary))
        return str(summary)

    def clear_all_pos_proba(self) -> bool:
        changed = False
        for entry in self.delta:
            if isinstance(entry, list) and len(entry) >= 3 and isinstance(entry[2], dict):
                if entry[2].pop("pos_proba", None) is not None:
                    changed = True
        return changed

    def set_carto_label(self, idx: int, label: str) -> None:
        i, j = self.to_i_j[idx]
        df = self.carto.cont[i][0]
        df.iat[j, df.columns.get_loc("label_color")] = label

    def set_entry_label(self, idx: int, label: str) -> None:
        entry = self.delta[idx]
        if isinstance(entry, list) and len(entry) >= 2:
            entry[1] = label

    def save_to(self, path: str) -> None:
        self.session_store.save_delta(path=path, delta=self.delta)

    def load_from(self, path: str) -> None:
        self.delta = self.session_store.load_delta(path=path)

    def refresh_table_columns(self, app, *, refresh_predictions: bool = True) -> None:
        delta_texts = []
        labels = []
        for index, delt in enumerate(self.delta):
            labels.append(self.label_from_entry(index, delt))
            delta_texts.append(self.text_for_table(delt))

        app.Table["delta"] = pd.Series(delta_texts)
        app.Table["label_color"] = pd.Series(labels)
        tv = app.table.tree
        tv.update_column_values("delta", delta_texts)
        tv.update_column_values("label_color", labels)
        if "original_label" in list(tv["columns"]):
            orig = self.original_labels_flat()
            tv.update_column_values("original_label", orig)
            app.Table["original_label"] = orig
            tv.set_column_visible("original_label", bool(app.show_original_labels))
        if refresh_predictions:
            try:
                app._refresh_all_predictions()
            except Exception:
                traceback.print_exc()
