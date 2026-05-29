"""Mediator: keeps the 3D mesh viewer in sync with the table and plots.

The viewer renders one sphere per *table row* (one per electrode point in the
flat ``app.to_i_j`` ordering). When the user clicks a sphere the mediator
forwards the click to ``app.select`` so the table + plot react. When the user
changes the table row or the plot selection, ``sync_from_app`` is called and
the viewer's selected sphere is updated.
"""

from __future__ import annotations

import traceback

import numpy as np

from ..base_components import BaseMeshAppGlue


class MeshAppGlue(BaseMeshAppGlue):
    def __init__(self, app) -> None:
        super().__init__(app)
        self.panel = None
        self.viewer = None
        self._attached = False

    def attach(self, panel) -> None:
        self.panel = panel
        if panel is None:
            return
        self.viewer = panel.viewer
        if self.viewer is None:
            return
        self.viewer.on_pick_callback = self._on_pick_callback
        try:
            self.viewer.set_delta_provider(self)
        except Exception:
            traceback.print_exc()
        self._populate_electrodes()
        self.sync_from_app()
        self._attached = True

    def _populate_electrodes(self) -> None:
        if self.viewer is None:
            return
        positions: list[tuple[float, float, float]] = []
        gindices: list[int] = []
        labels: list[str] = []
        for global_idx, ij in enumerate(self.app.to_i_j):
            i, j = ij
            try:
                df = self.app.carto.cont[i][0]
                row = df.iloc[int(j)]
                x = float(row["x"])
                y = float(row["y"])
                z = float(row["z"])
            except Exception:
                continue
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                continue
            try:
                pn = row["point number"]
            except Exception:
                pn = ""
            positions.append((x, y, z))
            gindices.append(int(global_idx))
            labels.append(f"P{pn}")

        if not positions:
            return
        try:
            self.viewer.set_electrodes(
                np.asarray(positions, dtype=np.float64),
                gindices,
                labels,
            )
        except Exception:
            traceback.print_exc()

    def sync_from_app(self) -> None:
        if self.viewer is None:
            return
        try:
            i, j = int(self.app.i), int(self.app.j)
            global_idx = self.app.to_index[i][j]
            self.viewer.set_selected_global_index(int(global_idx))
        except Exception:
            traceback.print_exc()

    # Called by the viewer when the user clicks/picks an object.
    def _on_pick_callback(self, kind: str, payload, info: dict) -> None:
        try:
            self.on_pick(kind, payload, info)
        except Exception:
            traceback.print_exc()

    def on_pick(self, kind: str, payload, info: dict) -> None:
        if kind != "sphere" or payload is None:
            return
        global_idx = int(payload)
        try:
            self.app.select([global_idx])
        except Exception:
            traceback.print_exc()

    # ------------------------------------------------------- delta provider
    #
    # The plot presenter writes one entry per electrode:
    #
    #   delta[idx] = [point_number, label_color, c1]
    #
    # where ``c1`` is a dict with several list-valued metrics, ONE ENTRY PER
    # STIM (first / second / third) and one per sinus, for example:
    #
    #     c1 = {
    #         "refs_stim":       [r1, r2, r3, ...],
    #         "stim":            [[s,e], [s,e], ...],   # windows, NOT scalars
    #         "voltage_stim":    [v1, v2, v3, ...],
    #         "deflection_stim": [d1, d2, d3, ...],
    #         "refs_sinus": [...], "sinus": [...],
    #         "voltage_sinus": [...], "deflection_sinus": [...],
    #     }
    #
    # Each entry can be ``False`` to indicate "no measurable response for this
    # particular stim". We expose ONE virtual scalar field per metric per stim
    # index, named ``"<key>[i]"`` (1-based), e.g. ``voltage_stim[1]`` is the
    # voltage of the first stim. Window-typed values (``stim``, ``sinus``) are
    # not scalars and are skipped.

    _SKIP_LIST_KEYS = frozenset({"stim", "sinus"})

    # Hard cap on how many per-index variants of a list-typed metric are
    # exposed to the mesh viewer. The user's analysis pipeline reports:
    #   * at most 3 stim windows (S1 S2 S3 pacing protocol),
    #   * at most 1 sinus window (one chosen SR Q per electrode).
    # Older delta entries can still carry more (legacy saves with up to 4
    # sinus); the cap below stops them from leaking extra fields into the
    # field-selection UI of the mesh viewer.
    _MAX_PER_KIND_SUFFIX = (
        ("_stim", 3),
        ("_sinus", 1),
    )

    @classmethod
    def _index_cap_for(cls, key: str) -> int | None:
        for suffix, cap in cls._MAX_PER_KIND_SUFFIX:
            if key.endswith(suffix):
                return cap
        return None

    @staticmethod
    def _entry_metrics(entry) -> dict | None:
        """Return the ``metrics`` mapping of a delta entry, or None."""
        if entry is None or entry == 0:
            return None
        if hasattr(entry, "metrics") and isinstance(getattr(entry, "metrics"), dict):
            return entry.metrics
        try:
            metrics = entry[2]
        except Exception:
            return None
        return metrics if isinstance(metrics, dict) else None

    @staticmethod
    def _to_finite_float(x) -> float:
        """Coerce one cell to a finite float, or NaN if it isn't a real scalar."""
        if x is None:
            return float("nan")
        if isinstance(x, bool):  # explicit "no data" marker the presenter writes
            return float("nan")
        if isinstance(x, (list, tuple)):
            return float("nan")
        if isinstance(x, np.ndarray):
            if x.size != 1:
                return float("nan")
            try:
                v = float(x.item())
            except (TypeError, ValueError):
                return float("nan")
            return v if np.isfinite(v) else float("nan")
        try:
            v = float(x)
        except (TypeError, ValueError):
            return float("nan")
        return v if np.isfinite(v) else float("nan")

    # ``(window_key, ref_key, output_tag)`` triples used to derive LAT fields.
    _LAT_DERIVATIONS = (
        ("stim", "refs_stim", "lat_stim"),
        ("sinus", "refs_sinus", "lat_sinus"),
    )

    def _iter_entry_scalars(self, entry):
        """Yield ``(field_tag, value)`` for each finite scalar in this entry.

        In addition to the raw list-valued metrics, we synthesize per-stim
        local activation times: ``lat_stim[i] = stim[i][0] - refs_stim[i]``
        (and the analogous ``lat_sinus[i]``). These derived fields are what
        you actually want to colour the mesh by.
        """
        metrics = self._entry_metrics(entry)
        if metrics is None:
            return
        # 1) Raw list / scalar metrics from the dict.
        for key, val in metrics.items():
            ks = str(key)
            if ks in self._SKIP_LIST_KEYS:
                # window-only metric: no per-stim scalar to interpolate from
                continue
            if isinstance(val, (list, tuple)) or (
                isinstance(val, np.ndarray) and val.ndim >= 1 and val.size != 1
            ):
                cap = self._index_cap_for(ks)
                for i, sub in enumerate(val):
                    if cap is not None and i >= cap:
                        break
                    v = self._to_finite_float(sub)
                    if np.isfinite(v):
                        yield f"{ks}[{i + 1}]", v
            else:
                v = self._to_finite_float(val)
                if np.isfinite(v):
                    yield ks, v

        # 2) Derived per-stim activation times. The presenter writes
        #    metrics["stim"][i] as [start, end] (or False) and
        #    metrics["refs_stim"][i] as the stim reference time, so the LAT
        #    of the i-th stim is start - ref.
        for win_key, ref_key, out_key in self._LAT_DERIVATIONS:
            windows = metrics.get(win_key)
            refs = metrics.get(ref_key)
            if not isinstance(windows, (list, tuple, np.ndarray)):
                continue
            if not isinstance(refs, (list, tuple, np.ndarray)):
                continue
            n = min(len(windows), len(refs))
            cap = self._index_cap_for(out_key)
            if cap is not None:
                n = min(n, cap)
            for i in range(n):
                w = windows[i]
                if not isinstance(w, (list, tuple, np.ndarray)) or len(w) < 1:
                    continue
                try:
                    start = float(w[0])
                    ref = float(refs[i])
                except (TypeError, ValueError):
                    continue
                lat = start - ref
                if np.isfinite(lat):
                    yield f"{out_key}[{i + 1}]", lat

    def get_delta_metric_keys(self) -> list[str]:
        """Union of available ``<metric>[i]`` tags across every populated entry."""
        keys: set[str] = set()
        for entry in getattr(self.app, "delta", []) or []:
            for tag, _v in self._iter_entry_scalars(entry):
                keys.add(tag)
        return sorted(keys, key=self._sort_key)

    @staticmethod
    def _sort_key(tag: str):
        """Sort so that ``voltage_stim[1] < voltage_stim[2] < voltage_stim[10]``."""
        bracket = tag.find("[")
        if bracket < 0:
            return (tag, -1)
        try:
            idx = int(tag[bracket + 1 : -1])
        except ValueError:
            idx = -1
        return (tag[:bracket], idx)

    def get_delta_values_for(self, key: str) -> dict[int, float]:
        """Return ``{global_idx -> float}`` for electrodes that have ``key``."""
        out: dict[int, float] = {}
        delta = getattr(self.app, "delta", None) or []
        to_ij = getattr(self.app, "to_i_j", None) or []
        for gidx, entry in enumerate(delta):
            try:
                i, j = to_ij[gidx]
                lab = str(self.app.carto.cont[i][0].loc[int(j), "label_color"]).strip().lower()
            except Exception:
                lab = ""
            if lab == "reject":
                continue
            for tag, v in self._iter_entry_scalars(entry):
                if tag == key:
                    out[gidx] = v
                    break
        return out

    # ---------------------------------- forward triple-extra updates
    def notify_delta_changed(self, global_idx=None) -> None:
        if self.viewer is None:
            return
        try:
            self.viewer.notify_delta_changed(global_idx)
        except Exception:
            traceback.print_exc()
