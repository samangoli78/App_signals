from __future__ import annotations
# Future annotations:
# keep hints as strings until runtime resolves them.

# Import order note:
# built-in/third-party imports first, then local application imports.
import re
import traceback

import numpy as np
import librosa
from matplotlib.patches import Rectangle
from scipy.signal import find_peaks

from ..analyzers import DefaultAnalyzer, TripleExtraAnalyzer
from ..base_components import BasePlotPresenter
from ..mediators.work_queue import LatestWinsWorker
from ..triple_extra import Triple_Extra
from ..ui import Area
from ..utility import (
    butter_bandpass_filter,
    filter_indices_outside_windows,
    find_start,
    r_q_markers_for_display,
    stim_pulse_exclusion_spans_signed_m,
)
from ..utility.signals_ecg import pan_tompkins_r_indices, q_indices_before_r

SR_PRE_MS = 300
SR_POST_MS = 100
STIM_SLICE_HEAD = 8
STIM_SLICE_TAIL = 10


def _decimate_xy(x: np.ndarray, y: np.ndarray, max_pts: int = 12_000) -> tuple[np.ndarray, np.ndarray]:
    """Down-sample long traces so matplotlib line draws stay interactive."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = min(x.size, y.size)
    if n <= max_pts:
        return x[:n], y[:n]
    step = int(np.ceil(n / max_pts))
    return x[:n:step], y[:n:step]


def _ref_ecg_compute(payload: dict) -> dict:
    """Worker-thread: band-pass + Pan–Tompkins R + Q nadir (no Tk / no mpl).

    When ``payload['stim_data']`` is provided, R / Q indices that fall inside
    any detected stim pulse window are dropped — sinus reference markers must
    never sit on pacing artefacts.
    """
    raw = payload["raw_ref"]
    fs = float(payload.get("fs", 1000.0))
    v, r_idx, q_idx = r_q_markers_for_display(raw, fs=fs)

    stim_data = payload.get("stim_data")
    if stim_data is not None and np.asarray(stim_data).size > 0:
        s_starts, s_ends = stim_pulse_exclusion_spans_signed_m(
            np.asarray(stim_data, dtype=np.float64), fs=fs
        )
        if s_starts.size:
            r_idx = filter_indices_outside_windows(r_idx, s_starts, s_ends)
            q_idx = filter_indices_outside_windows(q_idx, s_starts, s_ends)
    return {"v_ref": v, "r_idx": r_idx, "q_idx": q_idx}


def compute_deflection_peaks(y, start, end):
    """Math-only deflection extractor: return peak indices in selected window."""
    signal = y[start:end]
    if len(signal) == 0:
        return np.array([], dtype=int), signal
    peak_indices, _ = find_peaks(signal, prominence=signal.max() * 0.15, height=signal.max() * 0.2, distance=5)
    return peak_indices, signal


def plot_deflection_peaks(ax, peak_indices, signal, start, inverse=False):
    """Plot-only helper for deflection peaks."""
    if len(peak_indices) == 0:
        return
    y_values = -signal[peak_indices] if inverse else signal[peak_indices]
    ax.scatter((peak_indices + start) * 0.001, y_values, s=2, color="black")


def deflection(y, ax, start, end, inverse=False):
    # Feature extraction concept:
    # count local peaks in a selected signal region to estimate deflection complexity.
    peak_indices, signal = compute_deflection_peaks(y=y, start=start, end=end)
    plot_deflection_peaks(ax=ax, peak_indices=peak_indices, signal=signal, start=start, inverse=inverse)
    return len(peak_indices)


class PlotPresenter(BasePlotPresenter):
    def __init__(self, app):
        # Strategy pattern:
        # choose analyzer behavior at runtime (default vs triple-extra mode).
        super().__init__(app)
        self.default_analyzer = DefaultAnalyzer()
        self.triple_analyzer = TripleExtraAnalyzer()
        # Background worker for the heavy triple-extra compute (filtfilt +
        # librosa spectrograms + per-window analytics). One pending slot,
        # latest-wins: clicking through points while a previous solve is
        # still running just replaces the request — never queues. We bind
        # ``tk_after`` lazily because ``self.app.after`` doesn't exist until
        # ``tk.Tk.__init__`` has run (which happens in ``App.start()``).
        def _tk_after(ms, fn):
            return self.app.after(ms, fn)

        self._plot_worker = LatestWinsWorker(tk_after=_tk_after, name="plot-compute")
        self._ref_worker = LatestWinsWorker(tk_after=_tk_after, name="ref-ecg")
        self._plot_done_listeners: list = []
        self._plot_tick = 0
        self._current_ref_col_name: str | None = None

    # ---------------------------------------------------------------- events
    def add_plot_done_listener(self, cb) -> None:
        if cb not in self._plot_done_listeners:
            self._plot_done_listeners.append(cb)

    def _fire_plot_done(self) -> None:
        for cb in list(self._plot_done_listeners):
            try:
                cb()
            except Exception:
                traceback.print_exc()

    def on_right_click(self, event):
        # Mouse interaction state machine:
        # start rectangle-selection tracking when right click begins.
        if event.button == 3:
            for axis_name, axis in self.app.axes.items():
                if axis == event.inaxes:
                    self.app.selected_ax = axis
                    self.app.selected_axis_name = axis_name
                    self.app.start_x_y = [event.xdata, event.ydata]
                    self.app.end_x_y = self.app.start_x_y
                    self.app.follower = self.app.canvas.mpl_connect("motion_notify_event", self.on_right_follow_mouse)
                    self.app.rect = None

    def on_right_follow_mouse(self, event):
        self.app.end_x_y = [event.xdata, event.ydata]
        if self.app.rect:
            self.app.rect.set_xy(self.app.start_x_y)
            self.app.rect.set_height(self.app.end_x_y[1] - self.app.start_x_y[1])
            self.app.rect.set_width(self.app.end_x_y[0] - self.app.start_x_y[0])
        else:
            rect = Rectangle(
                self.app.start_x_y,
                width=self.app.end_x_y[0] - self.app.start_x_y[0],
                height=self.app.end_x_y[1] - self.app.start_x_y[1],
                color="#00008525",
            )
            self.app.selected_ax.add_artist(rect)
            self.app.rect = rect
        self.app.canvas.draw_idle()

    def on_right_release(self, event):
        if event.button != 3:
            return
        if self.app.rect:
            self.app.rect.remove()
        self.app.canvas.mpl_disconnect(self.app.follower)
        if event.inaxes == self.app.selected_ax:
            xlim = [self.app.start_x_y[0], self.app.end_x_y[0]]
            ylim = [self.app.start_x_y[1], self.app.end_x_y[1]]
            if xlim[1] - xlim[0] < 0:
                xlim = xlim[-1::-1]
            if ylim[1] - ylim[0] < 0:
                ylim = ylim[-1::-1]
            if (xlim[1] - xlim[0]) < 0.01 and (ylim[1] - ylim[0]) < 0.01:
                if self.app.selected_axis_name == "bot":
                    xlim, ylim = [0, 2.5], [-10, 10]
                else:
                    xlim, ylim = [0, 2.5], [-1, 1]
            self.app.selected_ax.set_xlim(xlim)
            self.app.selected_ax.set_ylim(ylim)
            for area in self.app.Areas[self.app.selected_axis_name]:
                area.configure_shade_attr(x=area.x, y=self.app.selected_ax.get_ylim())
                for line in area.lines:
                    line.set_ydata(self.app.selected_ax.get_ylim())
        self.app.canvas.draw_idle()

    def on_scroll(self, event):
        # Guard clause:
        # skip axes that were not the scroll target to avoid unintended zoom changes.
        for axis_name, axis in self.app.axes.items():
            if axis != event.inaxes:
                continue
            ylim = axis.get_ylim()
            zoom_factor = 0.1
            if event.button == "up":
                scale_factor = 1 - zoom_factor
            elif event.button == "down":
                scale_factor = 1 + zoom_factor
            else:
                return
            axis.set_ylim([ylim[0] * scale_factor, ylim[1] * scale_factor])
            if self.app.Areas[axis_name]:
                for area in self.app.Areas[axis_name]:
                    area.configure_shade_attr(x=area.x, y=axis.get_ylim())
                    for line in area.lines:
                        line.set_ydata(axis.get_ylim())
        self.app.canvas.draw_idle()

    @staticmethod
    def _resolve_channel_name(df, channel_name: str) -> str:
        """Resolve channel names across exact, deduped, and base-name variants."""
        # @staticmethod means this function is namespaced inside the class
        # but does not need instance state (`self`).
        channel_name = str(channel_name)
        if channel_name in df.columns:
            return channel_name

        # If duplicate columns were made unique, they may appear as "<name>_1".
        prefixed = [col for col in df.columns if col.startswith(f"{channel_name}_")]
        # List comprehension builds a filtered list in one expression.
        if prefixed:
            return prefixed[0]

        # Normalization concept:
        # convert different string formats into a comparable canonical form.
        def canon(name: str) -> str:
            name = str(name).strip().upper()
            name = re.sub(r"\(\d+\)", "", name)   # drop channel index annotations like (23)
            name = re.sub(r"_\d+$", "", name)     # drop dedupe suffixes like _1
            name = re.sub(r"\s+", "", name)       # normalize whitespace
            return name

        # Fallback: compare canonical base forms.
        target = canon(channel_name)
        for col in df.columns:
            if canon(col) == target:
                return col

        # If channel is annotated like V2(23), use index hint when headers are numeric-only.
        idx_match = re.search(r"\((\d+)\)\s*$", channel_name)
        if idx_match:
            idx = int(idx_match.group(1))
            for candidate in (idx, idx - 1):  # tolerate both 0-based and 1-based conventions
                if 0 <= candidate < len(df.columns):
                    return df.columns[candidate]

        # Last-resort prefix match only when both sides are non-numeric channel-like strings.
        for col in df.columns:
            c_col = canon(col)
            if c_col.isdigit() or target.isdigit():
                continue
            if c_col.startswith(target) or target.startswith(c_col):
                return col

        raise KeyError(channel_name)

    def _resolve_first_existing(self, df, candidates):
        # Fallback chain:
        # try candidates in priority order until one resolves.
        for candidate in candidates:
            if candidate is None:
                continue
            try:
                return self._resolve_channel_name(df, candidate)
            except KeyError:
                # Expected exception for missing channels; continue trying alternatives.
                continue
        raise KeyError(f"No channel found in candidates: {candidates}")

    # ---------------------------------------------------------------- plot_main
    def plot_main(self, ax, x, y2, arg="top", reff=None):
        """Schedule the triple-extra pipeline on the background worker.

        The Tk thread does only the cheap snapshot work and returns
        immediately. ``_pm_compute`` runs on the worker (no matplotlib /
        no Tk), then ``_pm_apply`` draws the result on the Tk main loop.
        """
        if getattr(self.app, "_compute_all_running", False):
            return
        try:
            payload = self._pm_snapshot(ax, x, y2, arg, reff)
        except Exception:
            traceback.print_exc()
            self._fire_plot_done()
            return
        if payload is None:
            self._fire_plot_done()
            return
        self._plot_worker.post(self._pm_compute, self._pm_apply, payload)

    def _resolve_ref_col(self, data_pd, ref_channel: str | None):
        """Resolve THE reference column used for both PT markers and SR anchoring.

        Priority: V5 first (canonical surface lead the user trusts for Q
        detection), then the metadata-declared ``refference_chanel``, then
        common intracardiac fallbacks. Same priority is reused by
        ``plot()`` (red R/Q overlays on bot axis) and ``_pm_snapshot``
        (SR-window Q indices inside ``Triple_Extra``) so the shaded SR
        windows always line up with the visible PT markers.
        """
        candidates = ["V5", ref_channel, "CS1", "M4", "M3"]
        try:
            return self._resolve_first_existing(data_pd, candidates)
        except Exception:
            return None

    def _pm_snapshot(self, ax, x, y2, arg, reff):
        reff_arr = np.asarray([] if reff is None else reff, dtype=np.float64)
        data_pd = self.app.cont[self.app.i][2]
        car = self.app.cont[self.app.i][0]
        try:
            ref_meta = str(car.loc[self.app.j, "refference_chanel"])
        except Exception:
            ref_meta = None
        # Reuse the column already chosen by plot() for the red R/Q markers,
        # so the SR windows are anchored on the exact same Q indices the user
        # sees on the bot axis. Falls back to the V5-prioritised resolver if
        # plot() didn't run first (defensive — e.g. tests / scripted calls).
        reference_col = getattr(self, "_current_ref_col_name", None)
        if reference_col is None or reference_col not in data_pd.columns:
            reference_col = self._resolve_ref_col(data_pd, ref_meta)
        if reference_col is None:
            traceback.print_exc()
            return None
        # We must NOT hand back direct references to the cell's Series for the
        # worker, in case the App swaps cont entries mid-flight. ``.values``
        # gives a numpy copy view; copy() pins it.
        reference_data = np.asarray(data_pd[reference_col].values, dtype=np.float64).copy()
        if reff_arr.size > 0:
            stim_data = np.asarray(reff_arr, dtype=np.float64).copy()
            stim_margin = 0
        else:
            # Defensive fallback when plot() did not pass an M differential
            # (scripted call path). Rebuild M1 − M2 (or M − M2), and only
            # fall back to single-channel detection if both are missing.
            stim_data = None
            for hi_name, lo_name in (("M1", "M2"), ("M", "M2")):
                try:
                    hi_col = self._resolve_channel_name(data_pd, hi_name)
                    lo_col = self._resolve_channel_name(data_pd, lo_name)
                    stim_data = np.asarray(
                        data_pd.loc[:, hi_col].values - data_pd.loc[:, lo_col].values,
                        dtype=np.float64,
                    ).copy()
                    break
                except KeyError:
                    stim_data = None
                    continue
            if stim_data is None:
                try:
                    stim_col = self._resolve_first_existing(
                        data_pd, ["CS1", "M4", "M3", ref_meta, "V5"]
                    )
                except Exception:
                    traceback.print_exc()
                    return None
                stim_data = np.asarray(data_pd[stim_col].values, dtype=np.float64).copy()
            stim_margin = 20
        row_idx = self.app.to_index[self.app.i][self.app.j]
        cached_delta = self.app.delta[row_idx]
        return {
            "ax_ref": ax,
            "arg": arg,
            "x": np.asarray(x, dtype=np.float64).copy(),
            "y2": np.asarray(y2, dtype=np.float64).copy(),
            "stim_data": stim_data,
            "stim_margin": stim_margin,
            "reff_was_supplied": bool(reff_arr.size > 0),
            "reff_passthrough": stim_data,  # what to bake into c1's stim reference
            "reference_data": reference_data,
            "row_idx": row_idx,
            "i": int(self.app.i),
            "j": int(self.app.j),
            "forcefull": bool(self.app.forcefull),
            "TH": float(self.app.TH[0]),
            "cached_delta": cached_delta,
            "point_number": self.app.cont[self.app.i][0]["point number"].values[self.app.j],
            "label_color": self.app.cont[self.app.i][0]["label_color"].values[self.app.j],
            # Energy params snapshot — used by worker's energy compute.
            "energy_params": {
                "n_fft": int(self.app.win_length[0]) + 25,
                "hop_length": int(self.app.hop_length[0]),
                "win_length": int(self.app.win_length[0]),
                "len_han": int(self.app.len_hann[0]),
                "low_b0": int(self.app.low_b0[0]),
                "low_b1": int(self.app.low_b1[0]),
            },
        }

    def _pm_compute(self, payload):
        """Pure compute (worker thread). No Tk, no matplotlib, no app.* writes."""
        x = payload["x"]
        y2 = payload["y2"]
        stim_data = payload["stim_data"]
        ref_data = payload["reference_data"]
        stim_margin = int(payload["stim_margin"])
        cached_delta = payload["cached_delta"]
        forcefull = bool(payload["forcefull"])
        TH = payload["TH"]
        ep = payload["energy_params"]

        # Reference signal: bandpass to clean for peak detection / drawing.
        reference_signal = butter_bandpass_filter(
            data=ref_data, cutoff=[5, 180], fs=1000, order=2
        )

        egm = Triple_Extra(t=x, EGM=y2)
        windows_info = egm.compute_windows(
            stimulation=stim_data, reference=reference_signal, margin=stim_margin
        )

        # Pre-filter once for deflection — used by both stim and sinus loops.
        bp_signal = butter_bandpass_filter(y2, (2, 250), order=2)

        c1 = {
            "stim": [],
            "sinus": [],
            "refs_stim": [],
            "refs_sinus": [],
            "voltage_stim": [],
            "voltage_sinus": [],
            "deflection_stim": [],
            "deflection_sinus": [],
        }
        stim_results: list[dict] = []
        sinus_results: list[dict] = []

        stim_starts = list(egm.stim_start)
        stim_durs = list(egm.stim_duration)
        stim_refs = list(egm.stim_ref)

        for ii in range(len(stim_starts)):
            dont_save = False
            start = int(stim_starts[ii])
            end = int(stim_starts[ii]) + int(stim_durs[ii])
            xx = x[start + 8 : end - 10]
            yy = y2[start + 8 : end - 10]
            yy1 = yy
            if xx.size == 0 or yy.size == 0:
                continue
            x_energy, y_low, y_high, _y_total = self._energy_compute(xx, yy, ep)
            # Window boundary: cached unless forcefull / no cache.
            if cached_delta == 0 or forcefull:
                output = find_start(
                    x_energy, y_low, length=2, ax=None, operation=None, Th=0.15, alpha=TH
                )
                if output is not None:
                    start_n, end_n = int(start + 8 + output[0]), int(start + 8 + output[1])
                else:
                    start_n, end_n = start, end
                    dont_save = True
            else:
                memory = cached_delta[2]["stim"][ii] if ii < len(cached_delta[2]["stim"]) else None
                if isinstance(memory, list) and len(memory) == 2:
                    start_n, end_n = int(memory[0]), int(memory[1])
                else:
                    start_n, end_n = start, end
                    dont_save = True
            xx_w = x[start_n:end_n]
            yy_w = y2[start_n:end_n]
            ok = bool(yy1.max() - yy1.min() > 0.05 and not dont_save)
            defl = None
            pi_pos = pi_neg = None
            sig_pos = sig_neg = None
            if ok:
                pi_pos, sig_pos = compute_deflection_peaks(bp_signal, start_n, end_n)
                pi_neg, sig_neg = compute_deflection_peaks(-bp_signal, start_n, end_n)
                defl = int(len(pi_pos) + len(pi_neg))
            stim_results.append({
                "ii": int(ii),
                "start": start, "end": end,
                "start_n": int(start_n), "end_n": int(end_n),
                "voltage": float(yy1.max() - yy1.min()),
                "xx_w": xx_w, "yy_w": yy_w,
                "x_energy": x_energy, "y_low": y_low, "y_high": y_high,
                "ok": ok, "dont_save": bool(dont_save), "defl": defl,
                "pi_pos": pi_pos, "sig_pos": sig_pos,
                "pi_neg": pi_neg, "sig_neg": sig_neg,
                "stim_ref": int(stim_refs[ii]),
            })
            if ok:
                c1["refs_stim"].append(int(stim_refs[ii]))
                c1["stim"].append([int(start_n), int(end_n)])
                c1["voltage_stim"].append(float(yy1.max() - yy1.min()))
                c1["deflection_stim"].append(int(defl))
            elif dont_save:
                c1["refs_stim"].append(int(stim_refs[ii]))
                c1["stim"].append(False)
                c1["voltage_stim"].append(False)
                c1["deflection_stim"].append(False)
            else:
                c1["refs_stim"].append(int(stim_refs[ii]))
                c1["stim"].append([int(start_n), int(end_n)])
                c1["voltage_stim"].append(float(yy1.max() - yy1.min()))
                c1["deflection_stim"].append(False)

        if stim_starts:
            stim_block_lo = int(stim_starts[0])
            stim_block_hi = int(stim_starts[-1]) + int(stim_durs[-1])
        else:
            stim_block_lo = -1
            stim_block_hi = -1

        # SR: locate Q, delineate inside [Q−300 ms, Q+100 ms] with stim params.
        cached_sinus = None
        if (
            cached_delta != 0
            and isinstance(cached_delta, list)
            and len(cached_delta) >= 3
            and isinstance(cached_delta[2], dict)
        ):
            sinus_mem = cached_delta[2].get("sinus") or []
            if sinus_mem:
                cached_sinus = sinus_mem[0]

        ref_idx = self._pick_sr_q_index(
            reference_signal,
            stim_block_lo,
            stim_block_hi,
            egm,
        )
        sr_out = None
        if ref_idx is not None:
            sr_out = self._delineate_sr_at_q(
                x,
                y2,
                bp_signal,
                int(ref_idx),
                cached_sinus,
                forcefull,
                TH,
                ep,
            )

        if sr_out is not None:
            ii = 0
            sinus_results.append({
                "ii": int(ii),
                "start": int(sr_out["coarse_start"]),
                "end": int(sr_out["coarse_end"]),
                "start_n": int(sr_out["start_n"]),
                "end_n": int(sr_out["end_n"]),
                "voltage": float(sr_out["voltage"]),
                "xx_w": sr_out["xx_w"],
                "yy_w": sr_out["yy_w"],
                "x_energy": sr_out["x_energy"],
                "y_low": sr_out["y_low"],
                "y_high": sr_out["y_high"],
                "ok": bool(sr_out["ok"]),
                "dont_save": bool(sr_out["dont_save"]),
                "defl": sr_out["defl"],
                "pi_pos": sr_out["pi_pos"],
                "sig_pos": sr_out["sig_pos"],
                "pi_neg": sr_out["pi_neg"],
                "sig_neg": sr_out["sig_neg"],
                "sinus_ref": int(sr_out["ref_idx"]),
            })
            if sr_out["ok"]:
                c1["refs_sinus"].append(int(sr_out["ref_idx"]))
                c1["sinus"].append([int(sr_out["start_n"]), int(sr_out["end_n"])])
                c1["voltage_sinus"].append(float(sr_out["voltage"]))
                c1["deflection_sinus"].append(int(sr_out["defl"]))
            elif sr_out["dont_save"]:
                c1["refs_sinus"].append(int(sr_out["ref_idx"]))
                c1["sinus"].append(False)
                c1["voltage_sinus"].append(False)
                c1["deflection_sinus"].append(False)
            else:
                c1["refs_sinus"].append(int(sr_out["ref_idx"]))
                c1["sinus"].append([int(sr_out["start_n"]), int(sr_out["end_n"])])
                c1["voltage_sinus"].append(float(sr_out["voltage"]))
                c1["deflection_sinus"].append(False)
            pre_n = int(SR_PRE_MS * 1000 / 1000)
            post_n = int(SR_POST_MS * 1000 / 1000)
            windows_info["sinus_start"] = [max(0, int(sr_out["ref_idx"]) - pre_n)]
            windows_info["sinus_ref"] = [int(sr_out["ref_idx"])]
            windows_info["sinus_duration"] = [pre_n + post_n]

        return {
            "windows_info": windows_info,
            "stim_results": stim_results,
            "sinus_results": sinus_results,
            "c1": c1,
        }

    def _pm_persist_delta_and_table(self, payload: dict, result: dict) -> None:
        """Write ``app.delta`` row and refresh the table's delta column cell."""
        c1 = result["c1"]
        self.app.delta[payload["row_idx"]] = [
            payload["point_number"],
            payload["label_color"],
            c1,
        ]
        tv = self.app.table.tree
        iid = f"row{payload['row_idx']}"
        try:
            vals = list(tv.item(iid)["values"])
            vals[-1] = ", ".join(
                f"{k}: {', '.join(map(str, v))}"
                for k, v in c1.items()
                if "voltage" not in k
            )
            tv.item(iid, values=vals)
        except Exception:
            traceback.print_exc()
        # Predict for this row using the currently selected ML model(s),
        # if any. Both interactive plotting and Compute all go through
        # this persist step, so a single hook covers both flows.
        try:
            predictor = getattr(self.app, "_predict_row", None)
            if callable(predictor):
                predictor(int(payload["row_idx"]))
        except Exception:
            traceback.print_exc()

    def _pm_apply(self, payload, result):
        """Matplotlib drawing + app.delta / table update on the Tk thread."""
        plotless = bool(payload.get("_compute_all_plotless"))
        after_apply = payload.get("_compute_all_after_apply")
        try:
            if result is None or result.get("_compute_failed"):
                return
            if plotless:
                self._pm_persist_delta_and_table(payload, result)
                return

            ax = payload["ax_ref"]
            arg = payload["arg"]
            x = payload["x"]
            y2 = payload["y2"]
            addition = [["#FFf5F5", "High frequency"], ["#FFc5c5", "Low frequency"]]

            # Base signal trace + bot-axis scatter markers from find_windows.
            ax.plot(x, y2, alpha=0.5, linewidth=0.6)
            bot_ax = self.app.axes.get("bot")
            wi = result.get("windows_info") or {}
            # Only draw stim pulse markers (entry 0). Pre/post Q markers are
            # drawn separately on the band-passed reference trace by the
            # ECG worker (red triangles), so don't duplicate them here.
            if bot_ax is not None:
                scatter = wi.get("scatter", [])
                if scatter:
                    xs, ys = scatter[0]
                    if xs.size and ys.size:
                        bot_ax.plot(xs, ys, "x", color="#FFCC33", label="Stim")

            energy_on = any(
                state.get() and key.lower() == "energy"
                for key, state in self.app.check_boxes.items()
            )

            def _draw_energy(entry):
                if not energy_on:
                    return
                ax.plot(entry["x_energy"], entry["y_high"], color=addition[0][0])
                ax.plot(entry["x_energy"], entry["y_low"], color=addition[1][0])

            def _draw_deflection_scatter(entry):
                if entry["pi_pos"] is not None:
                    plot_deflection_peaks(ax, entry["pi_pos"], entry["sig_pos"], entry["start_n"], inverse=False)
                if entry["pi_neg"] is not None:
                    plot_deflection_peaks(ax, entry["pi_neg"], entry["sig_neg"], entry["start_n"], inverse=True)

            for entry in result.get("stim_results", []):
                _draw_energy(entry)
                if entry["ok"]:
                    _draw_deflection_scatter(entry)
                    area = Area(self.app, ind=entry["ii"])
                    area.add_area(
                        np.array([entry["start_n"], entry["end_n"]]),
                        ylim=ax.get_ylim(),
                        xlim=ax.get_xlim(),
                        color="green",
                    )
                    area.plot_area(ax)
                    self.app.Areas[arg].append(area)
                    self.app.canvas.mpl_connect("pick_event", area.clickonline)
                    ax.plot(
                        entry["xx_w"], entry["yy_w"],
                        label=f"duration: {entry['end_n']-entry['start_n']} ms",
                        linewidth=0.6,
                    )

            for entry in result.get("sinus_results", []):
                _draw_energy(entry)
                if entry["ok"]:
                    _draw_deflection_scatter(entry)
                    area = Area(self.app, ind=entry["ii"])
                    area.add_area(
                        np.array([entry["start_n"], entry["end_n"]]),
                        ylim=ax.get_ylim(),
                        xlim=ax.get_xlim(),
                        color="#A776AD",
                    )
                    area.plot_area(ax)
                    self.app.Areas[arg].append(area)
                    self.app.canvas.mpl_connect("pick_event", area.clickonline)
                    ax.plot(
                        entry["xx_w"], entry["yy_w"],
                        label=f"duration: {entry['end_n']-entry['start_n']} ms",
                        linewidth=0.6,
                    )

            self._pm_persist_delta_and_table(payload, result)

            ax.set_title(payload["label_color"])
            self.create_legend(
                leg=ax.get_legend_handles_labels(), canvas=self.app.ccs[arg], addition=addition
            )
            try:
                self.app.canvas.draw_idle()
            except Exception:
                traceback.print_exc()
        finally:
            if plotless and callable(after_apply):
                try:
                    self.app.after(0, after_apply)
                except Exception:
                    traceback.print_exc()
            elif not plotless:
                # Always tell listeners (mesh interp etc.) that the plot finished
                # even if drawing raised — otherwise the 3D viewer would stop
                # updating its delta colours.
                self._fire_plot_done()

    def _pm_compute_safe(self, payload: dict) -> dict | None:
        """Worker entry: never raises; returns ``{_compute_failed: True}`` on error."""
        try:
            return self._pm_compute(payload)
        except Exception:
            traceback.print_exc()
            return {"_compute_failed": True}

    def _signal_bundle_for_table_row(self, row_idx: int) -> dict | None:
        """Mid-axis bipolar trace + stim reference for ``to_i_j[row_idx]``, or None."""
        try:
            i, j = self.app.to_i_j[int(row_idx)]
        except Exception:
            return None
        data_pd = self.app.cont[i][2]
        car = self.app.cont[i][0]
        requested_unipolar = car.iat[j, car.columns.get_loc("unipolar")]
        requested_bipolar = car.iat[j, car.columns.get_loc("bipolar")]
        try:
            unipolar_col = data_pd.columns[data_pd.columns.get_loc(requested_unipolar)]
        except KeyError:
            unipolar_col = None
        try:
            bipolar_col = data_pd.columns[data_pd.columns.get_loc(requested_bipolar)]
        except KeyError:
            bipolar_col = None
        if unipolar_col is None or bipolar_col is None:
            return None
        x = np.asarray(data_pd.index, dtype=np.float64)
        y2 = np.asarray(data_pd.loc[:, bipolar_col].values, dtype=np.float64)
        ref_channel = None
        try:
            ref_channel = str(car.loc[j, "refference_chanel"])
        except Exception:
            ref_channel = None
        v5_col_name = self._resolve_ref_col(data_pd, ref_channel)
        self._current_ref_col_name = v5_col_name
        m = None
        for hi_name, lo_name in (("M1", "M2"), ("M", "M2")):
            try:
                hi_col = self._resolve_channel_name(data_pd, hi_name)
                lo_col = self._resolve_channel_name(data_pd, lo_name)
                m = np.asarray(
                    data_pd.loc[:, hi_col].values - data_pd.loc[:, lo_col].values,
                    dtype=np.float64,
                )
                break
            except KeyError:
                m = None
                continue
        if m is None:
            try:
                cs1_col = self._resolve_channel_name(data_pd, "CS1")
                m = np.asarray(data_pd.loc[:, cs1_col].values, dtype=np.float64)
            except KeyError:
                m = None
        return {
            "i": int(i),
            "j": int(j),
            "x": x,
            "y2": y2,
            "m": m,
            "ax": self.app.axes["mid"],
        }

    def start_compute_all_row(self, row_idx: int, *, after_apply) -> None:
        """Background delta for one table row; table updated on Tk thread; optional chain.

        When ``_compute_all_plotless`` is set in the payload, drawing is skipped
        and ``after_apply`` is scheduled after persist (used for batch compute).
        """
        bundle = self._signal_bundle_for_table_row(row_idx)
        if bundle is None:
            self.app.after(0, after_apply)
            return
        self.app.i = bundle["i"]
        self.app.j = bundle["j"]
        saved_ff = bool(self.app.forcefull)
        self.app.forcefull = True
        try:
            reff = bundle["m"]
            if reff is None:
                reff = np.asarray([], dtype=np.float64)
            payload = self._pm_snapshot(bundle["ax"], bundle["x"], bundle["y2"], "mid", reff)
        finally:
            self.app.forcefull = saved_ff
        if payload is None:
            self.app.after(0, after_apply)
            return
        payload["_compute_all_plotless"] = True
        payload["_compute_all_after_apply"] = after_apply
        self._plot_worker.post(self._pm_compute_safe, self._pm_apply, payload)

    def _ref_track_apply(self, payload: dict, result: dict | None) -> None:
        """Draw V5 / reference trace + Pan–Tompkins R / Q markers (Tk thread)."""
        if result is None or payload.get("cycle") != self._plot_tick:
            return
        ax1 = self.app.axes["bot"]
        x_full = np.asarray(payload["x_full"], dtype=np.float64).ravel()
        v = np.asarray(result["v_ref"], dtype=np.float64).ravel()
        n = min(x_full.size, v.size)
        if n <= 0:
            return
        x_full = x_full[:n]
        v = v[:n]
        xd, yd = _decimate_xy(x_full, v)
        ax1.plot(xd, yd, label="Reference (V5)", alpha=1.0, color="#7fdbff", linewidth=0.55)
        m_dec = payload.get("m_dec")
        if m_dec is not None:
            xm, ym = m_dec
            if xm.size and ym.size:
                ax1.plot(xm, ym, label="M", alpha=0.9, linewidth=0.5)
        r_idx = np.asarray(result.get("r_idx", []), dtype=np.int64)
        q_idx = np.asarray(result.get("q_idx", []), dtype=np.int64)
        r_idx = r_idx[(r_idx >= 0) & (r_idx < n)]
        q_idx = q_idx[(q_idx >= 0) & (q_idx < n)]
        if q_idx.size:
            ax1.scatter(
                x_full[q_idx], v[q_idx], c="#ff5555", s=14, marker="v",
                zorder=6, linewidths=0.4, edgecolors="white", label="Q (PT)",
            )
        if r_idx.size:
            ax1.scatter(
                x_full[r_idx], v[r_idx], c="#44ff88", s=16, marker="o",
                zorder=5, linewidths=0.4, edgecolors="black", label="R (PT)",
            )
        ax1.grid(True)
        try:
            self.create_legend(leg=ax1.get_legend_handles_labels(), canvas=self.app.ccs["bot"])
        except Exception:
            traceback.print_exc()
        try:
            self.app.canvas.draw_idle()
        except Exception:
            traceback.print_exc()

    def plot(self):
        # Presenter concept:
        # this method translates model/data state into what the user sees on each axis.
        if getattr(self.app, "_compute_all_running", False):
            return
        self._plot_tick += 1
        tick = self._plot_tick

        [canvas.delete("all") for canvas in self.app.ccs.values()]
        self.app.Areas = {axis_name: [] for axis_name in self.app.axes.keys()}

        data_pd = self.app.cont[self.app.i][2]
        x = np.asarray(data_pd.index, dtype=np.float64)
        car = self.app.cont[self.app.i][0]

        requested_unipolar = car.iat[self.app.j, car.columns.get_loc("unipolar")]
        requested_bipolar = car.iat[self.app.j, car.columns.get_loc("bipolar")]
        try:
            unipolar_col = data_pd.columns[data_pd.columns.get_loc(requested_unipolar)]
        except KeyError:
            unipolar_col = None
        try:
            bipolar_col = data_pd.columns[data_pd.columns.get_loc(requested_bipolar)]
        except KeyError:
            bipolar_col = None

        if unipolar_col is None or bipolar_col is None:
            self._fire_plot_done()
            try:
                self.app.canvas.draw_idle()
            except Exception:
                traceback.print_exc()
            return

        y_uni = np.asarray(data_pd.loc[:, unipolar_col].values, dtype=np.float64)
        y2 = np.asarray(data_pd.loc[:, bipolar_col].values, dtype=np.float64)

        ref_channel = None
        try:
            ref_channel = str(car.loc[self.app.j, "refference_chanel"])
        except Exception:
            ref_channel = None
        # Resolve the V5/reference column ONCE with V5 prioritised, fuzzy
        # match (handles deduped or annotated headers like "V5_1", "V5(7)").
        # Stash it so ``_pm_snapshot`` uses the exact same column for the
        # SR-window Q anchor → shading aligns with the red Q markers.
        v5_col_name = self._resolve_ref_col(data_pd, ref_channel)
        self._current_ref_col_name = v5_col_name
        v5_col = v5_col_name if (v5_col_name is not None and v5_col_name in data_pd.columns) else None

        # Top + cheap bot trace: decimated so matplotlib stays responsive on
        # long Carto exports.
        ax_uni = self.app.axes["top"]
        x_u, y_u = _decimate_xy(x, y_uni)
        ax_uni.plot(x_u, y_u, label="Unipolar", alpha=0.8, linewidth=0.7)
        ud = (np.append(y_uni[1:], [0.0]) - y_uni) + (y_uni - np.append([0.0], y_uni[:-1]))
        ud = ud / 2.0
        x_ud, y_ud = _decimate_xy(x, ud)
        self.app.axes["bot"].plot(x_ud, y_ud, label="unipolar_diff", alpha=0.8, linewidth=0.7)
        ax_uni.grid(True)
        try:
            self.create_legend(leg=ax_uni.get_legend_handles_labels(), canvas=self.app.ccs["top"])
        except Exception:
            traceback.print_exc()

        # Stim signal = M1 − M2 differential. Triple-extra pulses are detected
        # as the prominent local peaks on this single trace. "M" (bare) is
        # accepted as an alias for "M1" so exports that strip the index still
        # work; CS1 stays as a last-resort fallback for legacy exports missing
        # the M channels entirely.
        m = None
        m_dec = None
        m_arr = None
        for hi_name, lo_name in (("M1", "M2"), ("M", "M2")):
            try:
                hi_col = self._resolve_channel_name(data_pd, hi_name)
                lo_col = self._resolve_channel_name(data_pd, lo_name)
                m_arr = (
                    data_pd.loc[:, hi_col].values - data_pd.loc[:, lo_col].values
                )
                break
            except KeyError:
                m_arr = None
                continue
        if m_arr is not None:
            m = np.asarray(m_arr, dtype=np.float64)
            m_dec = _decimate_xy(x, m)
        else:
            try:
                cs1_col = self._resolve_channel_name(data_pd, "CS1")
                m = np.asarray(data_pd.loc[:, cs1_col].values, dtype=np.float64)
                m_dec = _decimate_xy(x, m)
            except KeyError:
                m = None
                m_dec = None

        if self.app.check_boxes["Only_Green"].get() and str(
            self.app.cont[self.app.i][0].values[self.app.j][0]
        ).upper() not in ("VERDE", "VER", "GREEN", "POSITIVE", "POS"):
            if self.app.direction >= 0:
                self.app.p_increase()
            else:
                self.app.p_decrease()
            self._fire_plot_done()
            try:
                self.app.canvas.draw_idle()
            except Exception:
                traceback.print_exc()
            return

        # V5 / reference: heavy filtfilt + Pan–Tompkins on a worker thread.
        if v5_col is not None:
            raw_ref = np.asarray(data_pd.loc[:, v5_col].values, dtype=np.float64).copy()
            stim_for_ref = None
            # When triple-extra is active, hand the stim signal to the ref
            # worker so detected R/Q peaks that fall inside any pacing pulse
            # are dropped — they would never be valid sinus references.
            if self.app.triple_active:
                # Reuse the same M − M2 (or M1 − M2) differential we plot on
                # the bot axis so the ref-ECG worker's pulse exclusion zones
                # line up exactly with the M peaks the user sees.
                if m is not None:
                    stim_for_ref = np.asarray(m, dtype=np.float64).copy()
                else:
                    try:
                        ref_meta_for_stim = None
                        try:
                            ref_meta_for_stim = str(car.loc[self.app.j, "refference_chanel"])
                        except Exception:
                            ref_meta_for_stim = None
                        stim_col = self._resolve_first_existing(
                            data_pd, ["CS1", "M4", "M3", ref_meta_for_stim, "V5"]
                        )
                        stim_for_ref = np.asarray(
                            data_pd[stim_col].values, dtype=np.float64
                        ).copy()
                    except Exception:
                        stim_for_ref = None
            self._ref_worker.post(
                _ref_ecg_compute,
                self._ref_track_apply,
                {
                    "raw_ref": raw_ref,
                    "fs": 1000.0,
                    "cycle": tick,
                    "x_full": x.copy(),
                    "m_dec": m_dec,
                    "stim_data": stim_for_ref,
                },
            )
        else:
            try:
                self.create_legend(
                    leg=self.app.axes["bot"].get_legend_handles_labels(),
                    canvas=self.app.ccs["bot"],
                )
            except Exception:
                traceback.print_exc()

        if self.app.triple_active:
            self.triple_analyzer.render_signal_axis(self.app, self.app.axes, x, y2, m)
        else:
            self.default_analyzer.render_signal_axis(self.app, self.app.axes, x, y2, m)
            self._fire_plot_done()

        try:
            self.app.canvas.draw_idle()
        except Exception:
            traceback.print_exc()

    def create_legend(self, leg, canvas, addition=None):
        def _colour_of(item) -> str:
            """Best-effort colour string for any matplotlib artist.

            ``ax.plot`` produces ``Line2D`` (has ``get_color``), ``ax.scatter``
            produces ``PathCollection`` (no ``get_color`` — only
            ``get_facecolor``). Be lenient so the legend draws cleanly for
            mixed line / marker plots (e.g. ECG trace + PT R/Q dots).
            """
            try:
                return item.get_color()
            except AttributeError:
                pass
            try:
                fc = item.get_facecolor()
            except Exception:
                return "#888888"
            try:
                if hasattr(fc, "__len__") and len(fc) and hasattr(fc[0], "__len__"):
                    fc = fc[0]
                from matplotlib.colors import to_hex
                return to_hex(fc)
            except Exception:
                return "#888888"

        colours = [_colour_of(item) for item in leg[0]]
        labels = list(leg[1])
        legend = np.vstack([colours, labels]).T
        if addition is not None:
            legend = np.concatenate([legend, np.vstack(addition)], 0)
        for idx, item in enumerate(legend):
            canvas.create_text(50, 20 + 20 * idx, text=item[1])
            canvas.create_line(100, 20 + 20 * idx, 150, 20 + 20 * idx, fill=item[0])
            canvas.create_line(0, 10 + 20 * idx, 150, 10 + 20 * idx, fill="black")
        canvas.create_line(0, 10 + 20 * len(legend), 150, 10 + 20 * len(legend), fill="black")

    def _sr_roi_bounds(self, ref_idx: int, n_samples: int, fs: float = 1000.0) -> tuple[int, int, int, int]:
        pre_n = int(round(SR_PRE_MS * fs / 1000.0))
        post_n = int(round(SR_POST_MS * fs / 1000.0))
        roi_start = max(0, int(ref_idx) - pre_n)
        roi_end = min(int(n_samples), int(ref_idx) + post_n)
        return roi_start, roi_end, pre_n, post_n

    def _pick_sr_q_index(
        self,
        reference_signal,
        stim_block_lo: int,
        stim_block_hi: int,
        egm,
    ) -> int | None:
        """Pick the SR Q anchor (last pre-stim Q, else first post-stim Q)."""
        sinus_refs = list(getattr(egm, "sinus_ref", []) or [])
        if sinus_refs:
            ref_idx = int(sinus_refs[0])
            if stim_block_lo >= 0 and stim_block_lo <= ref_idx < stim_block_hi:
                return None
            return ref_idx

        ref_bp = np.asarray(reference_signal, dtype=np.float64).ravel()
        if ref_bp.size < 50:
            return None
        r_idx = pan_tompkins_r_indices(ref_bp, fs=1000.0)
        q_local = q_indices_before_r(ref_bp, r_idx, fs=1000.0, back_ms=95.0)
        if q_local.size == 0:
            return None
        q_abs = q_local.astype(np.int64)
        if stim_block_lo >= 0:
            in_stim = (q_abs >= stim_block_lo) & (q_abs < stim_block_hi)
            q_abs = q_abs[~in_stim]
        if q_abs.size == 0:
            return None
        if stim_block_lo >= 0:
            pre = q_abs[q_abs < stim_block_lo]
            if pre.size:
                return int(pre.max())
            post = q_abs[q_abs >= stim_block_hi]
            if post.size:
                return int(post.min())
        return int(q_abs.min())

    def _delineate_sr_at_q(
        self,
        x,
        y2,
        bp_signal,
        ref_idx: int,
        cached_memory,
        forcefull,
        TH,
        ep,
    ):
        """Delineate SR inside [Q−300 ms, Q+100 ms] using stim-matching params."""
        ref_idx = int(ref_idx)
        roi_start, roi_end, pre_n, post_n = self._sr_roi_bounds(ref_idx, len(x))
        inner_lo = roi_start + STIM_SLICE_HEAD
        inner_hi = roi_end - STIM_SLICE_TAIL
        if inner_hi - inner_lo < 20:
            return None

        dont_save = False
        x_energy = y_low = y_high = None

        if (
            not forcefull
            and isinstance(cached_memory, list)
            and len(cached_memory) == 2
        ):
            start_n, end_n = int(cached_memory[0]), int(cached_memory[1])
        else:
            xx = x[inner_lo:inner_hi]
            yy = y2[inner_lo:inner_hi]
            if xx.size == 0:
                return None
            x_energy, y_low, y_high, _y_total = self._energy_compute(xx, yy, ep)
            output = find_start(
                x_energy,
                y_low,
                length=2,
                ax=None,
                operation=None,
                Th=0.15,
                alpha=TH,
                pick="earliest",
            )
            if output is not None:
                start_n = int(inner_lo + output[0])
                end_n = int(inner_lo + output[1])
            else:
                start_n, end_n = roi_start, min(roi_end, ref_idx)
                dont_save = True

        end_n = min(int(end_n), ref_idx)
        start_n = int(start_n)
        if end_n <= start_n:
            return None

        if x_energy is None:
            xx = x[inner_lo:inner_hi]
            yy = y2[inner_lo:inner_hi]
            x_energy, y_low, y_high, _y_total = self._energy_compute(xx, yy, ep)

        yy1 = y2[start_n:end_n]
        xx_w = x[start_n:end_n]
        yy_w = y2[start_n:end_n]
        ok = bool(yy1.size and yy1.max() - yy1.min() > 0.05 and not dont_save)
        defl = None
        pi_pos = pi_neg = None
        sig_pos = sig_neg = None
        if ok:
            pi_pos, sig_pos = compute_deflection_peaks(bp_signal, start_n, end_n)
            pi_neg, sig_neg = compute_deflection_peaks(-bp_signal, start_n, end_n)
            defl = int(len(pi_pos) + len(pi_neg))

        return {
            "start_n": start_n,
            "end_n": end_n,
            "ref_idx": ref_idx,
            "coarse_start": roi_start,
            "coarse_end": roi_end,
            "dont_save": bool(dont_save),
            "ok": ok,
            "defl": defl,
            "pi_pos": pi_pos,
            "sig_pos": sig_pos,
            "pi_neg": pi_neg,
            "sig_neg": sig_neg,
            "xx_w": xx_w,
            "yy_w": yy_w,
            "x_energy": x_energy,
            "y_low": y_low,
            "y_high": y_high,
            "voltage": float(yy1.max() - yy1.min()) if yy1.size else 0.0,
        }

    def _energy_compute(self, x, y, params: dict | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Pure energy compute (librosa spectrogram + convolutions).

        Works without ``app.*`` reads when ``params`` is supplied — that's how
        we make this safe to call from the worker thread, where reading mutable
        Tk state would be racy.
        """
        if params is None:
            params = {
                "n_fft": int(self.app.win_length[0]) + 25,
                "hop_length": int(self.app.hop_length[0]),
                "win_length": int(self.app.win_length[0]),
                "len_han": int(self.app.len_hann[0]),
                "low_b0": int(self.app.low_b0[0]),
                "low_b1": int(self.app.low_b1[0]),
            }
        _, _, mags = librosa.reassigned_spectrogram(
            y,
            n_fft=int(params["n_fft"]),
            hop_length=int(params["hop_length"]),
            win_length=int(params["win_length"]),
            window="hann",
            center=True,
        )
        xdb = mags
        freq = xdb.shape[0]
        len_han = max(1, int(params["len_han"]))
        window = np.ones(len_han)
        y_high = np.convolve(
            np.sum(xdb[freq // 5 :, :], 0) / max(xdb.shape[0], 1),
            window, mode="same",
        ) / np.sum(window)
        x_out = np.linspace(np.min(x), np.max(x), len(y_high))
        y_low_raw = np.sum(
            xdb[int(freq * params["low_b0"] / 500) : int(freq * params["low_b1"] / 500), :], 0
        ) / max(xdb.shape[0], 1)
        y_low = np.convolve(y_low_raw, window, mode="same") / np.sum(window)
        y_total = np.convolve(np.sum(xdb[:, :], 0) / max(xdb.shape[0], 1), window, mode="same") / np.sum(window)
        return x_out, y_low, y_high, y_total

    def energy(self, ax, x, y, legends=None):
        # Backwards-compatible: pure compute + plot if Energy checkbox is on.
        x_out, y_low, y_high, y_total = self._energy_compute(x, y)
        for key, state in self.app.check_boxes.items():
            if state.get() and key.lower() == "energy":
                if legends is None:
                    ax.plot(x_out, y_high, color="#FFf5F5")
                    ax.plot(x_out, y_low, color="#FFc5c5")
                else:
                    ax.plot(x_out, y_high, color=legends[0][0])
                    ax.plot(x_out, y_low, color=legends[1][0])
        return [x_out, y_low, y_high, y_total]


class AdvancedPlotPresenter(PlotPresenter):
    """Child presenter reserved for future protocol extensions."""

    def before_plot(self):
        return None

    def after_plot(self):
        return None

    def plot(self):
        self.before_plot()
        out = super().plot()
        self.after_plot()
        return out
