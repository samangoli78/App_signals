from scipy.signal import find_peaks
import numpy as np


class Triple_Extra:
    def __init__(self, t, EGM, T=2.5, fs=1000, order=2):
        self.signal = EGM
        self.t = t
        self.T = T
        self.fs = fs
        self.nyq = 0.5 * self.fs
        self.order = order
        self.N = int(self.T * self.fs)

    # ---------------------------------------------------------------- compute
    def compute_windows(self, stimulation, reference, margin=0):
        """Pure compute: detect stim/sinus windows and the markers we want to draw.

        Returns a dict with both the window descriptors AND the scatter
        coordinates the caller can later push to matplotlib on the Tk thread.
        No plotting happens here, so this method is safe to run on a
        background worker thread.

        Detection rules (kept deliberately simple):
          * Stim pulses are the prominent local peaks on |M1 − M2|. We use
            ``find_peaks(distance=100)`` — peaks must be at least 100 ms
            apart (±100 ms refractory) — and we ALWAYS keep the three
            highest peaks, sorted in time order. No Q-overlap, no SR
            classification: those filters were causing S1 to drop out
            whenever it sat near a V5 Q. The user's protocol is "find three
            peaks on M1 − M2, period".
          * Sinus rhythm window is anchored on a single Q (Pan–Tompkins on
            the reference channel):
              start = Q − 600 ms, end = Q + 200 ms (total 800 ms).
            The chosen Q is the **last** valid one before the first stim
            window; if none exists, the **first** valid one after the last
            stim window.
        """
        stimulation = np.asarray(stimulation, dtype=np.float64).ravel()
        if stimulation.size == 0:
            return self._empty_windows()

        # |M1 − M2| — polarity-agnostic so detection works regardless of
        # which way the bipole is oriented.
        abs_stim = np.abs(stimulation)
        amax = float(np.max(abs_stim))
        if amax <= 0.0:
            return self._empty_windows()
        # distance=100 → ±100 ms refractory; low height floor (0.15 * peak)
        # so a smaller S1 isn't lost when later pulses are much taller.
        P, stim_dic = find_peaks(
            abs_stim, distance=100, height=0.15 * amax
        )
        if P.size == 0:
            return self._empty_windows()
        peak_heights = np.asarray(stim_dic.get("peak_heights", np.zeros_like(P)), dtype=np.float64)

        # Guarantee three: take the three highest by amplitude and re-sort
        # in time order. If find_peaks returned fewer than three (sparse
        # data), we keep whatever we have.
        if P.size > 3:
            top3 = np.argsort(peak_heights)[-3:]
            order = np.argsort(P[top3])
            keep = top3[order]
            P = P[keep].astype(np.int64)
            peak_heights = peak_heights[keep]
        else:
            order = np.argsort(P)
            P = P[order].astype(np.int64)
            peak_heights = peak_heights[order]

        from .utility.signals_ecg import (
            sinus_reference_indices_pt,
            filter_indices_outside_windows,
        )

        ref_bp = np.asarray(reference, dtype=np.float64).ravel()
        n_ref = int(ref_bp.size)

        def _window_durations(peak_indices: np.ndarray) -> np.ndarray:
            if peak_indices.size == 0:
                return np.array([], dtype=np.int64)
            if peak_indices.size == 1:
                return np.array([28], dtype=np.int64)
            dd = np.diff(peak_indices).astype(np.int64)
            return np.concatenate([dd, [int(dd[0]) + 28]])

        diffs = _window_durations(P)

        self.stim_start = P
        self.stim_ref = P
        self.stim_duration = list(diffs)

        scatter_stim_x = (P * 0.001).astype(np.float64)
        scatter_stim_y = peak_heights.astype(np.float64)

        # Narrow exclusion zones (±100 ms) used only to keep the sinus Q
        # away from pacing artefacts — never to demote stim candidates.
        pulse_half = max(1, int(round(0.100 * float(self.fs))))
        stim_window_starts = np.maximum(0, P.astype(np.int64) - pulse_half)
        stim_window_ends = np.minimum(n_ref, P.astype(np.int64) + pulse_half + 1)
        self.stim_window_starts = stim_window_starts
        self.stim_window_ends = stim_window_ends

        refference = np.abs(reference)
        ref_max = float(np.max(refference)) if refference.size else 0.0

        # When stim is gone (all candidates were really SR responses), we still
        # use PT to pick the single SR. The pre/post split is then degenerate
        # so we just take every Q and pick by the same rule.
        if P.size:
            first_stim_start = int(P[0])
            tail_start = int(P[-1]) + int(diffs[-1])
        else:
            first_stim_start = int(ref_bp.size)
            tail_start = 0

        P1_combined = sinus_reference_indices_pt(
            ref_bp,
            int(first_stim_start),
            int(tail_start),
            fs=float(self.fs),
        )

        # Hard-drop any Q peak that lies inside ANY stim pulse window so the
        # sinus reference never falls on a pacing artefact.
        if stim_window_starts.size:
            P1_combined = filter_indices_outside_windows(
                P1_combined, stim_window_starts, stim_window_ends
            )

        # Reduce to ONE sinus reference per electrode point:
        #   - prefer the **last** Q strictly before the first stim window;
        #   - if there is none before, take the **first** Q strictly after
        #     the last stim window.
        # A 50 ms guard band on both sides of the stim block protects against
        # filter-ringing artefacts that look like Q peaks right next to the
        # pacing pulses.
        guard = int(0.050 * self.fs)
        first_stim_guarded = (int(P[0]) - guard) if P.size else int(ref_bp.size)
        last_stim_guarded = (int(P[-1]) + int(diffs[-1]) + guard) if P.size else 0
        if P1_combined.size:
            pre = P1_combined[P1_combined < first_stim_guarded]
            if pre.size:
                P1_combined = np.array([int(pre.max())], dtype=np.int64)
            else:
                post = P1_combined[P1_combined >= last_stim_guarded]
                if post.size:
                    P1_combined = np.array([int(post.min())], dtype=np.int64)
                else:
                    P1_combined = np.array([], dtype=np.int64)

        if P1_combined.size:
            scatter_pre_mask = P1_combined < max(0, first_stim_start - 20)
            scatter_post_mask = P1_combined >= tail_start
            P1 = P1_combined[scatter_pre_mask]
            PP = P1_combined[scatter_post_mask]
            scatter_pre_x = (P1 * 0.001).astype(np.float64) if P1.size else np.array([], dtype=np.float64)
            scatter_pre_y = ref_bp[P1].astype(np.float64) if P1.size else np.array([], dtype=np.float64)
            scatter_post_x = (PP * 0.001).astype(np.float64) if PP.size else np.array([], dtype=np.float64)
            scatter_post_y = ref_bp[PP].astype(np.float64) if PP.size else np.array([], dtype=np.float64)
        else:
            P1 = np.array([], dtype=np.int64)
            PP = np.array([], dtype=np.int64)
            scatter_pre_x = scatter_pre_y = np.array([], dtype=np.float64)
            scatter_post_x = scatter_post_y = np.array([], dtype=np.float64)

        # Legacy fallback if PT finds nothing usable on this trace.
        if not P1_combined.size and ref_bp.size > 100:
            ref_pre = refference[: max(0, first_stim_start - 20)] if first_stim_start > 20 else np.array([])
            if ref_pre.size:
                P1, dic_pre = find_peaks(ref_pre, distance=500, height=0.5 * ref_max)
                scatter_pre_x = (P1 * 0.001).astype(np.float64)
                scatter_pre_y = np.asarray(dic_pre.get("peak_heights", []), dtype=np.float64)
            else:
                P1 = np.array([], dtype=np.int64)
                scatter_pre_x = np.array([], dtype=np.float64)
                scatter_pre_y = np.array([], dtype=np.float64)
            ref_post = refference[tail_start:] if tail_start < refference.size else np.array([])
            if ref_post.size:
                PP, dic_post = find_peaks(ref_post, distance=500, height=0.5 * ref_max)
                PP = PP + tail_start
                scatter_post_x = (PP * 0.001).astype(np.float64)
                scatter_post_y = np.asarray(dic_post.get("peak_heights", []), dtype=np.float64)
            else:
                PP = np.array([], dtype=np.int64)
                scatter_post_x = np.array([], dtype=np.float64)
                scatter_post_y = np.array([], dtype=np.float64)
            P1_combined = np.hstack([P1, PP]) if (P1.size or PP.size) else np.array([], dtype=np.int64)
            if stim_window_starts.size:
                P1_combined = filter_indices_outside_windows(
                    P1_combined, stim_window_starts, stim_window_ends
                )
            if P1_combined.size:
                pre = P1_combined[P1_combined < first_stim_guarded]
                if pre.size:
                    P1_combined = np.array([int(pre.max())], dtype=np.int64)
                else:
                    post = P1_combined[P1_combined >= last_stim_guarded]
                    P1_combined = (
                        np.array([int(post.min())], dtype=np.int64)
                        if post.size
                        else np.array([], dtype=np.int64)
                    )

        self.sinus_ref = P1_combined

        # SR window: strictly [Q − 600 ms, Q + 200 ms]. The onset/offset
        # refinement (find_start) still narrows the analysed segment inside
        # this window downstream.
        pre_ms = 600
        post_ms = 200
        duration = int((pre_ms + post_ms) * self.fs / 1000)
        pre_n = int(pre_ms * self.fs / 1000)
        sinus_start = [max(0, int(ii) - pre_n) for ii in P1_combined]
        self.sinus_start = sinus_start
        self.sinus_duration = [duration] * len(self.sinus_start)
        _ = margin  # accepted for API stability; unused in the new SR scheme.

        return {
            "stim_start": list(self.stim_start),
            "stim_ref": list(self.stim_ref),
            "stim_duration": list(self.stim_duration),
            "sinus_start": list(self.sinus_start),
            "sinus_ref": list(self.sinus_ref),
            "sinus_duration": list(self.sinus_duration),
            "scatter": [
                (scatter_stim_x, scatter_stim_y),
                (scatter_pre_x, scatter_pre_y),
                (scatter_post_x, scatter_post_y),
            ],
        }

    def _empty_windows(self) -> dict:
        self.stim_start = np.array([], dtype=np.int64)
        self.stim_ref = np.array([], dtype=np.int64)
        self.stim_duration = []
        self.stim_window_starts = np.array([], dtype=np.int64)
        self.stim_window_ends = np.array([], dtype=np.int64)
        self.sinus_ref = np.array([], dtype=np.int64)
        self.sinus_start = []
        self.sinus_duration = []
        return {
            "stim_start": [], "stim_ref": [], "stim_duration": [],
            "sinus_start": [], "sinus_ref": [], "sinus_duration": [],
            "scatter": [],
        }

    # ---------------------------------------------------------------- drawing
    def find_windows(self, ax, stiulation, refference, margin=0):
        """Backwards-compatible wrapper: compute + draw stim scatter markers only.

        Pre/post-stim Q markers are now drawn separately by the presenter on
        top of the bandpassed reference trace (see ``signals_ecg`` /
        ``_ref_track_apply``), so this wrapper no longer duplicates them here.
        """
        info = self.compute_windows(stimulation=stiulation, reference=refference, margin=margin)
        if ax is not None:
            scatter = info.get("scatter", [])
            if scatter:
                xs, ys = scatter[0]
                if xs.size and ys.size:
                    ax.plot(xs, ys, "x", color="#FFCC33")
        return info

    def median_convolution(self, y, n=3):
        f = np.zeros(len(y))
        for i, _ in enumerate(y):
            if i // n == 0:
                f[i] = np.median(y[i - i % n : i])
            else:
                f[i] = np.median(y[i - n : i])
        f = np.nan_to_num(f)
        return f
