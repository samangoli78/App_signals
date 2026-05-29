"""ECG fiducials: Pan–Tompkins-style R detection and Q nadir before each R.

Surface ECG (e.g. V5) is not a perfect textbook trace, but the classical
Pan–Tompkins pipeline (band-limit → derivative → squaring → moving-window
integration → adaptive peak picking with a refractory period) is far more
stable than ``scipy.signal.find_peaks`` on the rectified signal for locating
beat onsets for timing overlays.

Q wave: taken as the **minimum** sample in a short backward search window
ending at each R index on the band-passed trace (first significant downward
deflection before the R upslope in many leads). If no clear minimum exists,
the R index itself is kept as a fallback so downstream code still has a beat
time.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sps


def _bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 2) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < max(32, order * 8):
        return x.copy()
    nyq = 0.5 * fs
    b, a = sps.butter(order, [lo / nyq, hi / nyq], btype="band")
    return sps.filtfilt(b, a, x).astype(np.float64)


def pan_tompkins_r_indices(ecg: np.ndarray, fs: float = 1000.0) -> np.ndarray:
    """Return sample indices of R-wave candidates (integer array, may be empty)."""
    x = np.asarray(ecg, dtype=np.float64).ravel()
    n = x.size
    if n < int(0.5 * fs):
        return np.array([], dtype=np.int64)

    # Stage 1: band-pass emphasises QRS (~5–18 Hz at 1 kHz).
    y = _bandpass(x, fs, 5.0, 18.0, order=2)

    # Stage 2: five-point derivative (Hamilton / PT style).
    d = np.zeros_like(y)
    d[2:-2] = (
        2 * y[4:]
        + y[3:-1]
        - y[1:-3]
        - 2 * y[:-4]
    ) / 8.0

    # Stage 3: squaring.
    s = d * d

    # Stage 4: moving-window integration (~150 ms).
    win = max(3, int(round(0.150 * fs)))
    k = np.ones(win, dtype=np.float64) / win
    m = np.convolve(s, k, mode="same")

    # Stage 5: peak picking with refractory period (~200 ms).
    refractory = max(1, int(round(0.200 * fs)))
    med = float(np.median(m))
    mad = float(np.median(np.abs(m - med))) + 1e-12
    thr = med + 4.0 * mad
    thr = max(thr, 0.35 * float(np.max(m)))

    peaks, _ = sps.find_peaks(m, height=thr, distance=refractory)
    return np.asarray(peaks, dtype=np.int64)


def q_indices_before_r(
    bandpassed_ecg: np.ndarray,
    r_indices: np.ndarray,
    fs: float = 1000.0,
    back_ms: float = 90.0,
) -> np.ndarray:
    """For each R, Q = argmin of ``bandpassed_ecg`` in ``(R - back_ms, R]``."""
    x = np.asarray(bandpassed_ecg, dtype=np.float64).ravel()
    r_indices = np.asarray(r_indices, dtype=np.int64).ravel()
    if r_indices.size == 0 or x.size == 0:
        return np.array([], dtype=np.int64)
    back = max(3, int(round(back_ms * fs / 1000.0)))
    out: list[int] = []
    for r in r_indices:
        r = int(r)
        if not (0 <= r < x.size):
            continue
        a = max(0, r - back)
        seg = x[a : r + 1]
        if seg.size < 2:
            out.append(r)
            continue
        q = a + int(np.argmin(seg))
        out.append(q)
    return np.unique(np.asarray(out, dtype=np.int64))


def r_q_markers_for_display(
    raw_ecg: np.ndarray,
    fs: float = 1000.0,
    display_band: tuple[float, float] = (5.0, 180.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(v_display, r_idx, q_idx)`` for overlay on the reference trace.

    ``v_display`` uses the same wide band-pass as the legacy V5 plot so the
    waveform shape users already recognise is preserved; R/Q indices are
    derived from the narrow-band PT chain for robust timing.
    """
    raw = np.asarray(raw_ecg, dtype=np.float64).ravel()
    if raw.size == 0:
        return raw.copy(), np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    lo, hi = display_band
    v_disp = _bandpass(raw, fs, lo, hi, order=2)
    y_pt = _bandpass(raw, fs, 5.0, 18.0, order=2)
    r_idx = pan_tompkins_r_indices(raw, fs=fs)
    if r_idx.size == 0:
        # Fallback: one broad peak per ~300 ms on |display| trace.
        env = np.abs(v_disp)
        r_idx, _ = sps.find_peaks(
            env, distance=max(1, int(0.30 * fs)), prominence=0.15 * (np.max(env) + 1e-12)
        )
        r_idx = np.asarray(r_idx, dtype=np.int64)
    q_idx = q_indices_before_r(v_disp, r_idx, fs=fs, back_ms=95.0)
    return v_disp, r_idx, q_idx


def detect_stim_windows(
    stim_data: np.ndarray,
    fs: float = 1000.0,
    distance_samples: int = 200,
    height_frac: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Same logic as ``Triple_Extra.compute_windows`` for stim windowing.

    Returns ``(starts, ends)`` (both int64). Each window is
    ``[start, end)`` on the input sample grid. Empty arrays if no clean stim
    peak is detected.
    """
    s = np.asarray(stim_data, dtype=np.float64).ravel()
    s = np.abs(s)
    if s.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    mx = float(np.max(s)) if s.size else 0.0
    if mx <= 0.0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    P, _ = sps.find_peaks(s, distance=int(distance_samples), height=height_frac * mx)
    if P.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    if P.size > 1:
        diffs = np.diff(P).astype(np.int64)
        durations = np.concatenate([diffs, [int(diffs[0]) + 28]])
    else:
        durations = np.array([28], dtype=np.int64)
    starts = P.astype(np.int64)
    ends = starts + durations
    return starts, ends


def stim_pulse_exclusion_spans_signed_m(
    m_signed: np.ndarray,
    fs: float = 1000.0,
    distance_samples: int = 200,
    height_frac: float = 0.3,
    half_width_ms: float = 50.0,
    max_pulses: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Narrow ``[start, end)`` bands around each pacing spike on signed M.

    Matches ``Triple_Extra.compute_windows`` for positive-peak detection and
    the ±``half_width_ms`` exclusion used for V5 Q filtering — not the long
    inter-peak intervals from :func:`detect_stim_windows`.
    """
    s = np.asarray(m_signed, dtype=np.float64).ravel()
    n = int(s.size)
    if n == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    smax = float(np.max(s))
    if smax <= 0.0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    P, _ = sps.find_peaks(s, distance=int(distance_samples), height=height_frac * smax)
    if P.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    if P.size > max_pulses:
        P = P[:max_pulses].copy()
    half = max(1, int(round(half_width_ms * fs / 1000.0)))
    starts = np.maximum(0, P.astype(np.int64) - half)
    ends = np.minimum(n, P.astype(np.int64) + half + 1)
    return starts, ends


def filter_indices_outside_windows(
    indices: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> np.ndarray:
    """Return sample indices that fall **outside** every ``[start, end)`` window."""
    indices = np.asarray(indices, dtype=np.int64).ravel()
    starts = np.asarray(starts, dtype=np.int64).ravel()
    ends = np.asarray(ends, dtype=np.int64).ravel()
    if indices.size == 0 or starts.size == 0:
        return indices.copy()
    keep = np.ones(indices.size, dtype=bool)
    for s, e in zip(starts.tolist(), ends.tolist()):
        keep &= ~((indices >= s) & (indices < e))
    return indices[keep]


def sinus_reference_indices_pt(
    reference_bandpassed: np.ndarray,
    stim_start0: int,
    tail_start: int,
    fs: float = 1000.0,
) -> np.ndarray:
    """Beat times (sample indices) for sinus gating: Q before R, inside pre/post segments.

    Replaces naive ``find_peaks`` on ``|reference|`` for the pre-stim and
    post-stim segments. Falls back to rectified-peak detection if PT finds
    nothing usable.
    """
    ref = np.asarray(reference_bandpassed, dtype=np.float64).ravel()
    n = ref.size
    if n == 0:
        return np.array([], dtype=np.int64)

    r_all = pan_tompkins_r_indices(ref, fs=fs)
    q_all = q_indices_before_r(ref, r_all, fs=fs, back_ms=95.0)

    pre_end = max(0, int(stim_start0) - 20)
    post_beg = max(0, min(int(tail_start), n))

    def _in_range(idxs: np.ndarray, lo: int, hi: int) -> np.ndarray:
        idxs = idxs[(idxs >= lo) & (idxs < hi)]
        return np.unique(np.asarray(idxs, dtype=np.int64))

    q_pre = _in_range(q_all, 0, pre_end)
    q_post = _in_range(q_all, post_beg, n)

    if q_pre.size == 0 and pre_end > 50:
        abs_pre = np.abs(ref[:pre_end])
        mx = float(np.max(abs_pre)) if abs_pre.size else 0.0
        if mx > 0:
            p1, _ = sps.find_peaks(abs_pre, distance=500, height=0.5 * mx)
            q_pre = np.asarray(p1, dtype=np.int64)

    if q_post.size == 0 and n - post_beg > 50:
        abs_post = np.abs(ref[post_beg:])
        mx = float(np.max(abs_post)) if abs_post.size else 0.0
        if mx > 0:
            pp, _ = sps.find_peaks(abs_post, distance=500, height=0.5 * mx)
            q_post = np.asarray(pp, dtype=np.int64) + post_beg

    return np.sort(np.unique(np.concatenate([q_pre, q_post]))) if (q_pre.size or q_post.size) else np.array([], dtype=np.int64)
