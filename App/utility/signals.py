from scipy.signal import butter, filtfilt, get_window, find_peaks, argrelextrema, resample

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, CubicSpline
from sklearn.preprocessing import minmax_scale
from scipy.ndimage import binary_closing

import cv2


def zero_pad_signal(signal, pad_left=0, pad_right=0):
    """
    Zero-pads the signal on the left and right (tails).

    Parameters:
    - signal: The original signal (1D numpy array).
    - pad_left: Number of zeros to add at the start (left).
    - pad_right: Number of zeros to add at the end (right).

    Returns:
    - The padded signal.
    """
    padded_signal = np.pad(signal, (pad_left, pad_right), mode="constant", constant_values=0)
    return padded_signal


def otsu_threshold(energy_signal, alpha=0.5):
    signal_min = np.min(energy_signal)
    signal_max = np.max(energy_signal)
    energy_signal_normalized = (energy_signal - signal_min) / (signal_max - signal_min) * 255
    energy_signal_normalized = energy_signal_normalized.astype(np.uint8)
    th_normalized = custom_threshold_1d(energy_signal_normalized, alpha=alpha)
    th_original_scale = th_normalized * (signal_max - signal_min) / 255 + signal_min
    return th_original_scale


def butter_lowpass_filter(data, cutoff=60, fs=1000, order=None):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low")
    y = filtfilt(b, a, data)
    return y


def butter_highpass_filter(data, cutoff=60, fs=1000, order=None):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="high")
    y = filtfilt(b, a, data)
    return y


def butter_notch_filter(data, f=60, Q=30, fs=1000, order=None):
    nyq = 0.5 * fs
    b1 = (f - (f / Q) / 2) / nyq
    b2 = (f + (f / Q) / 2) / nyq
    b, a = butter(order, [b1, b2], btype="bandstop")
    y = filtfilt(b, a, data)
    return y


def butter_bandpass_filter(data, cutoff=60, fs=1000, order=None):
    nyq = 0.5 * fs
    b1 = cutoff[0] / nyq
    b2 = cutoff[1] / nyq
    b, a = butter(order, [b1, b2], btype="bandpass")
    y = filtfilt(b, a, data)
    return y


def derivative(y):
    yt = np.append(y, 0)
    dy = np.array([(yt[i] - yt[i - 1]) / 0.001 for i in np.arange(1, len(yt))])
    return dy


def apply_closing(binary_signal, structure_size=5):
    structure = np.ones(structure_size)
    return binary_closing(binary_signal, structure=structure)


def find_start(x, y, length=7, ax=None, operation="min", Th=0.2, alpha=0.5):
    y_paded = zero_pad_signal(y, pad_left=10, pad_right=10)
    Th = otsu_threshold(y_paded, alpha=alpha)
    out = np.array(y)
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < 2 or out.size < 2:
        return None
    x_lo = float(x.min())
    x_hi = float(x.max())
    if not np.isfinite(x_lo) or not np.isfinite(x_hi) or x_hi <= x_lo:
        return None
    # ``bounds_error=False`` + endpoint fill_values protect against the
    # microscopic float drift in ``np.arange`` (last sample can land a few
    # ULPs above ``x_hi`` and trip interp1d's strict bounds check).
    cs = interp1d(x, out, bounds_error=False, fill_value=(out[0], out[-1]))
    xs = np.arange(x_lo, x_hi, 0.001)
    # Belt-and-braces: also clip in case xs got a stray value past x_hi.
    if xs.size and xs[-1] > x_hi:
        xs = np.clip(xs, x_lo, x_hi)
    out = cs(xs)
    out_paded = zero_pad_signal(out, pad_left=10, pad_right=10)
    out = np.array([1 if ii > Th else 0 for ii in out_paded])
    out = apply_closing(out, 15)
    out1 = out
    output = []
    start = 0
    end = -1
    in_1 = False
    for ii, val in enumerate(out1):
        if val == 1 and in_1 is False:
            start = ii
            in_1 = True
        elif val == 0 and in_1 is True:
            end = ii
            in_1 = False
            if end - start > 5:
                output.append([start - 10, end - 10, end - start])
    index = np.argsort(np.array([val[2] for val in output]))
    out1 = []
    for ii in index:
        out1.append([output[ii][jj] for jj in range(2)])
    if not isinstance(ax, type(None)):
        pass
    try:
        return out1[-1]
    except Exception:
        return None


def custom_threshold_1d(signal, alpha=0.5):
    hist, bin_edges = np.histogram(signal, bins=256, range=(np.min(signal), np.max(signal)))
    prob = hist / np.sum(hist)
    cumulative_sum = np.cumsum(prob)
    cumulative_mean = np.cumsum(prob * bin_edges[:-1])
    max_ = 0
    best_threshold = 0
    for t in range(1, 256):
        w0 = cumulative_sum[t - 1]
        w1 = 1 - w0
        if w0 == 0 or w1 == 0:
            continue
        m0 = cumulative_mean[t - 1] / w0
        m1 = (cumulative_mean[-1] - cumulative_mean[t - 1]) / w1
        variance = min(max((w0 - alpha + 0.5), 0), 1) * min(max((w1 + (alpha - 0.5)), 0), 1) * (m0 - m1) ** 2
        if variance > max_:
            max_ = variance
            best_threshold = t
    return best_threshold
