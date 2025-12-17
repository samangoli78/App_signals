from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from App.App import App
from scipy.signal import butter,filtfilt,get_window,find_peaks,argrelextrema,resample

import numpy as np
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d,CubicSpline
from sklearn.preprocessing import minmax_scale
from scipy.ndimage import binary_closing,gaussian_filter1d, label
    
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
    # Pad the signal
    padded_signal = np.pad(signal, (pad_left, pad_right), mode='constant', constant_values=0)
    
    return padded_signal
def otsu_threshold(energy_signal,alpha=0.5):
    # Get the original min and max values of the input signal
    signal_min = np.min(energy_signal)
    signal_max = np.max(energy_signal)
    
    # Normalize energy_signal to range [0, 255] for Otsu's method
    energy_signal_normalized = (energy_signal - signal_min) / (signal_max - signal_min) * 255
    energy_signal_normalized = energy_signal_normalized.astype(np.uint8)  # Convert to uint8 for OpenCV
    
    # Apply Otsu's thresholding
    #TH_normalized, _ = cv2.threshold(energy_signal_normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    TH_normalized=custom_threshold_1d(energy_signal_normalized,alpha=alpha)
    # Scale the Otsu threshold back to the original signal range
    TH_original_scale = TH_normalized * (signal_max - signal_min) / 255 + signal_min
    
    return TH_original_scale

def butter_lowpass_filter(data, cutoff=60, fs=1000, order=None):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low')
    y = filtfilt(b, a, data)
    return y


def butter_highpass_filter(data, cutoff=60, fs=1000, order=None):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='high')
    y = filtfilt(b, a, data)
    return y


def butter_notch_filter(data, f=60,Q=30, fs=1000, order=None):
    nyq = 0.5 * fs
    B1 = (f - (f/Q)/2) / nyq
    B2 = (f + (f/Q)/2) / nyq
    # Get the filter coefficients 
    b, a = butter(order, [B1,B2], btype='bandstop')
    y = filtfilt(b, a, data)
    return y
        

def butter_bandpass_filter(data, cutoff=60, fs=1000, order=None):
    nyq = 0.5 * fs
    B1 = cutoff[0] / nyq
    B2 = cutoff[1] / nyq
    # Get the filter coefficients 
    b, a = butter(order, [B1,B2], btype='bandpass')
    y = filtfilt(b, a, data)
    return y

def derivative(y, step=0.001):
    y = np.asarray(y, dtype=np.float64)
    dy = np.diff(y) / step
    return dy

import numpy as np
from scipy.ndimage import gaussian_filter1d, binary_closing, label
from scipy.interpolate import interp1d

import numpy as np
from scipy.ndimage import gaussian_filter1d, binary_closing, label
from scipy.interpolate import interp1d

def find_start(app:App, x_woi, y_woi, length=7, ax=None, operation="min",
               Th=0.2, alpha=0.5, close_width=15, pad=10,
               top_k_longest=3, min_seg_len=5):
    """
    - Detect chunks above threshold (after closing)
    - For each chunk, compute mean power of low/high/total
    - Keep the top_k_longest chunks by length
    - Select the chunk with the highest LOW mean power among them
    - Return [start_idx, end_idx] in xs-grid indices
    """
    x_E, y_low, y_high, y_total = app.Energy(ax, x_woi, y_woi)

    # Smooth + adaptive threshold
    y_smooth = gaussian_filter1d(y_total, sigma=2)
    Th = otsu_threshold(y_smooth, alpha=alpha)

    # Interpolate onto uniform grid
    N = len(y_smooth)
    xs = np.linspace(np.min(x_E), np.max(x_E), N)

    def _interp(arr):
        f = interp1d(x_E, arr, kind="linear", bounds_error=False, fill_value="extrapolate")
        return f(xs)

    total_i = _interp(y_smooth)   # use smoothed total for segmentation
    low_i   = _interp(y_low)
    high_i  = _interp(y_high)

    # Pad for morphology (keep stats arrays edge-padded to avoid artificial zeros)
    total_p = np.pad(total_i, (pad, pad), mode="constant", constant_values=0.0)
    low_p   = np.pad(low_i,   (pad, pad), mode="edge")
    high_p  = np.pad(high_i,  (pad, pad), mode="edge")

    # Threshold + closing
    binary = total_p > Th
    closed = binary_closing(binary, structure=np.ones(close_width, dtype=bool))

    labels, num = label(closed)
    if num == 0:
        return None

    segments = []  # each: dict(s,e,len,p_low,p_high,p_total)

    for k in range(1, num + 1):
        idx_p = np.flatnonzero(labels == k)
        if idx_p.size < min_seg_len:
            continue

        # unpad indices (on xs grid)
        s = int(idx_p[0]) - pad
        e = int(idx_p[-1]) - pad

        # clip to valid range
        s0 = max(s, 0)
        e0 = min(e, len(y_woi) - 1)
        seg_len = e0 - s0 + 1
        if seg_len < min_seg_len:
            continue

        # Mean power = mean(signal^2)
        low_seg   = low_i[s0:e0 + 1]
        high_seg  = high_i[s0:e0 + 1]
        total_seg = total_i[s0:e0 + 1]

        p_low   = float(np.mean(low_seg))
        p_high  = float(np.mean(high_seg))
        p_total = float(np.mean(total_seg))

        segments.append({
            "s": s, "e": e, "len": seg_len,
            "p_low": p_low, "p_high": p_high, "p_total": p_total
        })

    if not segments:
        return None

    # 1) take the longest top_k_longest (tie-break with p_low so it's deterministic)
    new_segments=[]
    for seg in segments:
        start_idx_temp=seg["s"]
        end_idx_temp=seg["e"]
        temp_sig=y_woi[start_idx_temp:end_idx_temp+1]
        temp_p_p_v=temp_sig.max()-temp_sig.min()
        if temp_p_p_v>=0.1:
            new_segments.append(seg)
    segments=new_segments
    if not segments:
        return None
    segments.sort(key=lambda d: (d["len"], d["p_low"]), reverse=True)
    candidates = segments[:min(len(segments), top_k_longest)]

    # 2) among them, choose highest LOW energy (mean power of low)
    best = max(candidates, key=lambda d: d["p_low"])
    print(best["p_low"])
    start_idx, end_idx = best["s"], best["e"]

    if ax is not None:
        ax.plot(xs, total_i, label="y_smooth(interp)")
        ax.axhline(Th, ls="--", label=f"Th={Th:.3g}")
        a = max(start_idx, 0)
        b = min(end_idx, N - 1)
        ax.fill_between(xs[a:b+1], 0, total_i[a:b+1], alpha=0.3,
                        label=f"len={best['len']}  p_low={best['p_low']:.3g}")
        ax.legend()

    return [start_idx, end_idx]



def custom_threshold_1d(signal,alpha=0.5):
    # Calculate the histogram of the signal
    hist, bin_edges = np.histogram(signal, bins=256, range=(np.min(signal), np.max(signal)))
    
    # Normalize the histogram to get probabilities
    prob = hist / np.sum(hist)
    
    # Cumulative sum of probabilities
    cumulative_sum = np.cumsum(prob)
    
    # Cumulative mean
    cumulative_mean = np.cumsum(prob * bin_edges[:-1])
    
    # Total mean of the signal
    total_mean = cumulative_mean[-1]
    
    max_ = 0
    best_threshold = 0
    
    # Iterate through all possible thresholds
    for t in range(1, 256):
        # Class probabilities
        w0 = cumulative_sum[t - 1]
        w1 = 1 - w0
        
        if w0 == 0 or w1 == 0:
            continue
        
        # Class means
        m0 = cumulative_mean[t - 1] / w0
        m1 = (cumulative_mean[-1] - cumulative_mean[t - 1]) / w1
        
        # Class standard deviations
        #sigma0 = np.std(signal[:t]) if w0 > 0 else 0
        #sigma1 = np.std(signal[t:]) if w1 > 0 else 0
        
        # Modified between-class variance formula with standard deviation adjustment
        variance = min(max((w0-alpha+0.5),0),1) * min(max((w1+(alpha-0.5)),0),1) * (m0 - m1) ** 2 
        
        # Update the maximum mean difference and threshold
        if variance > max_:
            max_ = variance
            best_threshold = t
    
    return best_threshold