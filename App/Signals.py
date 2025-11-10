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

def find_start(x, y, length=7, ax=None, operation="min", Th=0.2, alpha=0.5):
    """
    Detect the start–end region of activity in a 1-D signal y(x).
    Organized in conceptual 'levels' for clarity.
    """

    # ---------------- LEVEL 1: Pre-processing ----------------
    # Smooth signal with a small Gaussian kernel to suppress noise
    y_smooth = gaussian_filter1d(y, sigma=2)

    # Compute threshold adaptively (Otsu method, user-defined)
    Th = otsu_threshold(y_smooth, alpha=alpha)

    # ---------------- LEVEL 2: Interpolation ----------------
    # Resample to a dense uniform grid (~0.001 step) so we can detect edges precisely
    step = 0.001
    N = len(y_smooth)
    xs = np.linspace(x.min(), x.max(), N)
    cs = interp1d(x, y_smooth, kind="linear", bounds_error=False, fill_value="extrapolate")
    out = cs(xs)
    # >>> Added fixed zero padding <<<
    out = np.pad(out, (10, 10), mode="constant", constant_values=0)
    # ---------------- LEVEL 3: Thresholding ----------------
    # Create a binary mask for regions above threshold
    out = np.array([1 if val > Th else 0 for val in out], dtype=bool)

    # ---------------- LEVEL 4: Morphological Closing ----------------
    # Merge small gaps and noise with morphological closing
    out = binary_closing(out, structure=np.ones(15, dtype=bool))

    # ---------------- LEVEL 5: Segment Detection ----------------
    # Label contiguous True regions and pick the longest one
    labels, num = label(out)
    if num == 0:
        return None

    sizes = np.array([(labels == k).sum() for k in range(1, num + 1)])
    kmax = 1 + int(np.argmax(sizes))
    idx = np.flatnonzero(labels == kmax)

    if len(idx) < 5:  # too short to be meaningful
        return None

    start_idx, end_idx = int(idx[0]), int(idx[-1])
    start_idx -= 10
    end_idx   -= 10

    # ---------------- LEVEL 6: Visualization (optional) ----------------
    if ax is not None:
        ax.plot(xs, y_smooth)
        #ax.axhline(Th, color='r', ls='--', label='threshold')
        ax.fill_between(xs[start_idx:end_idx], 0, y_smooth[start_idx:end_idx],
                        color='orange', alpha=0.3)
        ax.legend()

    # ---------------- LEVEL 7: Output ----------------
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