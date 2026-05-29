"""General signal-processing utilities used across the App.

Filters, threshold helpers, ECG fiducial detection (Pan–Tompkins) and other
DSP building blocks that aren't tied to any particular UI panel or analysis
strategy. Import everything you need directly from this package, e.g.::

    from App.utility import butter_bandpass_filter, find_start
    from App.utility import r_q_markers_for_display
"""
from .signals import (
    apply_closing,
    butter_bandpass_filter,
    butter_highpass_filter,
    butter_lowpass_filter,
    butter_notch_filter,
    custom_threshold_1d,
    derivative,
    find_start,
    otsu_threshold,
    zero_pad_signal,
)
from .signals_ecg import (
    detect_stim_windows,
    filter_indices_outside_windows,
    pan_tompkins_r_indices,
    q_indices_before_r,
    r_q_markers_for_display,
    sinus_reference_indices_pt,
    stim_pulse_exclusion_spans_signed_m,
)

__all__ = [
    "apply_closing",
    "butter_bandpass_filter",
    "butter_highpass_filter",
    "butter_lowpass_filter",
    "butter_notch_filter",
    "custom_threshold_1d",
    "derivative",
    "detect_stim_windows",
    "filter_indices_outside_windows",
    "find_start",
    "otsu_threshold",
    "pan_tompkins_r_indices",
    "q_indices_before_r",
    "r_q_markers_for_display",
    "sinus_reference_indices_pt",
    "stim_pulse_exclusion_spans_signed_m",
    "zero_pad_signal",
]
