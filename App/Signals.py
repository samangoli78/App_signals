from scipy.signal import butter,filtfilt,get_window,find_peaks,argrelextrema,resample

import numpy as np
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d,CubicSpline
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

def derivative(y):
    yt=np.append(y,0)
    dy=np.array([(yt[i]-yt[i-1])/0.001 for i in np.arange(1,len(yt))])
    return dy
def apply_closing(binary_signal, structure_size=5):
    structure = np.ones(structure_size)
    return binary_closing(binary_signal, structure=structure)
def find_start(x,y,length=7,ax=None,operation="min",Th=0.2,alpha=0.5):
    #if y.max()<0.2:
    #    Th+=0.2
    #else:
    #    pass
    #y=minmax_scale(y, feature_range=(0, 1), axis=0, copy=True)
    y_paded=zero_pad_signal(y,pad_left=10,pad_right=10)
    Th=otsu_threshold(y_paded,alpha=alpha)
    #window=np.array([1]*length)
    #out=[]
    #for i,value in enumerate(y):
    #    start=i-length+1
    #    end=i
    #    if start<0:
    #        start=0
        
        #w=window[:end+1]*y[start:end+1]
    #    w=y[start:end+1]
    #    if operation =="min":
    #        out.append(w.min())
    #    else:
    #        out.append(w.max())
    #out=np.array(out)
    out=np.array(y)
    cs=interp1d(x,out)
    xs=np.arange(x.min(),x.max(),0.001)
    out=cs(xs)
    out_paded=zero_pad_signal(out,pad_left=10,pad_right=10)
    out=np.array([1 if ii>Th else 0 for ii in out_paded])
    out=apply_closing(out,15)



    #out1=np.append(out,0)
    out1=out
    output=[]
    start=0
    end=-1
    in_1=False
    for ii,val in enumerate(out1):
        if val==1 and in_1==False:
            start=ii
            in_1=True
        elif val==0 and in_1==True:
            end=ii
            in_1=False
            if end-start>5:
                output.append([start-10,end-10,end-start])
    index=np.argsort(np.array([val[2] for val in output]))
    out1=[]
    for ii in index:
        out1.append([output[ii][jj] for jj in range(2)])
        
    
    if not isinstance(ax,type(None)):
        pass
        #ax.plot(xs,out)
        #ax.plot(x,y)
    try:
        return out1[-1]
    except:
        return None
    



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