from scipy.signal import butter,filtfilt,get_window,find_peaks,argrelextrema,resample

import numpy as np
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d,CubicSpline
from sklearn.preprocessing import minmax_scale
    
    

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

def find_start(x,y,length=7,ax=None,operation="min",Th=0.2):
    if y.max()<0.3:
        Th+=0.1
    else:
        pass
    y=minmax_scale(y, feature_range=(0, 1), axis=0, copy=True)
    
    window=np.array([1]*length)
    out=[]
    for i,value in enumerate(y):
        start=i-length+1
        end=i
        if start<0:
            start=0
        
        #w=window[:end+1]*y[start:end+1]
        w=y[start:end+1]
        if operation =="min":
            out.append(w.min())
        else:
            out.append(w.max())
    out=np.array(out)
    cs=interp1d(x,out)
    xs=np.arange(x.min(),x.max(),0.001)
    out=cs(xs)
    out=np.array([1 if ii>Th else 0 for ii in out])
    out1=np.append(out,0)
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
                output.append([start,end,end-start])
    index=np.argsort(np.array([val[2] for val in output]))
    out1=[]
    for ii in index:
        out1.append([output[ii][jj] for jj in range(2)])
        
    
    if not isinstance(ax,type(None)):
        pass
        #ax.plot(xs,out)
        #ax.plot(x,y)
    return out1[-1]