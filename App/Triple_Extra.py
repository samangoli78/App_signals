
from scipy.signal import find_peaks
import numpy as np
from .Signals import *

class Triple_Extra():

    def __init__(self,t, EGM,T=2.5,fs=1000,order=2):
        self.signal=EGM  #EGM givven to find triple attributes
        self.t=t
        self.T = T       # Sample Period
        self.fs = fs     # sample rate, Hz
        self.nyq = 0.5 * self.fs  # Nyquist Frequency
        self.order = order        # sin wave can be approx represented as quadratic
        self.N = int(self.T * self.fs)       # total number of samples


    def find_windows(self,ax,stiulation,refference,margin=0):
        # units here is based on samples number
        stiulation=np.abs(stiulation)
        P,dic=find_peaks(stiulation,distance=200,height=0.3*np.max(stiulation))
        diff=[P[i+1]-P[i] for i in np.arange(len(P)-1)]
        ax.plot(P*0.001,dic["peak_heights"],"x")
        diff.append(diff[0]+28) 
        self.stim_start=P
        self.stim_ref=P
        self.stim_duration=diff
        refference=np.abs(refference)
        ref=refference[:self.stim_start[0]-20]
        P1,dic=find_peaks(ref,distance=500,height=0.5*np.max(refference))
        
        P1=P1
        ax.plot(P1*0.001,dic["peak_heights"],"x")
        ref=refference[self.stim_start[-1]+self.stim_duration[-1]:]
        PP,dic=find_peaks(ref,distance=500,height=0.5*np.max(refference))
        PP=PP+self.stim_start[-1]+self.stim_duration[-1]
        ax.plot(PP*0.001,dic["peak_heights"],"x")
        P1=np.hstack([P1,PP])
        self.sinus_ref=P1
        margin
        sinus_end=[ii+margin if 0<ii+margin<self.N else ii for ii in P1 ]
        duration=325
        
        self.sinus_start=[ii-duration  if 0<ii-duration else 0 for ii in sinus_end]
        self.sinus_duration=[duration]*len(self.sinus_start)
    
    def median_convolution(self,y,n=3):
        f=np.zeros(len(y))
        for i ,_ in enumerate(y):
            if i//n==0:
                f[i]=np.median(y[i-i%n:i])
            else:
                f[i]=np.median(y[i-n:i])
        f=np.nan_to_num(f)
        return f