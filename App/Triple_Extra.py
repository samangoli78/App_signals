
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
        P,dic=find_peaks(stiulation,distance=250,height=0.75*np.max(stiulation))
        diff=[P[i+1]-P[i] for i in np.arange(len(P)-1)]
        if ax:
            ax.plot(P*0.001,dic["peak_heights"],"x")
        diff.append(diff[0]+28) 
        self.stim_start=P
        self.stim_ref=P
        self.stim_duration=diff
        refference=np.abs(refference)


        P,dic=find_peaks(refference,distance=300,height=0.45*np.max(refference))
        if ax:
            ax.plot(P*0.001,dic["peak_heights"],"x")

        sinus_ref=P
        sinus_ref_new=[]
        sinus_end=[]
        sinus_start=[]
        duration=225
        for ii in sinus_ref:
            start=ii-duration
            end=ii+margin 
            if 0<end<self.N and 0<start<self.N and (start<self.stim_start[0] or self.stim_start[-1]+self.stim_duration[-1]<start ):
                sinus_ref_new.append(ii)
                sinus_start.append(start)
                sinus_end.append(end)

        
        self.sinus_ref=sinus_ref_new
        self.sinus_start=sinus_start
        self.sinus_end=sinus_end
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