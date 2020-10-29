### Xiuli copied from https://codeocean.com/capsule/8467067/tree/v1
### 14 Oct 2020



# A parametric model for saccadic eye movement.
#
# The saccade model corresponds to the 'main sequence' formula:
#    Vp = eta*(1 - exp(-A/c))
# where Vp is the peak saccadic velocity and A is the saccadic amplitude.
#
# Reference:
# W. Dai, I. Selesnick, J.-R. Rizzo, J. Rucker and T. Hudson.
# 'A parametric model for saccadic eye movement.'
# IEEE Signal Processing in Medicine and Biology Symposium (SPMB), December 2016.
# DOI: 10.1109/SPMB.2016.7846860.

import numpy as np

def _vel_profiles(amplitude,hand_or_eye,time_step):
    # Time axis
    Fs = 1000/time_step                            # sampling rate (samples/sec)
    t = np.arange(-0.1, 0.1+1.0/Fs, 1.0/Fs) # time axis (sec)
    if hand_or_eye==0:
      eta= 600.0                             # (degree/sec)
    else:
      eta=200.0
    c = 8.8                                 # (no units)
    threshold=1 # the velocity threshold (deg/s), below this is considered as 'stop moving'.
    trajectory, velocity, tmp = vel_model(t, eta, c, amplitude)

    
    idx=np.where(velocity<threshold)
    trajectory=np.delete(trajectory,idx)
    velocity=np.delete(velocity,idx)
    t1=np.delete(t,idx)

    stage=np.where(t1<0,0.5,1)
 
    #t1=t1+max(t1)
    
    return trajectory,velocity


def vel_model(t, eta=600.0, c=6.0, amplitude=9.5, t0=0.0, s0=0.0):
    """
    A parametric model for saccadic eye movement.
    This function simulates saccade waveforms using a parametric model.
    The saccade model corresponds to the 'main sequence' formula:
        Vp = eta*(1 - exp(-A/c))
    where Vp is the peak saccadic velocity and A is the saccadic amplitude.
    
    Usage:
        waveform, velocity, peak_velocity = 
            saccade_model(t, [eta,] [c,] [amplitude,] [t0,] [s0])
    
    Input:
        t         : time axis (sec)
        eta       : main sequence parameter (deg/sec)
        c         : main sequence parameter (no units)
        amplitude : amplitude of saccade (deg)
        t0        : saccade onset time (sec)
        s0        : initial saccade angle (degree)

    Output:
        waveform      : time series of saccadic angle
        velocity      : time series of saccadic angular velocity
        peak_velocity : peak velocity of saccade

    Reference:
    W. Dai, I. Selesnick, J.-R. Rizzo, J. Rucker and T. Hudson.
    'A parametric model for saccadic eye movement.'
    IEEE Signal Processing in Medicine and Biology Symposium (SPMB), December 2016.
    DOI: 10.1109/SPMB.2016.7846860.
    """
    
    fun_f = lambda t: t*(t>=0)+0.25*np.exp(-2*t)*(t>=0)+0.25*np.exp(2*t)*(t<0)
    fun_df = lambda t: 1*(t>=0)-0.5*np.exp(-2*t)*(t>=0)+0.5*np.exp(2*t)*(t<0)
    tau = amplitude/eta         # tau: amplitude parameter (amplitude = eta*tau)
    
    if t0 == 0:
        t0 = -tau/2             # saccade onset time (sec)
    
    waveform = c*fun_f(eta*(t-t0)/c) - c*fun_f(eta*(t-t0-tau)/c) + s0
    velocity = eta*fun_df(eta*(t-t0)/c) - eta*fun_df(eta*(t-t0-tau)/c)
    peak_velocity = eta * (1 - np.exp(-amplitude/c))
    
    return waveform, velocity, peak_velocity
    
    
