
import numpy as np
import math


# some tool functions
def calc_dis(p,q):
    #calculate the Euclidean distance between points p and q 
    return np.sqrt(np.sum((p-q)**2))

###########################################################

def get_new_target(D):
    '''
    generate a target at a random angle, distance D away.
    '''
    angle=np.random.uniform(0,math.pi*2) 
    x_target=math.cos(angle)*D
    y_target=math.sin(angle)*D
    return np.array([x_target,y_target])
###########################################################

def get_trajectory(mode_eta,amp,current_pos,actual_pos,time_step):
    
    # calculate the moving distance
    trajectory,velocity=_vel_profiles(amp,mode_eta,time_step)
    pos=[]
    pos.append(current_pos)
    for r in (trajectory/amp):
      pos.append(current_pos+r*(actual_pos-current_pos))
    pos.append(actual_pos)
    velocity=[0,*velocity,0]

    return pos, velocity

def _vel_profiles(amplitude,mode_eta,time_step):
    # Time axis
    Fs = 1000/time_step                            # sampling rate (samples/sec)
    t = np.arange(-0.1, 0.1+1.0/Fs, 1.0/Fs) # time axis (sec)

    eta= mode_eta                             # (degree/sec)

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



if __name__=="__main__":
    import matplotlib.pyplot as plt
    # unit testing
    p=np.array([0,0])
    q=np.array([1,1])
    dis=calc_dis(p,q)
    print(dis)

    D=0.5
    for i in range(100):
        target_pos=get_new_target(D)
        plt.plot(target_pos[0],target_pos[1],'ro')
 

    plt.figure()
    
    mode=1 # eye
    current_pos=np.array([0,0])
    actual_pos=np.array([0,0.7])
    amp=calc_dis(current_pos,actual_pos)*20
    time_step=20

    pos,velocity=get_trajectory(mode,amp,current_pos,actual_pos,time_step)
    plt.plot(current_pos[0],current_pos[1],'g>',markersize=15)
    plt.plot(actual_pos[0],actual_pos[1],'rs',markersize=15)
    print(pos)
    for i in pos:
        plt.plot(i[0],i[1],'b*-')
    plt.xlim([-1,1])
    plt.ylim([-0.1,1])

    
    plt.figure()
    tra,vel1=_vel_profiles(9,550,1)
    plt.plot(vel1,'k.',label='eye')
    tra,vel2=_vel_profiles(29,150,1)
    plt.plot(vel2,'r.',label='hand')
    plt.legend()
    plt.grid()
    plt.yticks(np.arange(0,700,50))

    plt.savefig(f'vel.png')





    




    
    
