import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from vel_model import vel_model

# Time axis
Fs = 1000                             # sampling rate (samples/sec)
t = np.arange(-0.1, 0.1+1.0/Fs, 1.0/Fs) # time axis (sec)
eta_eye = 600.0                             # (degree/sec)
# hand
eta_hand=300.0
c = 8.8                                 # (no units)
amplitude = 10                        # (degree)

threshold=5 # the velocity threshold (deg/s), below this is considered as 'stop moving'.

for eta in [eta_eye,eta_hand]:
	trajectory, velocity, tmp = vel_model(t, eta, c, amplitude)

	idx=np.where(velocity<threshold)

	trajectory=np.delete(trajectory,idx)
	velocity=np.delete(velocity,idx)
	t1=np.delete(t,idx)

	stage=np.where(t1<0,11,22)





	t1=t1+max(t1)
    # Plot
	fig = plt.figure(3, figsize=(8,10))
	ax = plt.subplot(211)    

	ax.plot(1000*t1, trajectory,'o:')

	
	plt.ylim([0,20])
	plt.xlabel("Time (ms)")
	plt.ylabel("Angle (deg)")
	plt.title(f'amplitude= {amplitude} deg, sample_time={1000/Fs} ms')
	plt.legend(['eye','hand'])



	ax = plt.subplot(212)    
	ax.plot(1000*t1,velocity,'o:')
	#ax.plot(velocity_hand, 'g',label='hand')


	plt.ylim([0,600])
	plt.xlabel("Time (ms)")
	plt.ylabel("Velocity (deg/s)")
	plt.title("")
	plt.legend(['eye','hand'])

plt.show()