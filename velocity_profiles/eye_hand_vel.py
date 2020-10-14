import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from vel_model import vel_model

# Time axis
Fs = 2000                               # sampling rate (samples/sec)
t = np.arange(-0.1, 0.1+1.0/Fs, 1.0/Fs) # time axis (sec)
eta_eye = 600.0                             # (degree/sec)
# hand
eta_hand=300.0
c = 8.8                                 # (no units)
amplitude = 10                        # (degree)

threshold=1 # the velocity threshold (deg/s), below this is considered as 'stop moving'.

for eta in [eta_eye,eta_hand]:
	waveform_eye, velocity_eye, tmp = vel_model(t, eta, c, amplitude)

	idx1=np.where(velocity_eye<threshold)

	waveform_eye=np.delete(waveform_eye,idx1)
	t1=np.delete(t,idx1)
	velocity_eye=np.delete(velocity_eye,idx1)
	t1=t1+max(t1)




	# Plot
	fig = plt.figure(3, figsize=(8,10))
	ax = plt.subplot(211)    

	ax.plot(1000*t1, waveform_eye)

	
	plt.ylim([0,20])
	plt.xlabel("Time (ms)")
	plt.ylabel("Angle (deg)")
	plt.title(f'amplitude= {amplitude} deg')
	plt.legend(['eye','hand'])



	ax = plt.subplot(212)    
	ax.plot(1000*t1,velocity_eye)
	#ax.plot(velocity_hand, 'g',label='hand')


	plt.ylim([0,600])
	plt.xlabel("Time (ms)")
	plt.ylabel("Velocity (deg/s)")
	plt.title("")
	plt.legend(['eye','hand'])

plt.show()