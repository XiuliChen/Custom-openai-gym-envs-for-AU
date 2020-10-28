
import numpy as np
import glob
from PIL import Image
import matplotlib.pyplot as plt
from numpy import genfromtxt


def _calc_dis(p,q):
  '''
  calculate the Euclidean distance between points p and q 
  '''
  return np.sqrt(np.sum((p-q)**2))


my_data = genfromtxt('test.csv', delimiter=',')
episodes=np.unique(my_data[:,0])
plt.figure(figsize=(5,5))

for e in episodes:
	data=my_data[:,0]==e
	data_episode=my_data[data,1:]
	n_steps=len(data_episode)
	target_pos=data_episode[0,5:7]

	plt.plot(0,0,'k+',markersize=25,label='Start')
	'''
	ax = plt.gca()
	circle1 = plt.Circle((target_pos[0],target_pos[1]), 0.2, color='grey',label='Target')
	ax.add_artist(circle1)
	'''
	
	
	
	'''
	if target_pos[1]>0:
		plt.ylim([-0.1,0.6])
	else:
		plt.ylim([-0.6,0.1])
	'''
	for step in range(n_steps):
		
		if step>0:
			plt.clf()
			plt.plot(target_pos[0],target_pos[1],'g*',markersize=25,label='Target')
			plt.axis('off')
			plt.grid('on')
			eye_pos_prev=data_episode[step-1,1:3]
			eye_pos=data_episode[step,1:3]
			plt.plot(eye_pos[0],eye_pos[1],'ko',label='eye',markersize=20)
			'''
			if step==1:
				plt.plot([eye_pos_prev[0],eye_pos[0]],[eye_pos_prev[1],eye_pos[1]],'ko:',label='Eye')
			else:
				plt.plot([eye_pos_prev[0],eye_pos[0]],[eye_pos_prev[1],eye_pos[1]],'ko:')
			'''
			

			hand_pos_prev=data_episode[step-1,3:5]
			hand_pos=data_episode[step,3:5]
			plt.plot(hand_pos[0],hand_pos[1],'r>',label='hand',markersize=20)
			plt.xlim([-0.6,0.6])
			plt.ylim([-0.6,0.6])
			'''
			if step==1:
				plt.plot([hand_pos_prev[0],hand_pos[0]],[hand_pos_prev[1],hand_pos[1]],'r<-',label='hand')
			else:
				plt.plot([hand_pos_prev[0],hand_pos[0]],[hand_pos_prev[1],hand_pos[1]],'r<-')
			'''
			plt.legend()
			'''
			plt.subplot(1,2,2)
			dis=_calc_dis(target_pos,eye_pos)
			dis_hand=_calc_dis(target_pos,hand_pos)
			plt.plot(step,dis,'ko')
			plt.plot(step,dis_hand,'r>')
			plt.xlim(0,n_steps)
			plt.ylim(-0.1,0.6)
			plt.ylabel('Distance to target')
			'''

			
			
				


			


			plt.savefig(f'gif_figs/{step+10}.png')


# filepaths
fp_in = "gif_figs/*.png"
fp_out = "gif_figs/image.gif"

# https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
img.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=100)


