
import numpy as np
import glob
from PIL import Image
import matplotlib.pyplot as plt
from numpy import genfromtxt


def _calc_dis(p,q):
	return np.sqrt(np.sum((p-q)**2))


my_data = genfromtxt('test.csv', delimiter=',')
episodes=np.unique(my_data[:,0])
plt.figure(figsize=(5,5))
for e in episodes:
	data=my_data[:,0]==e
	data_episode=my_data[data,1:]
	n_steps=len(data_episode)
	target_pos=data_episode[0,5:7]
	for step in range(n_steps):
		prep_eye=data_episode[step,7:9]
		print(prep_eye)


		eye_pos=data_episode[step,1:3]
		hand_pos=data_episode[step,3:5]
		dis=_calc_dis(target_pos,eye_pos)
		dis_hand=_calc_dis(target_pos,hand_pos)
		if step==0:
			plt.plot(step,dis,'ko',label='Eye')
			plt.plot(step,dis_hand,'r>',label='Hand')
			plt.legend()
		else:
			if prep_eye==1:
				plt.plot(step,dis,'g*',markersize=15)
			else:
				plt.plot(step,dis,'ko')
			plt.plot(step,dis_hand,'r>')
		
		plt.xlim(-1,n_steps)
		plt.ylim(-0.1,0.6)
		plt.ylabel('Distance to target')
		plt.savefig(f'gif_figs_dis/{step+10}.png')


# filepaths
fp_in = "gif_figs_dis/*.png"
fp_out = "gif_figs_dis/dis.gif"

# https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
img.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=300, loop=1)

