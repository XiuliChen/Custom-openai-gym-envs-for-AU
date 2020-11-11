# this file is to plot the model performance in Fitts' Law Style.
import os
import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from envs.EyeHandEnv import EyeHandEnv

subfolder1=os.listdir('logs7')

swapping=0.1


eta_eye=600
eta_hand=200
scale_deg=40
for ocular in [0.01,0.05,0.1,0.2]:
	for motor in [0.01,0.05,0.1,0.2]:
		for w in [0.0125,0.05,0.1,0.2,0.25,0.3]:
			for file in subfolder1:
				if f'w{w}d0.5ocular{ocular}swapping{swapping}motor{motor}' in file:
					print(file)
					subfolder2=os.listdir(f'logs7/{file}')
					r=0
					for items in subfolder2:			
						if 'run' in items:
							r+=1
					n_eps=1000
					complete_time_all=np.zeros((n_eps,r),dtype=np.float32)
					names=[]

					
					r=0		
					for items in subfolder2:

						if 'run' in items:
							
							subfolder3=os.listdir(f'logs7/{file}/{items}/savedmodel')
							if len(subfolder3)>0:
								print(subfolder3[-1][0:-4])
								env = EyeHandEnv(fitts_W = w, fitts_D=0.5, ocular_std=ocular, 
									swapping_std=swapping, motor_std=0.1,eta_eye=eta_eye,
									eta_hand=eta_hand,scale_deg=scale_deg)

								model=PPO.load(f'logs7/{file}/{items}/savedmodel/{subfolder3[-1][0:-4]}')
								
								for eps in range(n_eps):
									done=False
									obs=env.reset()
									ct=0.0
									time_step=20
									while not done:
										action, _ = model.predict(obs)
										obs, reward, done, info = env.step(action)
										ct+=time_step
										if done:
											break
									complete_time_all[eps,r]=(ct)
								names.append(items)
								r+=1


					plt.boxplot(complete_time_all,showfliers=False)
					print(names)
					plt.title(names)
					plt.savefig(f'logs7/{file}/complete_time.png')
					plt.close('all')


	print('------')

