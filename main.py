
import os

import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback

# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(save_freq=100000, save_path='./logs/',
                                         name_prefix='rl_model')


#from envs.gaze import Gaze
from envs.EyeHandEnv import EyeHandEnv




# Create log dir
log_dir = "tmp/"

os.makedirs(log_dir, exist_ok=True)

# Instantiate the env
env = EyeHandEnv()
env = Monitor(env, log_dir)

# Train the agent
model = PPO('MlpPolicy', env, verbose=1)

# Train the agent
timesteps = 2e7
model.learn(total_timesteps=int(timesteps), callback=checkpoint_callback)


model.save('ppo_eye_hand2')


#plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "PPO_eye_hand")



def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results2(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=100)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.show()

plot_results2(log_dir)



'''

#model.save("ppo2_gaze5") # 600 vs 50
model.save("ppo2_gaze3") # 600 vs 600

# model=PPO2.load("ppo2_gaze")



# Test the trained agent
for eps in range(10):

	obs = env.reset()
	n_steps = 1000
	for step in range(n_steps):
		action, _ = model.predict(obs)
		obs, reward, done, info = env.step(action)
		print('obs=', obs, 'reward=', reward, 'done=', done)
		env.plot()
		# env.render(mode='console')
		if done:
			plt.savefig(f'fig{eps}')
			print("Target found!", "reward=", reward)
			break

'''
