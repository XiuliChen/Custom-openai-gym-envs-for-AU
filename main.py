


#from envs.gaze import Gaze
from envs.eh1 import EyeHandEnv

from stable_baselines import PPO2
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.common.callbacks import CheckpointCallback
# Save a checkpoint every 1000 steps

# Instantiate the env
env = EyeHandEnv()
# wrap it
#env = make_vec_env(lambda: env, n_envs=1)


# Train the agent
model = PPO2('MlpPolicy', env, verbose=1)


# train the agent
model.learn(int(5*1e5))

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


