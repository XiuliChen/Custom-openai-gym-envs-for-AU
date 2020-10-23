from envs.gaze import Gaze

from stable_baselines import PPO2
from stable_baselines.common.cmd_util import make_vec_env

# Instantiate the env
env = Gaze()
# wrap it
env = make_vec_env(lambda: env, n_envs=1)


# Train the agent
model = PPO2('MlpPolicy', env, verbose=1)

# train the agent
model.learn(total_steps=int(2e5))

model.save("ppo2_gaze")

# model=PPO2.load("ppo2_gaze")



# Test the trained agent
obs = env.reset()
n_steps = 20
for step in range(n_steps):
  action, _ = model.predict(obs, deterministic=True)
  obs, reward, done, info = env.step(action)
  print('obs=', obs, 'reward=', reward, 'done=', done)
  
  # env.render(mode='console')
  if done:
    # Note that the VecEnv resets automatically
    # when a done signal is encountered
    print("Target found!", "reward=", reward)
    break


