from envs.gaze import Gaze

from stable_baselines import PPO2
from stable_baselines.common.cmd_util import make_vec_env

# Instantiate the env
env = Gaze()
# wrap it
env = make_vec_env(lambda: env, n_envs=1)


# Train the agent
model = PPO2('MlpPolicy', env, verbose=1).learn(5000)



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


