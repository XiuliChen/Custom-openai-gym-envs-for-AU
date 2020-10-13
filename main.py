from envs.goleftenv import GoLeftEnv

from stable_baselines import PPO2
from stable_baselines.common.cmd_util import make_vec_env

# Instantiate the env
env = GoLeftEnv(grid_size=10)

# wrap it
env = make_vec_env(lambda: env, n_envs=1)


# Train the agent
model = PPO2('MlpPolicy', env, verbose=1).learn(5000)


# Test the trained agent
obs = env.reset()
n_steps = 20
for step in range(n_steps):
  action, _ = model.predict(obs, deterministic=True)
  print("Step {}".format(step + 1))
  print("Action: ", action)
  obs, reward, done, info = env.step(action)
  print('obs=', obs, 'reward=', reward, 'done=', done)
  env.render(mode='console')
  if done:
    # Note that the VecEnv resets automatically
    # when a done signal is encountered
    print("Goal reached!", "reward=", reward)
    break
'''

env = GoLeftEnv(grid_size=10)

obs = env.reset()
env.render()

print(env.observation_space)
print(env.action_space)
print(env.action_space.sample())


GO_LEFT = 0
# Hardcoded best agent: always go left!
n_steps = 20
for step in range(n_steps):
  print("Step {}".format(step + 1))
  obs, reward, done, info = env.step(GO_LEFT)
  print('obs=', obs, 'reward=', reward, 'done=', done)
  env.render()
  if done:
    print("Goal reached!", "reward=", reward)
    break

'''