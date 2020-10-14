import numpy as np
import gym
from gym import spaces
import math

###########################################################
def _calc_dis(p,q):
  '''
  calculate the Euclidean distance between points p and q 
  '''
  return np.sqrt(np.sum((p-q)**2))
###########################################################


class Gaze(gym.Env):
  '''
    Description:
            The agent moves the eye to the target on the display. 
            Constraints:
            (1) visual spatial swapping noise (eccentricity dependent)
            (2) Ocular motor noise (signal dependent)
            (3) Ocular shift ('jitter when fixation')

    States: the target position
            type: Box(2, )
            [-1,-1] top-left; 
            [1,1] bottom-right 

    Actions: the fixation position 
            type: Box(2, )
            [-1,-1] top-left; 
            [1,1] bottom-right 

    Observation: the estimate of where the target is based on one obs
            type: Box(2, )
            [-1,-1] top-left; 
            [1,1] bottom-right 

    Belief: the estimate of where the target is based on all obs
            type: Box(2, );
            [-1,-1] top-left; 
            [1,1] bottom-right 


    Reward:
            Reward of 0 is awarded if the eye reach the target.
            reward of -1 is awared if not


    Episode Termination:
            the eye reaches the target (within self.fitts_W/2 distance)
            or reach the maximum steps
  '''

  metadata = {'render.modes': ['console']}


  def __init__(self, fitts_W = 0.2, fitts_D=0.5, ocular_std=0.1, swapping_std=0.1):
    super(Gaze, self).__init__()
    # task setting
    self.fitts_W=fitts_W
    self.fitts_D=fitts_D

    # agent ocular motor noise and visual spatial noise
    self.ocular_std=ocular_std
    self.swapping_std=swapping_std

    self.potential_targets=np.array([[0,-self.fitts_D],[0,self.fitts_D],[self.fitts_D,0],[-self.fitts_D,0]])

    # Define action and observation space
    # They must be gym.spaces objects
    self.state_space = spaces.Box(low=-1, high=1, shape=(2, ), dtype=np.float64)
    self.action_space = spaces.Box(low=-1, high=1, shape=(2, ), dtype=np.float64)
    self.observation_space = spaces.Box(low=-1, high=1, shape=(2, ), dtype=np.float64)
    self.belief_space = spaces.Box(low=-1, high=1, shape=(2, ), dtype=np.float64)

    self.max_fixation=1000



  def reset(self):
    """
    Important: the observation must be a numpy array
    :return: (np.array) 
    """

    # choose a random target with distance (self.D) away from the center
    angle=np.random.uniform(0,math.pi*2)
    y=math.sin(angle)*self.fitts_D
    x=math.cos(angle)*self.fitts_D

    self.state = np.array([x,y])

    self.fixate=np.array([0,0])
    self.n_fixation=1

    # the first obs
    self.obs,self.obs_uncertainty=self._get_obs()

    # the initial belief
    self.belief,self.belief_uncertainty=self.obs, self.obs_uncertainty
    return self.belief


  def step(self, action):

    # execute the chosen action (Ocular motor noise, signal dependent)
    move_dis=_calc_dis(self.fixate,action)
    ocular_noise=np.random.normal(0, self.ocular_std*move_dis, action.shape)
    self.fixate= action + ocular_noise
    self.fixate=np.clip(self.fixate,-1,1)
    self.n_fixation+=1

    # check if the eye is within the target region
    dis_to_target=_calc_dis(self.state, self.fixate)
    if  dis_to_target < self.fitts_W/2:
      done = True
      reward = 0
    else:
      done = False
      reward = -1 
      # has not reached the target, get new obs at the new fixation location
      self.obs,self.obs_uncertainty=self._get_obs()
      self.belief,self.belief_uncertainty=self._get_belief()

    if self.n_fixation>self.max_fixation:
        done=True


    info={'target': self.state, 
      'belief': self.belief,
      'aim': action,
      'fixate': self.fixate,
      'fitts_W':self.fitts_W,
      'fitts_D':self.fitts_D}
    # Optionally we can pass additional info, we are not using that for now
    return self.belief, reward, done, info

    

    #return np.array([self.agent_pos]).astype(np.float32), reward, done, info

  def render(self, mode='console'):
    if mode != 'console':
      raise NotImplementedError()
    pass

  def close(self):
    pass

  def _get_obs(self):
    eccentricity=_calc_dis(self.state,self.fixate)
    obs_uncertainty=eccentricity
    spatial_noise=np.random.normal(0, self.swapping_std*eccentricity, self.state.shape)
    obs=self.state + spatial_noise
    obs=np.clip(obs,-1,1)

    return obs,obs_uncertainty

  def _get_belief(self):
    z1,sigma1=self.obs,self.obs_uncertainty
    z2,sigma2=self.belief,self.belief_uncertainty
    w1=sigma2**2/(sigma1**2+sigma2**2)
    w2=sigma1**2/(sigma1**2+sigma2**2)
    belief=w1*z1+w2*z2
    belief_uncertainty=np.sqrt( (sigma1**2+sigma2**2)/(sigma1**2 * sigma2**2))

    return belief, belief_uncertainty




if __name__=="__main__":
  env = Gaze()
  obs=env.reset()
  print('init obs=', obs)
  done =False
  while not done:
    action = env.action_space.sample() # your agent here (this takes random actions)
    print('action=', np.round(action,2))

    observation, reward, done, info = env.step(action)
    print('belief=', np.round(observation,2), 'reward=', reward, 'done=', done)


  env.close()




