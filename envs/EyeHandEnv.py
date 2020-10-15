import numpy as np
import gym
from gym import spaces
import math
from ./velocity_profile/vel_model import vel_model

###########################################################
def _calc_dis(p,q):
  '''
  calculate the Euclidean distance between points p and q 
  '''
  return np.sqrt(np.sum((p-q)**2))
###########################################################


class EyeHandEnv(gym.Env):
  '''
    Description:
            The agent searchs from a target on the display,
            moves its hand to the target, and stays on it for a fixed
            amount of time. 

            Constraints:
            Vision:
              Visual spatial swapping noise (eccentricity dependent)
            Motor:
              Ocular motor noise (signal dependent)
              Hand motor noise
            Cost:
              Time cost
              Energy cost
         

    States:
          target: 
            [x_target,y_target]

          eye:
            stage_eye: scalar [0=prep, 0.3=ramp-up 0.6=ramp-down 1=fixate]
            vel_eye: scalar
            moving_to_eye: [x_eye,y_eye],
            pos_eye: [x,y]
          hand:
            stage_hand: scalar [0=prep, 0.3=ramp-up 0.6=ramp-down 1=fixate]
            vel_hand: scalar
            moving_to_hand: [x,y],
            pos_hand: [x,y]

            on_target_hand

          type: Box(15, )


    Actions: 
          move_eye_to [x_eye,y_eye], 
          move_hand_to [x_hand,y_hand]
            
          type: Box(4, )


    Observation:   
          same as state, but        
          target: 
            [x_target_estimate,y_target_estimate], 

    Belief: 
          same as the state          
          target: 
            [x_target_estimate,y_target_estimate], 



    Reward:
            Reward of 0 is awarded if the eye reach the target.
            reward of -1 is awared if not


    Episode Termination:
            the eye reaches the target (within self.fitts_W/2 distance)
            or reach the maximum steps
  '''

  # define constants
  # stages for eye and hand
  PREP=0.25
  RAMP_UP=0.5
  RAMP_DOWN=0.75
  FIXATION=1


  def __init__(self, fitts_W = 0.2, fitts_D=0.5, ocular_std=0.1, swapping_std=0.1,motor_std=0.2):
    super(EyeHandEnv, self).__init__()

    # task setting
    self.fitts_W=fitts_W
    self.fitts_D=fitts_D

    # agent ocular motor noise and visual spatial noise
    self.ocular_std=ocular_std
    self.swapping_std=swapping_std
    self.motor_std=motor_std

    # Define action and observation space
    # They must be gym.spaces objects
    self.state_space = spaces.Box(low=-1, high=1, shape=(15, ), dtype=np.float32)
    self.observation_space = spaces.Box(low=-1, high=1, shape=(15, ), dtype=np.float32)
    self.belief_space = spaces.Box(low=-1, high=1, shape=(15, ), dtype=np.float32)

    self.action_space = spaces.Box(low=-1, high=1, shape=(4, ), dtype=np.float32)

    self.max_steps=1000
    self.time_step=int(50) # 50 ms

    self.time_prep_eye=int(50) # how many ms
    self.time_prep_hand=int(100)
    self.time_fixation=int(100)

  def reset(self):
    """
    Important: the observation must be a numpy array
    :return: (np.array) 
    """

    # Initialise the state

    # choose a random target with distance (self.D) away from the center
    angle=np.random.uniform(0,math.pi*2) 
    x_target=math.cos(angle)*self.fitts_D
    y_target=math.sin(angle)*self.fitts_D
    self.target_pos = np.array([x_target,y_target])

    self.stage_eye=0
    self.vel_eye=0.0
    self.moving_to_eye=np.array([0.0,0.0])
    self.pos_eye=np.array([0.0,0.0])
    

    self.stage_hand=0
    self.vel_hand=0.0
    self.moving_to_hand=np.array([0.0,0.0])
    self.pos_hand=np.array([0.0,0.0])

    self.hand_on_target=0.0

    self.state = np.concatenate((self.target_pos,
                              self.stage_eye,self.vel_eye,self.moving_to_eye,self.pos_eye,
                              self.stage_hand,self.vel_hand,self.moving_to_hand,self.pos_hand,
                              self.hand_on_target),axis=None)

    print(f'state={self.state}')

    # first obs and belief
    self.fixate=np.array([0,0])
    self.obs,self.obs_uncertainty=self._get_obs()
    print(f'obs={self.obs}')

    # the initial belief
    self.belief,self.belief_uncertainty=self.obs, self.obs_uncertainty
    
    self.belief_full=self.state
    self.belief_full[0:2]=self.belief

    # first fixation at the center
    self.n_fixation=1
    self.eps_time=self.time_fixation+self.time_step

    self.n_hand_move=0
    self.n_steps=0

    return self.belief_full


  def step(self, action):
    self.eps_time+=self.time_step
    self.n_steps+=1


    # take the action
    # state transition
    _state_transit(self,action)
    
    # new obs and new belief
    if self.stage_eye==FIXATION:
      self.fixate=[]
      self.obs,self.obs_uncertainty=self._get_obs()
      self.belief,self.belief_uncertainty=self._get_belief()
      self.belief_full=self.state
      self.belief_full[0:2]=self.belief



    

    # reward


    # check if the eye is within the target region
    dis_to_target=_calc_dis(self.target_pos, self.pos_hand)

    if  dis_to_target < self.fitts_W/2:
      self.on_target_hand=1
      done = True
      reward = 0
    else:
      done = False
      reward = -1 
      # has not reached the target, get new obs at the new fixation location
      self.obs,self.obs_uncertainty=self._get_obs()
      self.belief,self.belief_uncertainty=self._get_belief()

    if self.n_steps>self.max_steps:
        done=True


    info={}

    # Optionally we can pass additional info, we are not using that for now
    return self.belief_full, reward, done, info

    

    #return np.array([self.agent_pos]).astype(np.float32), reward, done, info



  def _state_transit(self,action):
    '''

    States:
      target: 
        [x_target,y_target]

      eye:
        stage_eye: scalar [0=prep, 0.3=ramp-up 0.6=ramp-down 1=fixate]
        vel_eye: scalar
        moving_to_eye: [x_eye,y_eye],
        pos_eye: [x,y]
      hand:
        stage_hand: scalar [0=prep, 0.3=ramp-up 0.6=ramp-down 1=fixate]
        vel_hand: scalar
        moving_to_hand: [x,y],
        pos_hand: [x,y]

        on_target_hand
    '''
    # execute the action: move the eye and/or move the hand
    eye_move_amp=_calc_dis(action[0:2],self.eye_moving_to)
    # New eye command
    if eye_move_amp>0.01:
      prep_seq=np.full((self.time_prep_eye,),fill_value=PREP)
      fixate_seq=np.full((self.time_fixation,),fill_value=FIXATION)

      # move the eye to a new location with Ocular motor noise
      # i.e. the end position is corrupted
      move_dis=_calc_dis(self.pos_eye,action[0:2])
      ocular_noise=np.random.normal(0, self.ocular_std*move_dis, (2,))
      end_eye= action[0:2] + ocular_noise
      end_eye=np.clip(end_eye,-1,1)
      amp=_calc_dis(self.eye_pos,end_eye)
      stage_e,trajectory_e,velocity_e=_vel_profile(amp,1)

      stage_squence=np.concatenate((prep_seq,stage_e,fixate_seq),axis=None)
      traj_sequence=np.concatenate((np.zeros_like(prep_seq),trajectory_e_e,np.zeros_like(fixate_seq)),axis=None)
      vel_sequence=np.concatenate((np.zeros_like(prep_seq),vel_e,np.zeros_like(fixate_seq)),axis=None)
      
      sequence_step=0
      # target no change
      self.stage_eye=stage_squence[sequence_step]
      self.vel_eye=vel_sequence[sequence_step]
      self.eye_moving_to=action[0:2]
      self.pos_eye=xxx


    else:
      sequence_step+=1



    hand_move_amp=_calc_dis(action[2:4],self.hand_moving_to)


    # move the eye (Ocular motor noise, signal dependent)
    move_dis=_calc_dis(self.pos_eye,action[0:2])
    ocular_noise=np.random.normal(0, self.ocular_std*move_dis, (2,))
    self.end_eye= action[0:2] + ocular_noise
    self.end_eye=np.clip(self.end_eye,-1,1)





    self.stage_eye=0
    self.vel_eye=0.0
    self.moving_to_eye=np.array([0.0,0.0])
    self.pos_eye=np.array([0.0,0.0])
    

    self.stage_hand=0
    self.vel_hand=0.0
    self.moving_to_hand=np.array([0.0,0.0])
    self.pos_hand=np.array([0.0,0.0])



    self.hand_on_target=0.0

    self.state = np.concatenate((self.target_pos,
                              self.stage_eye,self.vel_eye,self.moving_to_eye,self.pos_eye,
                              self.stage_hand,self.vel_hand,self.moving_to_hand,self.pos_hand,
                              self.hand_on_target),axis=None)






  def _get_obs(self):
    eccentricity=_calc_dis(self.target_pos,self.fixate)
    obs_uncertainty=eccentricity
    spatial_noise=np.random.normal(0, self.swapping_std*eccentricity, self.target_pos.shape)
    obs=self.target_pos + spatial_noise
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


  def _vel_profile(self,amplitude,hand_or_eye):
    # Time axis
    Fs = 1000                             # sampling rate (samples/sec)
    t = np.arange(-0.1, 0.1+1.0/Fs, 1.0/Fs) # time axis (sec)
    if hand_or_eye==1:
      eta= 600.0                             # (degree/sec)
    else:
      eta=300.0
    c = 8.8                                 # (no units)
    threshold=5 # the velocity threshold (deg/s), below this is considered as 'stop moving'.

    trajectory, velocity, tmp = vel_model(t, eta, c, amplitude)

    
    idx=np.where(velocity<threshold)
    trajectory=np.delete(trajectory,idx)
    velocity=np.delete(velocity,idx)
    t1=np.delete(t,idx)

    stage=np.where(t1<0,RAMP_UP,RAMP_DOWN)
 
    #t1=t1+max(t1)
    
    return stage,trajectory,velocity




if __name__=="__main__":
  env = EyeHandEnv()
  obs=env.reset()
  print('init obs=', obs)
  '''
  done =False
  while not done:
    action = env.action_space.sample() # your agent here (this takes random actions)
    print('action=', np.round(action,2))

    observation, reward, done, info = env.step(action)
    print('belief=', np.round(observation,2), 'reward=', reward, 'done=', done)


  env.close()
  '''





