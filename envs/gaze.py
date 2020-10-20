
import numpy as np
import gym
from gym import spaces
import math
from vel_model import _vel_profile

###########################################################
def _calc_dis(p,q):
  '''
  calculate the Euclidean distance between points p and q 
  '''
  return np.sqrt(np.sum((p-q)**2))
###########################################################



class Gaze(gym.Env):
  def __init__(self,fitts_W = 0.05, fitts_D=0.5, ocular_std=0.1, swapping_std=0.2, motor_std=0.1):
    super(Gaze,self).__init__()

    # task setting
    self.scale_deg=20 # 1.0 in the cavas equals to 20 degress

    self.fitts_W=fitts_W
    self.fitts_D=fitts_D
    # agent ocular motor noise and visual spatial noise
    self.ocular_std=ocular_std
    self.swapping_std=swapping_std
    self.motor_std=motor_std

    self.time_step=int(50) # the time step for the controller (unit ms)
    # timings (unit: self.time_step)
    self.time_prep_eye=self.time_step*2
    self.time_prep_hand=self.time_step*2
    self.time_fixation=self.time_step*2

    self.max_steps=20

    # action space and observation space
    self.action_space = spaces.Box(low=-1, high=1, shape=(2, ), dtype=np.float32)

  def reset(self):
    # STEP1:  initialize the state
    # the state of the env includes three elements: target, eye
    
    # (1) target

    angle=np.random.uniform(0,math.pi*2) 
    x_target=math.cos(angle)*self.fitts_D
    y_target=math.sin(angle)*self.fitts_D
    self.target_pos = np.array([x_target,y_target])


    # (2) eye
    self.pos_eye=np.array([0.0,0.0])
    self.moving_to_eye=np.array([0.0,0.0])
    '''
    self.prep_eye=0
    self.fixate_eye=0

    self.vel_eye=0.0
    self.eye_status=np.concatenate((self.pos_eye,self.prep_eye,self.moving_to_eye,self.fixate_eye,self.vel_eye),axis=None)



    self.state = np.concatenate((self.target_pos,
                              self.eye_status),axis=None)
    '''


    # step 2: initial obs and belief
    # first obs and belief
    self.fixate=np.array([0,0])
    self.obs,self.obs_uncertainty=self._get_obs()

    self.belief,self.belief_uncertainty=self.obs, self.obs_uncertainty
    
    '''
    self.belief_full=self.state
    self.belief_full[0:2]=self.belief
    '''

    print(f'target_pos={self.target_pos}')
    print(f'inital belief={self.belief}')

    self.n_steps=0

    self.PREP=False
    self.START_MOVE=False
    self.MOV=False
    self.FIX=False

    



    return self.belief

  def step(self,action):
    self.n_steps+=1
    self._state_transit(action)

    # check if the eye is within the target region
    dis_to_target=_calc_dis(self.target_pos, self.fixate)

    if  dis_to_target < self.fitts_W/2:
      done = True
      reward = 0
    else:
      done = False
      reward = -1 


    if self.n_steps>self.max_steps:
      done=True

    info={}
    return self.belief, reward, done, info

 

  ################################
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

  def _state_transit(self,action):
    # execute the action: move the eye and/or move the hand
    eye_move_amp=_calc_dis(action[0:2],self.moving_to_eye)
    #hand_move_amp=_calc_dis(action[2:4],self.moving_to_hand)

    
    if eye_move_amp>0.005:
      # new eye command
      self.moving_to_eye=action[0:2]
      print(f'New eye target:{action[0:2]}')
      self.PREP=True
      self.prep_eye=0
      self.START_MOVE=False
      self.MOV=False
      self.FIX=False

    if self.PREP:
      print(f'eye prep: {self.prep_eye} ms')
      prep_stage=np.round(self.prep_eye/self.time_prep_eye,2)
      

      self.prep_eye+=self.time_step
      if self.prep_eye>self.time_prep_eye:
        self.PREP=False
        self.START_MOVE=True


    if self.START_MOVE:
      # intend pos and actual pos (ocular motor noise)
      move_dis=_calc_dis(self.pos_eye,action[0:2])
      ocular_noise=np.random.normal(0, self.ocular_std*move_dis, (2,))
      end_eye= action[0:2] + ocular_noise
      self.end_eye=np.clip(end_eye,-1,1)
      amp=_calc_dis(self.pos_eye,end_eye)*self.scale_deg
      stage_e,self.trajectory_e,self.velocity_e=_vel_profile(amp,1,self.time_step)
      self.pos_e=[]
      for r in (self.trajectory_e/amp):
        self.pos_e.append(self.pos_eye+r*(end_eye-self.pos_eye))
      
      self.n_move_steps=len(self.trajectory_e)
      print(f'Start Moving to {self.end_eye}')



      
      self.mov_step=0
      self.START_MOVE=False
      self.MOV=True

      '''
      print(n_move_steps)
      print(f'start={self.pos_eye}')
      print(f'end={end_eye}')
      print(pos)
      '''
    

    if self.MOV:
      mov_stage=np.round(self.mov_step/(self.n_move_steps),2)

      # update vel and pos
      self.vel_eye=self.velocity_e[self.mov_step]
      self.pos_eye=self.pos_e[self.mov_step]


      self.mov_step+=1
      if self.mov_step==self.n_move_steps:
        self.MOV=False
        self.FIX=True
        self.fix_step=0
        

      

    if self.FIX:
      print(f'Fixating: {self.fix_step} ms')
      self.fix_step+=self.time_step

      if self.fix_step>self.time_fixation:
        print('Info Extract!')
        self.fixate=self.end_eye
        self.FIX=False
        self.obs,self.obs_uncertainty=self._get_obs()
        self.belief,self.belief_uncertainty=self._get_belief()
        print(f'self.belief={self.belief}')
        # update belief









if __name__=="__main__":
  env=Gaze()
  obs=env.reset()
  action=env.action_space.sample()

  for i in range(20):
    print('--------------------------------------')
    print(f'step={i}, time:{(i-1)*50}ms--{(i)*50}ms')
    observation, reward, done, info = env.step(action)
    action=observation
    print(f'observation, reward, done={observation}, {reward}, {done}')
    print(f'eye pos={env.pos_eye}')




    if done:
      break

