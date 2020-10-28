
import numpy as np
import gym
from gym import spaces
import math
from envs.vel_model import _vel_profiles
#from vel_model import _vel_profiles
import matplotlib.pyplot as plt
import itertools

###########################################################
def discretised_action():
  resolution=0.1
  a = np.round(np.arange(-1,1.1,resolution),1)
  b = np.round(np.arange(-1,1.1,resolution),1)
  c = list(itertools.product(a, b))
  return c, len(c)

def _calc_dis(p,q):
  '''
  calculate the Euclidean distance between points p and q 
  '''
  return np.sqrt(np.sum((p-q)**2))
###########################################################



class EyeHandEnv(gym.Env):

  def __init__(self,fitts_W = 0.2, fitts_D=0.5, ocular_std=0.1, swapping_std=0.2, motor_std=0.1):
    super(EyeHandEnv,self).__init__()

    self.EYE=0
    self.HAND=1

    # task setting
    # 1.0 in the canvas equals to 20 degress 
    # This is used to calculate the velocity profile
    self.scale_deg=20 

    self.fitts_W=fitts_W
    self.fitts_D=fitts_D

    # agent visual spatial noise
    self.swapping_std=swapping_std
    # agent ocular and motor noise 
    self.ocular_std=ocular_std
    self.motor_std=motor_std

    # timings (unit: self.time_step)
    self.time_step=int(20) # the time step for the controller (unit ms)

    # one for the eye and one for the hand
    self.prep_duration=[self.time_step, self.time_step*3]
    self.fixation_duration=[self.time_step,self.time_step]

    self.max_steps=2000

    # miscellaneous
    
    # action space and observation space
    '''
    self.action_coords,self.n_coords=discretised_action()
    self.action_space = spaces.Discrete(self.n_coords+1)
    '''
    self.action_space = spaces.Discrete(3)

    # belief/state
    low_b=np.array([-1.0, -1.0, # target_pos
      -1.0, -1.0, # moving to_eye
      0, # velocity_eye
      -1.0, -1.0, # moving to_hand
      0], # velocity_hand
      dtype=np.float32)
    
    high_b=np.array([1.0, 1.0, # target_pos
      1.0, 1.0, # moving to_eye
      1, # velocity_eye
      1.0, 1.0,# moving to_hand
      1], # velocity_hand
      dtype=np.float32)


    self.observation_space = spaces.Box(low=low_b, high=high_b, dtype=np.float32)

  def reset(self):
    # STEP1:  initialize the state

    # the state of the env includes three elements: target, eye, hand
    
    # (1) target
    angle=np.random.uniform(0,math.pi*2) 
    x_target=math.cos(angle)*self.fitts_D
    y_target=math.sin(angle)*self.fitts_D
    self.target_pos = np.array([x_target,y_target])


    # (2) eye and hand
    self.PREP=[False,False]
    self.prep_step=[0,0]

    self.START_MOVE=[False,False]
    self.moving_to=np.array([[.0,.0],[.0,.0]])
    self.actual_pos=np.array([[.0,.0],[.0,.0]])
    self.n_move_steps=[0,0]


    self.MOV=[False,False]
    self.mov_step=[0,0]
    self.vel=[0.0,0.0]

    self.FIX=[False,False]
    self.fixate_step=[0,0]
    self.fixate=np.array([[.0,.0],[.0,.0]])


    self.pos=np.array([[.0,.0],[.0,.0]])

    self.eye_status=self._get_status(self.EYE)
    self.hand_status=self._get_status(self.HAND)

    self.state = np.concatenate((self.target_pos,
                                self.eye_status,
                                self.hand_status),axis=None)


    # step 2: initial obs and belief
    # first obs and belief
    self.obs,self.obs_uncertainty=self._get_obs()
    self.belief,self.belief_uncertainty=self.obs, self.obs_uncertainty
    self.belief_full=self.state
    self.belief_full[0:2]=self.belief

    self.n_steps=0
    self.plot_target=False

    return self.belief_full

  def step(self,a):
    
    if a==0:
      self.action=[0,0]
    elif a==1:
      self.action=[0,1]
    else:
      self.action=[1,0]


    self.n_steps+=1

    self._state_transit(self.action[self.EYE],self.EYE)
    self._state_transit(self.action[self.HAND],self.HAND)

    self.eye_status=self._get_status(self.EYE)
    self.hand_status=self._get_status(self.HAND)

    self.state = np.concatenate((self.target_pos,self.eye_status,self.hand_status),axis=None)

    self.belief_full=self.state
    self.belief_full[0:2]=self.belief



    # check if the eye is within the target region
    dis_to_target=_calc_dis(self.target_pos, self.fixate[self.HAND])


    if  dis_to_target < self.fitts_W/2:
      print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
      done = True
      reward = 0
    else:
      done = False
      reward = -1

    if self.n_steps>self.max_steps:
      done=True


    nn=2
    info={'step': np.round(self.n_steps, nn),
    'target_pos':np.round(self.target_pos,nn),
      'aim_eye': np.round(self.moving_to[0],nn),
      'PREP_eye':self.PREP[0],
      'MOV_eye':self.MOV[0],
      'FIX_eye':self.FIX[0],
      'pos_eye':np.round(self.pos[0],nn),
      'aim_hand': np.round(self.moving_to[1],nn),
      'PREP_hand':self.PREP[1],
      'MOV_hand':self.MOV[1],
      'FIX_hand':self.FIX[1],
      'pos_hand':np.round(self.pos[1],nn),
      }


    return self.belief_full, reward, done, info
  

################################
  def _get_status(self,mode):
    status=np.concatenate((self.moving_to[mode],
                            self.vel[mode]),axis=None)

    return status

  def _get_obs(self):
    eccentricity=_calc_dis(self.target_pos,self.fixate[self.EYE])
    obs_uncertainty=eccentricity
    spatial_noise=np.random.normal(0, self.swapping_std*eccentricity, self.target_pos.shape)
    obs=self.target_pos + spatial_noise
    obs=np.clip(obs,-1,1)

    return obs,obs_uncertainty


  def _get_belief(self):
    z1,sigma1=self.obs,self.obs_uncertainty
    z2,sigma2=self.belief,self.belief_uncertainty
    
    # to avoid the following error
    # RuntimeWarning: invalid value encountered in double_scalars
    sigma1=max(0.001,sigma1)
    sigma2=max(0.001,sigma2)

    w1=sigma2**2/(sigma1**2+sigma2**2)
    w2=sigma1**2/(sigma1**2+sigma2**2)

    belief=w1*z1+w2*z2
    belief_uncertainty=np.sqrt( (sigma1**2+sigma2**2)/(sigma1**2 * sigma2**2))

    return belief, belief_uncertainty

  def _state_transit(self,action,mode):

    if action==1: # new command
      self.moving_to[mode]=self.belief_full[0:2]

      self.PREP[mode]=True
      self.prep_step[mode]=0

      self.START_MOVE[mode]=False
      self.MOV[mode]=False
      self.FIX[mode]=False



    if self.PREP[mode]:
      if self.prep_step[mode]==self.prep_duration[mode]:
        self.PREP[mode]=False
        self.START_MOVE[mode]=True
      
      self.prep_step[mode]+=self.time_step


    if self.START_MOVE[mode]:

      # intend pos and actual pos
      move_dis=_calc_dis(self.pos[mode],self.moving_to[mode])
      
      # motor noise
      stds=[self.ocular_std,self.motor_std] 
      noise=np.random.normal(0, stds[mode]*move_dis, (2,))
      self.actual_pos[mode]= self.moving_to[mode] + noise

      amp=_calc_dis(self.pos[mode],self.actual_pos[mode])*self.scale_deg
      
      trajectory,velocity=_vel_profiles(amp,mode,self.time_step)
      velocity=velocity/600 #normalise the speed, the max speed is 600 deg/s (as shown in the vel_model.py)
      
      
      pos=[]
      for r in (trajectory/amp):
            pos.append(self.pos[mode]+r*(self.actual_pos[mode]-self.pos[mode]))

      if mode==self.EYE:
        self.pos_e=[]
        self.pos_e.append(self.pos[mode])
        for r in (trajectory/amp):
            self.pos_e.append(self.pos[mode]+r*(self.actual_pos[mode]-self.pos[mode]))

        self.pos_e.append(self.actual_pos[mode])
        self.velocity_e=[0,*velocity,0]
        self.n_move_steps[mode]=len(self.velocity_e)

      else:
        self.pos_h=[]
        self.pos_h.append(self.pos[mode])
        
        for r in (trajectory/amp):
            self.pos_h.append(self.pos[mode]+r*(self.actual_pos[mode]-self.pos[mode]))
        self.pos_h.append(self.actual_pos[mode])
        self.velocity_h=[0,*velocity,0]
        self.n_move_steps[mode]=len(self.velocity_h)

      self.START_MOVE[mode]=False
      self.mov_step[mode]=0
      self.MOV[mode]=True
      self.vel=[0.0,0.0]

    

    if self.MOV[mode]:
    # update vel and pos
      if mode==self.EYE:
        self.vel[mode]=self.velocity_e[self.mov_step[mode]]
        self.pos[mode]=self.pos_e[self.mov_step[mode]]
      else:
        self.vel[mode]=self.velocity_h[self.mov_step[mode]]
        self.pos[mode]=self.pos_h[self.mov_step[mode]]

      self.mov_step[mode]+=1

      if self.mov_step[mode]==self.n_move_steps[mode]:
        self.MOV[mode]=False
        self.FIX[mode]=True
        self.fixate_step[mode]=0
        



    if self.FIX[mode]:
      self.fixate_step[mode]+=self.time_step
      if self.fixate_step[mode]>self.fixation_duration[mode]:
        self.fixate[mode]=self.actual_pos[mode]
        self.FIX[mode]=False
        if mode==self.EYE:
          self.obs,self.obs_uncertainty=self._get_obs()
          self.belief,self.belief_uncertainty=self._get_belief()





if __name__=="__main__":
  env=EyeHandEnv()
  obs=env.reset()
  actions=[2,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,2,0,0,0,1,0,0,0,0,0,2,0,0,0,0,0,0]



  for i in range(100):
    action=actions[i]
    observation, reward, done, info = env.step(action)
    plt.subplot(1,2,1)
    env.plot()
    plt.subplot(1,2,2)
    env.plot_1d()
    plt.savefig(f'fig')
    if done:
      break



