
import numpy as np
import gym
from gym import spaces
import math
from envs.utils import get_trajectory, calc_dis, get_new_target
#from utils import get_trajectory, calc_dis, get_new_target
import matplotlib.pyplot as plt
import itertools



class EyeHandEnv(gym.Env):
  def __init__(self, fitts_W = 0.1, fitts_D=0.5, ocular_std=0.1, swapping_std=0.2, motor_std=0.1,eta_eye=600,eta_hand=300,scale_deg=20):
    super(EyeHandEnv,self).__init__()

    # define the constants
    self.EYE,self.HAND=0,1 
    self.PREP,self.MOV,self.FIX=-1,0.5,1

    # task setting
    # the canvas is [-1,1]
    # 1.0 in the canvas equals to 20 degress 
    # This is used to calculate the velocity profile
    self.scale_deg=scale_deg
    self.fitts_W=fitts_W
    self.fitts_D=fitts_D
    # agent visual spatial noise
    self.swapping_std=swapping_std
    # agent ocular and motor noise 
    self.motor_noise_params=np.array([ocular_std,motor_std])
    self.eta=np.array([eta_eye,eta_hand])


    # timings (unit: self.time_step)
    # the time step for the controller (unit ms)
    self.time_step=int(20)
    # one for the eye and one for the hand
    self.prep_duration=[self.time_step, self.time_step]
    self.fixation_duration=[self.time_step,self.time_step]

    self.max_steps=2000
    ###############################################################
    # the action space and observation space
    self.actions=np.array([[0,0],[1,0],[0,1]])
    self.action_space = spaces.Discrete(3)

    # belief/state
    low_b=np.array([-1.0, -1.0, # target_pos_belief
      0, # belief uncertainty
      -1.0, -1.0], # eye and hand stage
      dtype=np.float32)
    
    high_b=np.array([1.0, 1.0, # target_pos_belief
      1, # belief uncertainty
      1.0,1.0, # eye and hand stage
      ], 
      dtype=np.float32)

    self.observation_space = spaces.Box(low=low_b, high=high_b, dtype=np.float32)

  def reset(self):
    # the stage for the eye and hand
    self.stage=[0,0]

    # the following is used to record the progress for each stage
    self.prep_step=[0,0]
    self.mov_step=[0,0]
    self.n_move_steps=[0,0]
    self.fixate_step=[0,0]

    self.aim_at=np.array([[.0,.0],[.0,.0]])
    # the eye and hand start position
    self.current_pos=np.array([[.0,.0],[.0,.0]])
    self.current_vel=[0,0]
    self.fixate=np.array([[.0,.0],[.0,.0]])

    self.n_steps=0


    # initialize the state
    self.target_pos = get_new_target(self.fitts_D)
    


    

    # first estimate of where the target is
    self.tgt_obs,self.tgt_obs_uncertainty=self._get_tgt_obs()
    self.tgt_belief,self.tgt_belief_uncertainty=self._get_tgt_obs()
    # the state of the environment includes three elements: target, eye, hand
    self._get_state_observation()

    return self.observation

  def step(self,a):
    self.n_steps+=1
    self.chosen_action=self.actions[a]

    self._state_transit(self.chosen_action[self.EYE],self.eta[self.EYE],self.EYE)
    self._state_transit(self.chosen_action[self.HAND],self.eta[self.HAND],self.HAND)


    self._get_state_observation()

    # check if the hand is within the target region
    dis_to_target=calc_dis(self.target_pos, self.fixate[self.HAND])
    if  dis_to_target < self.fitts_W/2:
      done = True
      reward = 0
    else:
      done = False
      reward = -1

    if self.n_steps>self.max_steps:
      done=True

    info=self._save_data()




    return self.observation, reward, done, info
  

  ################################

  def _get_state_observation(self):
    self.target_status=np.concatenate((self.tgt_belief,self.tgt_belief_uncertainty),axis=None)
    # these assignments are trivial, but as place holder for more complex eye and hand status
    self.eye_status=self.stage[self.EYE]
    self.hand_status=self.stage[self.HAND]
    self.state = np.concatenate((self.target_status,self.eye_status,self.hand_status),axis=None)
    self.observation=self.state


  def _get_tgt_obs(self):
    eccentricity=calc_dis(self.target_pos,self.fixate[self.EYE])
    
    spatial_noise=np.random.normal(0, self.swapping_std*eccentricity, self.target_pos.shape)
    obs=self.target_pos + spatial_noise
    obs=np.clip(obs,-1,1)

    obs_uncertainty=eccentricity

    return obs,obs_uncertainty


  def _get_tgt_belief(self):
    z1,sigma1=self.tgt_obs,self.tgt_obs_uncertainty
    z2,sigma2=self.tgt_belief,self.tgt_belief_uncertainty

    # to avoid the following error
    # RuntimeWarning: invalid value encountered in double_scalars

    sigma1=max(0.0001,sigma1)
    sigma2=max(0.0001,sigma2)
 
    w1=sigma2**2/(sigma1**2+sigma2**2)
    w2=sigma1**2/(sigma1**2+sigma2**2)

    belief=w1*z1+w2*z2
    belief_uncertainty=np.sqrt( (sigma1**2 * sigma2**2)/(sigma1**2 + sigma2**2))

    return belief, belief_uncertainty

  def _state_transit(self,action,eta,mode):

    if action==1: 
      # new command
      self.stage[mode]=self.PREP
      self.prep_step[mode]=0
      self.aim_at[mode]=self.observation[0:2]

    if self.stage[mode]==self.PREP:
      self.prep_step[mode]+=self.time_step
      
      if self.prep_step[mode]>self.prep_duration[mode]:
        # done with the prep, ready to move
        self.stage[mode]=self.MOV
        self.mov_step[mode]=0
        
        # generate the trajectory
        move_dis=calc_dis(self.current_pos[mode],self.aim_at[mode])
        # motor noise is dependent on the moving distance 
        noise=np.random.normal(0, self.motor_noise_params[mode]*move_dis, (2,))
        actual_pos= self.aim_at[mode] + noise

        amp=calc_dis(self.current_pos[mode],actual_pos)*self.scale_deg

        pos,vel=get_trajectory(eta,amp,self.current_pos[mode],actual_pos,self.time_step) 
        
        self.n_move_steps[mode]=len(pos)
        

        if mode==self.EYE:
          self.pos_e=pos 
          self.vel_e=vel 
        else:
          self.pos_h=pos
          self.vel_h=vel
        
      

    if self.stage[mode]==self.MOV:
      
      if self.mov_step[mode]<self.n_move_steps[mode]:
        # update pos
        if mode==self.EYE:
          self.current_pos[mode]=self.pos_e[self.mov_step[mode]]
          self.current_vel[mode]=self.vel_e[self.mov_step[mode]]
        else:
          self.current_pos[mode]=self.pos_h[self.mov_step[mode]]
          self.current_vel[mode]=self.vel_h[self.mov_step[mode]]
      else:
        # finish moving, start fixate
        self.stage[mode]=self.FIX
        self.fixate_step[mode]=0
 

      self.mov_step[mode]+=1




    if self.stage[mode]==self.FIX:
      self.fixate_step[mode]+=self.time_step

      if self.fixate_step[mode]>=self.fixation_duration[mode]:
        self.fixate[mode]=self.current_pos[mode]
        if mode==self.EYE:
          self.tgt_obs,self.tgt_obs_uncertainty=self._get_tgt_obs()
          self.tgt_belief,self.tgt_belief_uncertainty=self._get_tgt_belief()




    


  def _save_data(self):
    nn=4
    info={'step': np.round(self.n_steps, nn),
    'target_pos':np.round(self.target_pos,nn),
    'aim_eye': np.round(self.aim_at[0],nn),
    'stage_eye':self.stage[0],
    'pos_eye':np.round(self.current_pos[0],nn),
    'aim_hand': np.round(self.aim_at[1],nn),
    'stage_hand':self.stage[1],
    'pos_hand':np.round(self.current_pos[1],nn),
    'vel_eye':np.round(self.current_vel[0],nn),
    'vel_hand':np.round(self.current_vel[1],nn),
    }
    return info



if __name__=="__main__":

  prep1=True
  move1=True
  fix1=True

  prep2=True
  move2=True
  fix2=True

  env=EyeHandEnv()
  test_actions=[1,2,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0]
  obs=env.reset()
  for i in range(len(test_actions)):
    action=test_actions[i]
    observation, reward, done, info = env.step(action)
    dis=calc_dis(info['target_pos'],info['pos_eye'])
    dis_hand=calc_dis(info['target_pos'],info['pos_hand'])


    print(info)

    stage_eye=info['stage_eye']
    stage_hand=info['stage_hand']

    plt.subplot(1,2,1)

    plt.plot(0,0.5,'k+',markersize=15)
    



    if stage_eye==-1:
      plt.plot(i+1,dis,'ko',markersize=15)
      if prep1:
        plt.plot(i+1,dis,'ko',label='eye_prep',markersize=15)
        prep1=False
    elif stage_eye==0.5:
      plt.plot(i+1,dis,'k>')
      if move1:
        plt.plot(i+1,dis,'k>',label='eye_moving')
        move1=False
    elif stage_eye==1:
      plt.plot(i+1,dis,'k+',markersize=15)
      if fix1:
        plt.plot(i+1,dis,'k+',label='eye_fix',markersize=15)
        fix1=False
    else:
      plt.plot(i+1,dis,'gd',markersize=15,label='null')

    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(0,0.5,'r+',markersize=15)

    if stage_hand==-1:
      plt.plot(i+1,dis_hand,'ro',markersize=15)
      if prep2:
        plt.plot(i+1,dis_hand,'ro',label='hand_prep',markersize=15)
        prep2=False

    elif stage_hand==0.5:
      plt.plot(i+1,dis_hand,'r>')
      if move2:
        plt.plot(i+1,dis_hand,'r>',label='hand_moving')
        move2=False

    elif stage_hand==1:
      plt.plot(i+1,dis_hand,'r+',markersize=15)
      if fix2:
        plt.plot(i+1,dis_hand,'r+',label='hand_fix',markersize=15)
        fix2=False
    else:
      plt.plot(i+1,dis_hand,'gd',markersize=15,label='null')

    
    if done:
      break
  plt.legend()
  plt.show()



