
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

class EH1(gym.Env):
	def __init__(self,fitts_W = 0.2, fitts_D=0.5, ocular_std=0.1, swapping_std=0.1, motor_std=0.1):
		super(EH1,self).__init__()

		# task setting
		self.fitts_W=fitts_W
		self.fitts_D=fitts_D
		# agent ocular motor noise and visual spatial noise
		self.ocular_std=ocular_std
		self.swapping_std=swapping_std
		self.motor_std=motor_std

		self.action_space = spaces.Box(low=-1, high=1, shape=(4, ), dtype=np.float32)

	def reset(self):
		# STEP1:  initialize the state
		# the state of the env includes three elements: target, eye, hand
		# (1) target
		angle=np.random.uniform(0,math.pi*2) 
		x_target=math.cos(angle)*self.fitts_D
		y_target=math.sin(angle)*self.fitts_D
		self.target_pos = np.array([x_target,y_target])


		# (2) eye
		self.pos_eye=np.array([0.0,0.0])
		self.moving_to_eye=np.array([0.0,0.0])
		self.stage_eye=0
		self.vel_eye=0.0
		self.eye_status=np.concatenate((self.pos_eye,self.moving_to_eye,self.stage_eye,self.vel_eye),axis=None)


		# (3) hand
		self.pos_hand=np.array([0.0,0.0])
		self.moving_to_hand=np.array([0.0,0.0])
		self.stage_hand=0
		self.vel_hand=0.0
		self.hand_status=np.concatenate((self.pos_hand,self.moving_to_hand,self.stage_hand,self.vel_hand),axis=None)

		self.state = np.concatenate((self.target_pos,
                              self.eye_status,
                              self.hand_status),axis=None)


		# step 2: initial obs and belief
		# first obs and belief
		self.fixate=np.array([0,0])
		self.obs,self.obs_uncertainty=self._get_obs()

		self.belief,self.belief_uncertainty=self.obs, self.obs_uncertainty
		self.belief_full=self.state
		self.belief_full[0:2]=self.belief



		return self.belief_full

	def step(self,action):
		pass

	################################
	def _get_obs(self):
		eccentricity=_calc_dis(self.target_pos,self.fixate)
		obs_uncertainty=eccentricity
		spatial_noise=np.random.normal(0, self.swapping_std*eccentricity, self.target_pos.shape)
		obs=self.target_pos + spatial_noise
		obs=np.clip(obs,-1,1)

		return obs,obs_uncertainty


if __name__=="__main__":
	env=EH1()
	obs=env.reset()

	done =False
	while not done:
		action=env.action_space.sample()
		print(f'action={action}')
		xxx

