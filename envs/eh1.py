
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

class EH1(gym.Env):
	def __init__(self,fitts_W = 0.2, fitts_D=0.5, ocular_std=0.1, swapping_std=0.1, motor_std=0.1):
		super(EH1,self).__init__()

		# task setting
		self.scale_deg=20 # 1.0 in the cavas equals to 20 degress

		self.fitts_W=fitts_W
		self.fitts_D=fitts_D
		# agent ocular motor noise and visual spatial noise
		self.ocular_std=ocular_std
		self.swapping_std=swapping_std
		self.motor_std=motor_std

		self.max_steps=10

		self.time_step=int(20) # 50 ms
		self.time_prep_eye=self.time_step # how many ms
		self.time_prep_hand=self.time_step*2
		self.time_fixation=self.time_step*2

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
		self.prep_eye=0
		self.fixate_eye=0

		self.vel_eye=0.0
		self.eye_status=np.concatenate((self.pos_eye,self.prep_eye,self.moving_to_eye,self.fixate_eye,self.vel_eye),axis=None)


		# (3) hand
		self.pos_hand=np.array([0.0,0.0])
		self.moving_to_hand=np.array([0.0,0.0])
		self.prep_hand=0.0

		self.fixate_hand=0

		self.vel_hand=0.0
		self.hand_status=np.concatenate((self.pos_hand,self.prep_hand,self.moving_to_hand,self.fixate_hand,self.vel_hand),axis=None)

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

		self.n_steps=0

		self.PREP=False
		self.START_MOVE=False
		self.MOV=False
		self.FIX=False
		



		return self.belief_full

	def step(self,action):
		self.n_steps+=1

		self._state_transit(action)
		


		self.belief_full={}
		reward={}
		done={}
		



		if self.n_steps>self.max_steps:
			done=True

		info={}
		return self.belief_full, reward, done, info

	################################
	def _get_obs(self):
		eccentricity=_calc_dis(self.target_pos,self.fixate)
		obs_uncertainty=eccentricity
		spatial_noise=np.random.normal(0, self.swapping_std*eccentricity, self.target_pos.shape)
		obs=self.target_pos + spatial_noise
		obs=np.clip(obs,-1,1)

		return obs,obs_uncertainty

	def _state_transit(self,action):


		# execute the action: move the eye and/or move the hand
		eye_move_amp=_calc_dis(action[0:2],self.moving_to_eye)
		hand_move_amp=_calc_dis(action[2:4],self.moving_to_hand)

		print(f'eye_move_amp={eye_move_amp}')
		print(f'hand_move_amp={hand_move_amp}')


		# new command
		if eye_move_amp>0.1:
			self.moving_to_eye=action[0:2]
			self.moving_to_hand=action[2:4]
			print('New eye movement')
			self.PREP=True
			self.START_MOVE=False
			self.MOV=False
			self.FIX=False
			


		if self.PREP:
			self.prep_eye+=self.time_step
			print(f'eye prep: {np.round(self.prep_eye/self.time_prep_eye,2)*100} %')
			
			if self.prep_eye==self.time_prep_eye:
				self.PREP=False
				self.START_MOVE=True


		if self.START_MOVE:
			self.mov_step=0
			print('Start Moving')

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
			print(f'n_move_steps={self.n_move_steps}')

			self.START_MOVE=False
			self.MOV=True

			'''
			print(n_move_steps)
			print(f'start={self.pos_eye}')
			print(f'end={end_eye}')
			print(pos)
			'''
		

		if self.MOV:
			print(f'mov_step={self.mov_step} ')

			# update vel and pos
			self.vel_eye=self.velocity_e[self.mov_step]
			self.pos_eye=self.pos_e[self.mov_step]
			self.mov_step+=1

			if self.mov_step==self.n_move_steps:
				self.MOV=False
				self.FIX=True
				self.fix_step=0

			

		if self.FIX:
			self.fixate=self.end_eye

			self.fix_step+=self.time_step
			print(f'Fixating: {np.round(self.fix_step/self.time_fixation,2)*100} %')

			if self.fix_step==self.time_fixation:
				self.FIX=False
				print('Info Extract!')
				self.obs,self.obs_uncertainty=self._get_obs()
				# update belief









if __name__=="__main__":
	env=EH1()
	obs=env.reset()

	action=env.action_space.sample()
	print(f'action={action}')
	for i in range(10):
		print('--------------------------------------')
		observation, reward, done, info = env.step(action)
		if done:
			break

