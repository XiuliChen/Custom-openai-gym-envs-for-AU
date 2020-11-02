
import os
import csv

import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback



from envs.EyeHandEnv import EyeHandEnv
from envs.utils import calc_dis
from numpy import genfromtxt

import glob
from PIL import Image

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results2(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=100)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    #plt.show()

PREP,MOV,FIX=-1,0.5,1
timesteps = 2e6
save_feq_n=timesteps/10
for run in range(5):
    for swapping_std in [0.2,0.1]:
        for fitts_W in [0.3,0.05,0.2,0.1]:       
            # Create log dir
            log_dir = f"./logs/w{fitts_W*100}timesteps{int(timesteps)}swapping_std{swapping_std}run{run}/"
            os.makedirs(log_dir, exist_ok=True)
            TRAIN=True
            # Instantiate the env
            env = EyeHandEnv(fitts_W = fitts_W, fitts_D=0.5, ocular_std=0.1, swapping_std=swapping_std, motor_std=0.1)
            env = Monitor(env, log_dir)

            # Train the agent
            model = PPO('MlpPolicy', env, verbose=0)

            # Save a checkpoint every 1000 steps
            checkpoint_callback = CheckpointCallback(save_freq=save_feq_n, save_path=f'{log_dir}savedmodel/',
                                                     name_prefix='eh_ppo_model')

            # Train the agent
            
            model.learn(total_timesteps=int(timesteps), callback=checkpoint_callback)

            plot_results2(log_dir)
            plt.savefig(log_dir+'learning_curve.png')
            plt.close('all') 

            print('Done training!!!!')

#############################################################################################
            # save the step data
            print('Saving Data!!!!')
            # Test the trained agent
            n_steps = 2000
            

            # for saving the data
            NN=10000
            learned_behav_data=np.ndarray(shape=(NN,14), dtype=np.float32)
            row=0
            eps=0
            
            with open(f'{log_dir}steps_data_verbose.csv', mode='w') as csv_file:
                
                while row<NN-1:
                    eps+=1
                    plt.close('all')
                    plt.figure()

                    obs = env.reset()
                    for step in range(n_steps):

                        action, _ = model.predict(obs)
                        obs, reward, done, info = env.step(action)
                        info['eps']=eps
                        
                        

                        learned_behav_data[row,:]=[step+1,#0
                            info['target_pos'][0],#1
                            info['target_pos'][1],#2
                            info['aim_eye'][0],#3
                            info['aim_eye'][1],#4
                            info['stage_eye'],#5
                            info['pos_eye'][0],#6
                            info['pos_eye'][1],#7
                            info['aim_hand'][0],#8
                            info['aim_hand'][1],#9
                            info['stage_hand'],#10
                            info['pos_hand'][0],#11
                            info['pos_hand'][1],#12
                            eps]

           
                        row+=1

                        if row==NN:
                            break

                        if done:
                            break

                        fieldnames = info.keys()
                        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                        if step==0:
                            writer.writeheader()
                        writer.writerow(info)
                    
            learned_behav_data=learned_behav_data[0:row]
            np.savetxt( f'{log_dir}steps_data.csv', learned_behav_data, delimiter=',') 


#############################################################################################
            
            print('Making Gif!!!!')
            #my_data = genfromtxt(f'{log_dir}steps_data.csv', delimiter=',')
            my_data=learned_behav_data
            episodes=np.unique(my_data[:,-1])
            for e in range(1,20):
                save_path=f'{log_dir}/plots/'
                os.makedirs(save_path, exist_ok=True)

                plt.close('all')
                plt.figure(figsize=(7,7))
                data=my_data[:,-1]==e
                data_episode=my_data[data,:]
                n_steps=len(data_episode)

                

                for step in range(n_steps):
                    target_pos=data_episode[step,1:3]
                    eye_stage=data_episode[step,5:6]
                    eye_pos=data_episode[step,6:8]
                    
                    hand_stage=data_episode[step,10:11]
                    hand_pos=data_episode[step,11:13]

                    dis_eye=calc_dis(target_pos,eye_pos)
                    dis_hand=calc_dis(target_pos,hand_pos)

                    if step==0:
                        #Eye
                        plt.plot(-5,0.5,'ko',markersize=15,label='Eye Prep')
                        plt.plot(-5,0.5,'k<',label='Eye Moving')
                        plt.plot(0,0.5,'k+',markersize=15,label='Eye Fixate')
                        #Eye
                        plt.plot(-5,0.5,'ro',markersize=15,label='Hand Prep')
                        plt.plot(-5,0.5,'r<',label='Hand Moving')
                        plt.plot(0,0.5,'r+',markersize=15,label='Hand Fixate')
                        #Eye
                        plt.plot(-5,0.5,'gs',markersize=15,label='None')

                    t=step+1

                    if eye_stage==PREP:
                        plt.plot(t,dis_eye,'ko',markersize=15,)
                    elif eye_stage==MOV:
                        plt.plot(t,dis_eye,'k<')
                    elif eye_stage==FIX:
                        plt.plot(t,dis_eye,'k+',markersize=15,)
                    else:
                        plt.plot(t,dis_eye,'gs')


                    if hand_stage==PREP:
                        plt.plot(t+0.1,dis_hand,'ro',markersize=15,)
                    elif hand_stage==MOV:
                        plt.plot(t+0.1,dis_hand,'r<')
                    elif hand_stage==FIX:
                        plt.plot(t+0.1,dis_hand,'r+',markersize=15,)
                    else:
                        plt.plot(t+0.1,dis_hand,'gs')

                plt.xlim(-1,n_steps+3)
                plt.ylim(-0.1,0.6)
                plt.ylabel('Distance to target')
                plt.legend(loc='lower left')

                plt.savefig(f'{save_path}/eps{e}.png')
                '''
                # filepaths
                fp_in = f"{save_path}/*.png"
                fp_out = f"{save_path}/dis.gif"

                # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
                img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
                img.save(fp=fp_out, format='GIF', append_images=imgs,
                        save_all=True, duration=300, loop=1)
                '''





          


