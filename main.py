
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
    plt.title(f'last average (window=100)= {np.round(y[-1],3)}')
    #plt.show()
fitts_D=0.5
ocular_std=0.1 
motor_std=0.1

eta_eye=600
eta_hand=200
scale_deg=40
time_step=20
swapping_std=0.1

PREP,MOV,FIX=-1,0.5,1
timesteps = 4e6
save_feq_n=timesteps/10
for ocular_std in [0.05,0.1,0.2]:
    for fitts_W in [0.0125,0.05,0.1,0.2,0.25]:
        for run in range(10,13):       
        # Create log dir
            log_dir = f'./logs7/w{fitts_W}d{fitts_D}ocular{ocular_std}swapping{swapping_std}motor{motor_std}eta_eye{eta_eye}eta_hand{eta_hand}scale_deg{scale_deg}/run{run}/'
            log_dir2 = f'./logs7/w{fitts_W}d{fitts_D}ocular{ocular_std}swapping{swapping_std}motor{motor_std}eta_eye{eta_eye}eta_hand{eta_hand}scale_deg{scale_deg}/'
            os.makedirs(log_dir, exist_ok=True)
            TRAIN=True
            # Instantiate the env
            env = EyeHandEnv(fitts_W = fitts_W, 
                fitts_D=fitts_D, 
                ocular_std=ocular_std, 
                swapping_std=swapping_std, 
                motor_std=motor_std,
                eta_eye=eta_eye,
                eta_hand=eta_hand,
                scale_deg=scale_deg)

            env = Monitor(env, log_dir)

            # Train the agent
            model = PPO('MlpPolicy', env, verbose=0)

            # Save a checkpoint every 1000 steps
            checkpoint_callback = CheckpointCallback(save_freq=save_feq_n, save_path=f'{log_dir}savedmodel/',
                                                     name_prefix='eh_ppo_model')

            # Train the agent
            
            model.learn(total_timesteps=int(timesteps), callback=checkpoint_callback)

            plot_results2(log_dir)
            plt.savefig(f'{log_dir2}learning_curve{run}.png')
            plt.close('all') 

            print('Done training!!!!')

#############################################################################################
            # save the step data
            print('Saving Data!!!!')
            # Test the trained agent
            n_steps = 2000
            

            # for saving the data
            NN=20000
            learned_behav_data=np.ndarray(shape=(NN,16), dtype=np.float32)
            row=0
            eps=0
            
            with open(f'{log_dir}steps_data_verbose.csv', mode='w') as csv_file:
                
                while row<NN-1:
                    eps+=1
                    plt.close('all')
                    plt.figure()

                    obs = env.reset()
                    for step in range(n_steps):
                        # deterministic = True

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
                            info['vel_eye'],#13
                            info['vel_hand'],#14
                            eps]

           
                        row+=1
                        
                        fieldnames = info.keys()
                        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                        if step==0:
                            writer.writeheader()
                        writer.writerow(info)

                        if row==NN:
                            break

                        if done:
                            break
                    
            learned_behav_data=learned_behav_data[0:row]
            np.savetxt( f'{log_dir}steps_data.csv', learned_behav_data, delimiter=',') 


#############################################################################################
            
            print('Plotting!!!!')
            #my_data = genfromtxt(f'{log_dir}steps_data.csv', delimiter=',')
            my_data=learned_behav_data
            episodes=np.unique(my_data[:,-1])

            #plot the steps count
            n_steps_all=[]
            for e in episodes:
                data=my_data[:,-1]==e
                data_episode=my_data[data,:]
                n_steps=len(data_episode)
                n_steps_all.append(n_steps*time_step)

            plt.figure()
            plt.subplot(1,2,1)
            plt.hist(n_steps_all)
            plt.xlabel('Completion time (ms)')
            plt.subplot(1,2,2)
            plt.boxplot(n_steps_all)
            plt.ylabel('Completion time (ms)')
            plt.title(f'mean={np.round(np.mean(n_steps_all),2)} \n median={np.round(np.median(n_steps_all))}')
            plt.savefig(f'{log_dir2}/num_steps{run}.png')

            # plot each step
            for e in range(1,20):
                save_path=f'{log_dir}/plots/'
                os.makedirs(save_path, exist_ok=True)

                plt.close('all')
                plt.figure(figsize=(8,8))
                data=my_data[:,-1]==e
                data_episode=my_data[data,:]
                n_steps=len(data_episode)



                for step in range(n_steps):
                    target_pos=data_episode[step,1:3]
                    eye_stage=data_episode[step,5:6]
                    eye_pos=data_episode[step,6:8]

                    hand_stage=data_episode[step,10:11]
                    hand_pos=data_episode[step,11:13]

                    dis_eye=0.5-calc_dis(target_pos,eye_pos)
                    dis_hand=0.5-calc_dis(target_pos,hand_pos)
                    s=14
                    ss=10
                    


                    if step==0:
                        #Eye
                        plt.plot(-50,0,'ko',markersize=s,label='Eye Prep')
                        plt.plot(-50,0,'k<',label='Eye Moving',markersize=ss)
                        plt.plot(0,0,'k*',markersize=s,label='Eye Fixate')
                        #Eye
                        plt.plot(-50,0,'ro',markersize=s,label='Hand Prep')
                        plt.plot(-50,0,'r<',label='Hand Moving',markersize=ss)
                        plt.plot(5,0,'r*',markersize=s,label='Hand Fixate')
                        #Eye
                        plt.plot(-50,0,'ks',markersize=s,label='Eye None',markerfacecolor='w')
                        plt.plot(-50,0,'rs',markersize=s,label='Hand None',markerfacecolor='w')

                    time_step=20#ms
                    t=(step+1)*time_step

                    if eye_stage==PREP:
                        plt.plot(t,dis_eye,'ko',markersize=s,)
                    elif eye_stage==MOV:
                        plt.plot(t,dis_eye,'k<',markersize=ss)
                    elif eye_stage==FIX:
                        plt.plot(t,dis_eye,'k*',markersize=s,)
                    else:
                        plt.plot(t,dis_eye,'ks',markerfacecolor='w',markersize=ss)


                    if hand_stage==PREP:
                        plt.plot(t+5,dis_hand,'ro',markersize=s,)
                    elif hand_stage==MOV:
                        plt.plot(t+5,dis_hand,'r<',markersize=ss)
                    elif hand_stage==FIX:
                        plt.plot(t+5,dis_hand,'r*',markersize=s,)
                    else:
                        plt.plot(t+5,dis_hand,'rs',markerfacecolor='w',markersize=ss)


                plt.hlines(0.5,-time_step*2,(n_steps+3)*time_step,colors='b',linestyles='-',label='Target center')
                plt.hlines(-fitts_W/2+0.5,-time_step*2,(n_steps+3)*time_step,colors='g',linestyles='--',label='Target region')
                plt.xlim(-time_step,(n_steps+3)*time_step)
                plt.ylim(-0.02,0.6)
                plt.xticks(np.arange(0,(n_steps+3)*time_step,time_step),rotation='vertical')
                plt.xlabel('time (ms)')
                plt.ylabel(f'start ----> target (0.5= {scale_deg/2} degree)', fontsize=18)
                plt.legend(loc='lower right')
                plt.title(f'Target width and distance: W={fitts_W}, D={fitts_D} (1={scale_deg} degree) \n Noise: ocular={ocular_std}, perceptual={swapping_std} \n motor={motor_std}, \n For trajectory and speed profile: eta_eye={eta_eye} eta_hand={eta_hand}   ')
                plt.grid()
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





          


