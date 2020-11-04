import numpy as np
import math
import matplotlib.pyplot as plt

# some tool functions
def calc_dis(p,q):
    #calculate the Euclidean distance between points p and q 
    return np.sqrt(np.sum((p-q)**2))
# task


def get_new_target(D):
    '''
    generate a target at a random angle, distance D away.
    '''
    angle=np.random.uniform(0,math.pi*2) 
    x_target=math.cos(angle)*D
    y_target=math.sin(angle)*D
    return np.array([x_target,y_target])


D=0.5
scale=1
for i in range(100):
    target_pos=get_new_target(D)
    if i==0:
    	plt.plot(target_pos[0]*scale,target_pos[1]*scale,'ro',label='Target')
    else:
    	plt.plot(target_pos[0]*scale,target_pos[1]*scale,'ro')



plt.plot(0,0,'gs',label='Start')

plt.legend()
plt.xlabel('degree')
plt.ylabel('degree')
plt.savefig('task.png')




plt.figure()
fig, ax = plt.subplots()

target_pos=np.array([0,0.5])
fixate=[0,0]
fitts_W=0.3
plt.plot(target_pos[0],target_pos[1],'r+',label=f'target_fitts_W={fitts_W}')

circle1 = plt.Circle((target_pos[0],target_pos[1]), fitts_W/2, color='r')

 # note we must use plt.subplots, not plt.subplot
# (or if you have an existing figure)
# fig = plt.gcf()
# ax = fig.gca()

ax.add_artist(circle1)
plt.plot(0,0,'gs',label='Start',markersize=15)
plt.plot(fixate[0],fixate[1],'k*',label='fixate',markersize=15)
eccentricity=calc_dis(target_pos,fixate)
swapping_std=0.2
for i in range(200):
	spatial_noise=np.random.normal(0, swapping_std*eccentricity, target_pos.shape)
	obs=target_pos + spatial_noise
	if i==0:
		plt.plot(obs[0],obs[1],'k.',color='gray',label=f'obs(swapping={swapping_std})')
	else:
		plt.plot(obs[0],obs[1],'k.',color='gray')

plt.xlim(-0.4,0.4)
plt.ylim(-0.1,0.7)
plt.legend()
plt.xlabel('0.5 is 10 degree')
plt.ylabel('0.5 is 10 degree')
plt.title(f'Target position estimation (swapping={swapping_std}fitts_W={fitts_W})')
plt.savefig('perceptualnoise.png')

