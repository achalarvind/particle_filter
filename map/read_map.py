import numpy as np
import time
from matplotlib import pyplot as plt


# get path to current file location
	path=os.path.split(os.path.realpath(__file__))
	path=path[0]
	path=os.path.join(path,'csv_data')

def read_map(file):
	# start_time = time.time()
	robot_specification=dict()
	read_map=False
	row=0
	for line in file:
		if line[:20].strip()=='robot_specifications':
			robot_specification[line[22:40].strip()]=float(line[41:])
		elif line[:10].strip()=='global_map':
			map_size = map(float,line[15:].split(' '))
			occupancy_grid=np.empty(map_size)
			read_map=True
		elif read_map==True:
			occupancy_grid[row]=[float(value) for value in line.split(' ') if value != '\n']
			row+=1
# end_time = time.time()
# print end_time-start_time
plt.imshow(occupancy_grid, interpolation='nearest')
plt.gray()

N = 10000
x = np.random.rand(N)*800
y = np.random.rand(N)*800
theta = 2*np.pi*np.random.rand(N)
robot_pose=np.array([x,y,theta])
plt.scatter(x, y, s=1, color=[1,0,0], alpha=0.5)
plt.show()