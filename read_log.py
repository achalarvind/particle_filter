import numpy as np

def read_map(environment_file):
	# start_time = time.time()
	robot_specification=dict()
	read_map=False
	row=0
	for line in environment_file:
		if line[:20].strip()=='robot_specifications':
			robot_specification[line[22:40].strip()]=float(line[41:])
		elif line[:10].strip()=='global_map':
			map_size = map(float,line[15:].split(' '))
			occupancy_grid=np.empty(map_size)
			read_map=True
		elif read_map==True:
			occupancy_grid[row]=[float(value) for value in line.split(' ') if value != '\n']
			row+=1
	return [robot_specification, occupancy_grid]

