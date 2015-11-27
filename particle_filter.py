import numpy as np
import os
import sys
import read_log
from matplotlib import pyplot as plt
import json
from functools import partial

global path
path=os.path.split(os.path.realpath(__file__))
path=path[0]

ENVIRONMENT_FILE = 'wean.dat'
ROBOT_LOG = 'robotdata1.log'

plt.ion()
class localization_test(object):
    def __init__(self, env_file, robot_log_file):
        # get environemnt information and robot log
        self._enviroment = open(os.path.join(path, 'map',env_file), 'r')
        self._log = open(os.path.join(path, 'log', robot_log_file), 'r')
        [self._robot_spec, self._occupancy_grid] = read_log.read_map(self._enviroment)
        self._filter = particle_filter('configuration.json', self._occupancy_grid)

        self._occupancy_grid = 1 - self._occupancy_grid 
        self._occupancy_grid[self._occupancy_grid > 1.1] = -1

        self._previous_robot_pose = None

    def parse_log(self, log_line):
        if log_line[0] == 'L':
            data = map(float,log_line[2:].split(' '))
            return ['L', data[0:3], data[3:6], data[6:-1]]
        elif log_line[0] == 'O':
            return ['O', map(float,log_line[2:].split(' '))[:-1]]
        else:
            return ['E',[]]

    def run_test(self,):
        for line in self._log:
            parsed_data = self.parse_log(line)
            if parsed_data[0] == 'L':
                [robot_pose, laser_pose, laser_data] = parsed_data[1:]
                self._filter.infer(laser_data, self._occupancy_grid)
            elif parsed_data[0] == 'O':
                [robot_pose]=parsed_data[1:]
                if self._previous_robot_pose == None:
                    self._previous_robot_pose = robot_pose
                    odometry = np.array([0.0, 0.0, 0.0])
                else:
                    odometry = np.subtract(self._previous_robot_pose, robot_pose)
                    self._previous_robot_pose = robot_pose
                print(odometry)
                self._filter.propogate(odometry)
            self.visualize()

    def visualize(self):
        plt.clf()
        plt.imshow(self._occupancy_grid, interpolation='nearest')
        plt.gray()
        plt.scatter(self._filter._particles[:,0]/10, self._filter._particles[:,1]/10, s=1, color=[1,0,0], alpha=0.5)
        plt.draw()

class robot(object):
    def __init__(self, configuration_file):
        with open(configuration_file) as cfg_file:    
            configuration = json.load(cfg_file)
        self._max_laser_reading = configuration['max_laser_reading']
        self._alpha1 = configuration['alpha1']
        self._alpha2 = configuration['alpha2']
        self._alpha3 = configuration['alpha3']
        self._alpha4 = configuration['alpha4']

        self._z_hit  = configuration['z_hit']
        self._z_rand = configuration['z_rand']
        self._z_short= configuration['z_short']
        self._z_max  = configuration['z_max']
        self._z_sigma= configuration['z_sigma']
        
        self._z_hit_norm = 1/(np.sum((1/np.sqrt(2*np.pi*self._z_sigma**2))*np.exp(-0.5*(np.array(range(-100, 101))**2)/self._z_sigma**2)))

        self._min_std_xy = configuration['min_std_xy']
        self._min_std_theta = configuration['min_std_theta']
        
    def move(self, robot_pose, odometry):
        """Computes pose increment
        
        Note1: assumes odometry is <dx, dy, dtheta>
        Note2: assumes robot pose is a 3x1 numpy array <x, y, heading>
        http://www.mrpt.org/tutorials/programming/odometry-and-motion-models/probabilistic_motion_models/"""
        # Transform odometry data from differences to sequential rot1, trans, rot2 motions
        dx,dy,dtheta = odometry
        trans = np.sqrt(dx**2 + dy**2)
        rot1 = np.arctan2(dy,dx) - dtheta
        rot2 = dtheta - rot1
        # Compute standard deviations of measurements
        sigma_trans = self._alpha3*trans + self._alpha4*(np.abs(rot1)+np.abs(rot2))
        sigma_rot1 = self._alpha1*np.abs(rot1) + self._alpha2*trans
        sigma_rot2 = self._alpha1*np.abs(rot2) + self._alpha2*trans
        # Add zero-mean Gaussian noise to odometry measurements?
        trans -= np.random.normal(0, sigma_trans**2,1) if sigma_trans > 0 else 0
        rot1 -= np.random.normal(0, sigma_rot1**2,1) if sigma_rot1 > 0 else 0
        rot2 -= np.random.normal(0, sigma_rot2**2,1) if sigma_rot2 > 0 else 0
        # Compute new robot pose
        new_pose = np.array([robot_pose[0] + trans*np.cos(robot_pose[2]+rot1),
                             robot_pose[1] + trans*np.sin(robot_pose[2]+rot1),
                             robot_pose[2] + rot1 + rot2])
        new_pose.shape = (3,)
        return new_pose

    def sense(self, robot_pose, sensor_reading, occupancy_grid):
        num_interp = 900

        sensor_pose = np.array([robot_pose[0] + 25*np.cos(robot_pose[2]), robot_pose[1] + 25*np.sin(robot_pose[2]), robot_pose[2]])

        point_pose_x = np.array(sensor_pose[0] + np.multiply(self._max_laser_reading, np.cos(sensor_pose[2] + np.pi*(np.array(range(0,180))-90.0)/180.0)))
        point_pose_y = np.array(sensor_pose[1] + np.multiply(self._max_laser_reading, np.sin(sensor_pose[2] + np.pi*(np.array(range(0,180))-90.0)/180.0)))

        point_pose_x = np.floor(np.array([np.linspace(miner, maxer, 900) for miner,maxer in zip(np.full_like(point_pose_x, sensor_pose[0]), point_pose_x)], dtype=int)/10)
        point_pose_y = np.floor(np.array([np.linspace(miner, maxer, 900) for miner,maxer in zip(np.full_like(point_pose_y, sensor_pose[1]), point_pose_y)], dtype=int)/10)
    
        #mask = np.multiply(np.multiply(point_pose_x >=0, point_pose_x < 800), np.multiply(point_pose_y >= 0, point_pose_y < 800))
        point_pose_x = np.array(np.clip(point_pose_x, 0, 799), dtype=int)
        point_pose_y = np.array(np.clip(point_pose_y, 0, 799), dtype=int)
            
        point_ranges = occupancy_grid[point_pose_x, point_pose_y]
        point_ranges[:, -1] = 1

        z_star = np.array([np.where(ranges > 0.01)[0][0] for ranges in point_ranges], dtype=int) 

        z_hit = np.array((self._z_hit_norm/np.sqrt(2*np.pi*self._z_sigma**2))*np.exp(-0.5*(np.array(z_star*10-np.array(sensor_reading))**2)/self._z_sigma**2))
        z_short = np.full_like(z_star, 0) #not implimented atm
        z_max = np.array(np.array(sensor_reading) == self._max_laser_reading, dtype=int)
        z_rand = np.full_like(z_star, 1/self._max_laser_reading)

        weights = np.array(self._z_hit*z_hit + self._z_short*z_short + self._z_max*z_max + self._z_rand*z_rand)


   #     print(z_star)

    #    plt.clf()
    #    plt.imshow(point_ranges)
     #   plt.gray()
      #  plt.scatter(z_star, np.array(range(0, 180)), s=1, color=[1,0,0], alpha=0.5)

#        plt.draw()
 #       plt.show(block=True) 

    
        return np.prod(weights)










"""       q = 1.0
        z_hit = 0.95
        z_rand = 0.05

        #Go through the sweep (lookup table method)
        point_pose = np.array(
                        [sensor_pose[0] + np.multiply(np.array(sensor_reading), np.cos(sensor_pose[2] + np.pi*(np.array(range(0,180))-90.0)/180.0)),
                         sensor_pose[1] + np.multiply(np.array(sensor_reading), np.sin(sensor_pose[2] + np.pi*(np.array(range(0,180))-90.0)/180.0))])
        point_pose = np.array(np.floor(point_pose/10), dtype=int)
        good_range = np.multiply(np.multiply(point_pose[0, :] >= 0, point_pose[0, :] < 800), np.multiply(point_pose[1, :] >= 0, point_pose[1, :] < 800))
   
        good_data  = np.multiply(np.array(sensor_reading) < self._max_laser_reading, good_range)
        scores = occupancy_grid[point_pose[0, good_data], point_pose[1, good_data]]
        scores[scores < 0] = 0
        q = np.prod(z_hit*(1-scores) + z_rand)
        print q    
        return q
"""

class particle_filter(object):
    def __init__(self, configuration_file, occupancy_grid):
        with open(configuration_file) as cfg_file:    
            configuration = json.load(cfg_file)
        self._robot_model = robot(configuration_file)
        self._no_particles = configuration['particle_count']
        self._resample_theshold = configuration['resample_threshold']

        #Gotta get them all in the good areas
        
        num_good_particles = 0
        self._particles = np.empty([0, 3])

        while(num_good_particles < self._no_particles):
            particles = np.transpose(np.array([np.random.rand(self._no_particles)*8000, np.random.rand(self._no_particles)*8000, 2*np.pi*np.random.rand(self._no_particles)]))
            good_points = occupancy_grid[np.array(np.floor(particles[:, 0]/10), dtype=int), np.array(np.floor(particles[:, 1]/10), dtype=int)] == 1
            particles = particles[good_points, :]
            self._particles = np.append(self._particles, particles, 0)
            num_good_particles = self._particles.shape[0]

        self._particles = self._particles[:, :self._no_particles]


        self._weights = [1.0/self._no_particles]*self._no_particles

    # def resample(self):
    #   if np.var(self._weights)<self._resample_theshold:
    #       self._particles = self._particles[np.random.choice(self._particles.shape[0], self._no_particles, p = self._weights, replace = True)]

    def propogate(self, odometry):
        print 'propogating'
        move_function = partial(int, base=2)
        self._particles = np.apply_along_axis(self._robot_model.move, 1, self._particles, odometry)

    def infer(self, sensor_reading, occupancy_grid):
        print 'infering'
        #print sensor_reading
        
        #get sensor readings for each robot pose
        particle_sensor_readings = np.apply_along_axis(self._robot_model.sense, 1, self._particles, sensor_reading, occupancy_grid) 
        # subtract the current sensor reading from the sensor readings of the particles, take the L2 norm and readjust weights
        self._weights = particle_sensor_readings  #np.divide(1.0, np.linalg.norm(np.subtract(particle_sensor_readings, sensor_reading), axis = 1))
        normalization_factor = sum(self._weights)
        self._weights = [float(weight)/normalization_factor for weight in self._weights]

        # get new particles by sampling the new distibution of weights
        self._particles = self._particles[np.random.choice(self._particles.shape[0], self._no_particles, p = self._weights, replace = True)]


if __name__ == '__main__':
    test_object = localization_test(ENVIRONMENT_FILE, ROBOT_LOG)
    test_object.visualize()
    test_object.run_test()
    # filter_object.imshow()
