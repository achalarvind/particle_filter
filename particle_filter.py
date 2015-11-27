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
    def __init__(self, test_filter, env_file, robot_log_file):
        # get environemnt information and robot log
        self._enviroment = open(os.path.join(path, 'map',env_file), 'r')
        self._log = open(os.path.join(path, 'log', robot_log_file), 'r')
        [self._robot_spec, self._occupancy_grid] = read_log.read_map(self._enviroment)
        self._filter = test_filter
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
                self._filter.propogate(odometry)
            self.visualize()

    def visualize(self):
        plt.clf()
        plt.imshow(self._occupancy_grid, interpolation='nearest')
        plt.gray()
        plt.scatter(self._filter._particles[:,0], self._filter._particles[:,1], s=1, color=[1,0,0], alpha=0.5)
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
        sensor_pose = np.array([robot_pose[0] + 30*np.cos(robot_pose[2]), robot_pose[1] + 30*np.sin(robot_pose[2]), robot_pose[2]])
        q = 1.0
        z_hit = 0.95
        z_rand = 0.05

        #Go through the sweep (lookup table method)
        point_pose = np.array(np.floor(np.array(
                        [sensor_pose[0] + sensor_reading*np.cos(sensor_pose[2] + np.pi*(np.array(range(0,180))-90.0)/180.0),
                         sensor_pose[1] + sensor_reading*np.sin(sensor_pose[2] + np.pi*(np.array(range(0,180))-90.0)/180.0)])/10), dtype=int)
        

        good_range = np.multiply(np.multiply(point_pose[0] >= 0, point_pose[0] < 800), np.multiply(point_pose[1] >= 0, point_pose[1] < 800))
        good_data  = sensor_reading < self._max_laser_reading
        scores = occupancy_grid[point_pose[0], point_pose[1]]
   
        good_scores = scores[np.multiply(good_range, np.multiply(good_data, scores > 0))]

        q = np.prod(z_hit*good_scores + z_rand)
        
        return q


class particle_filter(object):
    def __init__(self, configuration_file):
        with open(configuration_file) as cfg_file:    
            configuration = json.load(cfg_file)
        self._robot_model = robot(configuration_file)
        self._no_particles = configuration['particle_count']
        self._resample_theshold = configuration['resample_threshold']
        self._particles = np.transpose(np.array([np.random.rand(self._no_particles)*800, np.random.rand(self._no_particles)*800, 2*np.pi*np.random.rand(self._no_particles)]))
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
    filter_object = particle_filter('configuration.json')
    test_object = localization_test(filter_object, ENVIRONMENT_FILE, ROBOT_LOG)
    test_object.visualize()
    test_object.run_test()
    # filter_object.imshow()
