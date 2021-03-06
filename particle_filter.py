import numpy as np
import os
import sys
import read_log
import json
from functools import partial
import multiprocessing 
import cPickle as pickle
import cv2

create_movies = False
import matplotlib
if create_movies:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
if create_movies:
    import matplotlib.animation as manimation
    fig = plt.figure(0)
    writer = manimation.FFMpegWriter(fps=10)
    
def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    if func_name.startswith('__') and not func_name.endswith('__'): #deal with mangled names
        cls_name = cls.__name__.lstrip('_')
        func_name = '_' + cls_name + func_name
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.__mro__:
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

import copy_reg
import types
copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)

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
        total_length = 2281 # length of log1
        for index, line in enumerate(self._log):
            print '{0}/{1} log line'.format(index,total_length)
            parsed_data = self.parse_log(line)
            if parsed_data[0] == 'L':
                [robot_pose, laser_pose, laser_data] = parsed_data[1:]
                self._filter.infer(laser_data, self._occupancy_grid)
            elif parsed_data[0] == 'O':
                [robot_pose]=parsed_data[1:]
                if self._previous_robot_pose == None:
                    odometry = np.array([0.0, 0.0, 0.0])
                    dx,dy,dtheta = odometry
                else:
                    dx,dy,dtheta = np.subtract(robot_pose, self._previous_robot_pose)
                    if not self._filter.started_moving and (dx != 0 or dy != 0):
                        print 'We started moving! dx: {0}, dy: {1}'.format(dx,dy)
                        self._filter.started_moving = True
                    # Transform odometry data from differences to sequential rot1, trans, rot2 motions
                self._previous_robot_pose = robot_pose
                trans = np.sqrt(dx**2 + dy**2)
                rot1 = np.arctan2(dy,dx) - self._previous_robot_pose[2]
                rot2 = dtheta - rot1
                odometry = np.array([trans,rot1,rot2])
                print 'odometry:', [dx,dy,dtheta]
                self._filter.propogate(odometry)
            self.visualize()

    def visualize(self):
        img = np.zeros([800, 800, 3])
        img[:, :, 0] = self._occupancy_grid
        img[:, :, 1] = self._occupancy_grid
        img[:, :, 2] = self._occupancy_grid
        for pnt in self._filter._robot_model._pnt_red:
            pnt = np.clip(pnt, 0, 7999)
            img[int(pnt[0]/10), int(pnt[1]/10), :] = np.array([1, 0, 0]) 

        for pnt in self._filter._robot_model._pnt_grn:
            pnt = np.clip(pnt, 0, 7999)
            img[int(pnt[0]/10), int(pnt[1]/10), :] = np.array([0, 1, 0]) 
        
        for pnt in self._filter._robot_model._pnt_blu:
            pnt = np.clip(pnt, 0, 7999)
            img[int(pnt[0]/10), int(pnt[1]/10), :] = np.array([0, 0, 1])

        for part in self._filter._particles:
            part = np.clip(part, 0, 7999)
            cv2.circle(img, (int(part[1]/10), int(part[0]/10)), 3, [1, 0, 1])

 
        self._filter._robot_model._pnt_grn = np.empty([0, 2])
        self._filter._robot_model._pnt_blu = np.empty([0, 2])
        self._filter._robot_model._pnt_red = np.empty([0, 2])

        cv2.imshow("Output Plot", img)
        cv2.waitKey(1)

#        plt.clf()
#        plt.imshow(self._occupancy_grid.T, interpolation='nearest')
#        plt.gray()
#        plt.scatter(self._filter._particles[:,0]/10, self._filter._particles[:,1]/10, s=1, color=[1,0,0], alpha=0.5)
        #plt.quiver(self._filter._particles[:,0]/10, self._filter._particles[:,1]/10, np.cos(self._filter._particles[:,2]), np.sin(self._filter._particles[:,2]),
        #   units='xy', scale=10., zorder=3, color='blue',
        #zs   width=0.007, headwidth=3., headlength=4.)
#        plt.axis([0, 800, 0, 800])
#        plt.draw()
#        if create_movies:
#            writer.grab_frame()

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

#        self._num_interp = 900
#        self._linspace_array = np.mat(np.linspace(0, 1, self._num_interp))

        self._pnt_grn = np.empty([0, 2])
        self._pnt_blu = np.empty([0, 2])
        self._pnt_red = np.empty([0, 2])



        self._z_dic = {}
        with open('point_dic.dic', 'rb') as fp:
            print "Starting dic read"
            self._z_dic = pickle.load(fp)
            print "Read in dict file!"

        self._min_std_xy = configuration['min_std_xy']
        self._min_std_theta = configuration['min_std_theta']
        
    def move(self, particles, odometry):
        """Computes pose increment
        
        Note1: assumes odometry is <dx, dy, dtheta>
        Note2: assumes robot pose is a 3x1 numpy array <x, y, heading>
        http://www.mrpt.org/tutorials/programming/odometry-and-motion-models/probabilistic_motion_models/"""
        num_particles = particles.shape[0]
        trans,rot1,rot2 = odometry
        # Compute standard deviations of measurements
        sigma_trans = self._alpha3*trans + self._alpha4*(np.abs(rot1)+np.abs(rot2))
        sigma_rot1 = self._alpha1*np.abs(rot1) + self._alpha2*trans
        sigma_rot2 = self._alpha1*np.abs(rot2) + self._alpha2*trans
        # Enforce minimums on std dev
        sigma_trans = sigma_trans if sigma_trans > self._min_std_xy else self._min_std_xy
        sigma_rot1 = sigma_rot1 if sigma_rot1 > self._min_std_theta else self._min_std_theta
        sigma_rot2 = sigma_rot2  if sigma_rot2 > self._min_std_theta else self._min_std_theta
        # Add zero-mean Gaussian noise to odometry measurements
        trans = trans - np.random.normal(0, sigma_trans**2,num_particles)
        rot1 = rot1 - np.random.normal(0, sigma_rot1**2,num_particles)
        rot2 = rot2 - np.random.normal(0, sigma_rot2**2,num_particles)
        # Compute corrected and noise-added robot pose changes
        dx = trans*np.cos(particles[:,2]+rot1)
        dy = trans*np.sin(particles[:,2]+rot1)
        dtheta = rot1 + rot2
        dx.shape = (num_particles,1)
        dy.shape = (num_particles,1)
        dtheta.shape = (num_particles,1)
        return particles + np.concatenate((dx,dy,dtheta),axis=1)

    def sense(self, particles, sensor_reading, occupancy_grid):
        weights = np.zeros(particles.shape[0])
        for i,particle in enumerate(particles):
            sensor_pose = np.array([particle[0] + 25*np.cos(particle[2]), particle[1] + 25*np.sin(particle[2]), particle[2]])

            x_pos = int(sensor_pose[0]/10)
            y_pos = int(sensor_pose[1]/10)
            if (x_pos, y_pos) in self._z_dic and abs(occupancy_grid[x_pos,y_pos])<0.005:
                z_star = self._z_dic[x_pos, y_pos]
                z_star = z_star[np.mod(np.array(range(int(sensor_pose[2]*180.0/np.pi)-90, int(sensor_pose[2]*180/np.pi)+90)),360)]
                    
                z_hit = np.array((self._z_hit_norm/np.sqrt(2*np.pi*self._z_sigma**2))*np.exp(-0.5*(np.array(z_star-np.array(sensor_reading))**2)/self._z_sigma**2))
                z_short = np.full_like(z_star, 0, dtype=float) #not implimented atm
                z_max = np.array(np.array(sensor_reading) == self._max_laser_reading, dtype=int)
                z_rand = np.full_like(z_star, 1.0/self._max_laser_reading, dtype=float)

                weights[i] = np.sum(np.array(self._z_hit*z_hit + self._z_short*z_short + self._z_max*z_max + self._z_rand*z_rand, dtype=float))
            else:
                continue
        return weights/weights.sum()


class particle_filter(object):
    def __init__(self, configuration_file, occupancy_grid):
        with open(configuration_file) as cfg_file:    
            configuration = json.load(cfg_file)
        self._robot_model = robot(configuration_file)
        self._no_particles = configuration['particle_count']
        self._resample_period = configuration['resample_period']
        self._iterations = 0
        self.started_moving = False

        #Gotta get them all in the good areas
        
        num_good_particles = 0
        self._particles = np.empty([0, 3])

        while(num_good_particles < self._no_particles):
            particles = np.transpose(np.array([np.random.rand(self._no_particles)*8000, np.random.rand(self._no_particles)*8000, 2*np.pi*np.random.rand(self._no_particles)]))
            good_points = occupancy_grid[np.array(np.floor(particles[:, 0]/10), dtype=int), np.array(np.floor(particles[:, 1]/10), dtype=int)] == 1
            particles = particles[good_points, :]
            self._particles = np.append(self._particles, particles, 0)
            num_good_particles = self._particles.shape[0]

        self._particles = self._particles[:self._no_particles, :]
        self._weights = [1.0/self._no_particles]*self._no_particles
        # self._weights = self._weights / np.sum(self._weights)
    
    # def low_variance_resample(self):
        # weight_var = np.var(self._weights, dtype=np.float64)
        # print 'Current weight variance:', weight_var
        # print 'Len Weights: {0}, Len particles: {1}, Num particles: {2}'.format(len(self._weights),len(self._particles),self._no_particles)
        # print('Resampling!')
        # newParticles = list()
        # r = np.random.rand(1)*self._no_particles
        # c = self._weights[0]
        # i = 0
        # for m in range(self._no_particles):
            # U = r + m*self._no_particles
            # while U > c and i < len(self._weights) - 1:
                # i += 1
                # c += self._weights[i]
            # newParticles.append(self._particles[i])
        # self._particles = np.array(newParticles)
    # def resample(self):
    #   if np.var(self._weights)<self._resample_theshold:
    #       self._particles = self._particles[np.random.choice(self._particles.shape[0], self._no_particles, p = self._weights, replace = True)]

    def propogate(self, odometry):
        self._particles = self._robot_model.move(self._particles, odometry)

    def infer(self, sensor_reading, occupancy_grid):
        print 'infering'
        if self.started_moving:
            self._iterations += 1
        #print sensor_reading
        #partial_sense = partial(self._robot_model.sense, temp_particles=self._particles, sensor_reading=sensor_reading, occupancy_grid=occupancy_grid)
        #pool = multiprocessing.Pool(processes=4)
        #particle_sensor_readings = list(pool.map(partial_sense, range(self._particles.shape[0])))
        #pool.close()
        #pool.join()
        #get sensor readings for each robot pose
        #particle_sensor_readings = np.apply_along_axis(self._robot_model.partial_sense, 1, self._particles, sensor_reading, occupancy_grid) 
        # subtract the current sensor reading from the sensor readings of the particles, take the L2 norm and readjust weights
           
        #particle_sensor_readings = particle_sensor_readings - np.min(particle_sensor_readings)
        #print particle_sensor_readings
        #self._weights = particle_sensor_readings/np.sum(particle_sensor_readings)
        self._weights *= self._robot_model.sense(self._particles, sensor_reading, occupancy_grid)
        self._weights = self._weights / np.sum(self._weights)
        # get new particles by sampling the new distibution of weights
        print 'weights:',self._weights, 'sum', np.sum(self._weights)

        if self._iterations % self._resample_period == 0 and self.started_moving:
            print 'Resampling. Previous weight variance:',np.var(self._weights)
            indices = np.random.choice(self._particles.shape[0], self._no_particles, p = self._weights, replace = True)
            self._weights = [1.0/self._no_particles]*self._no_particles
            self._particles = self._particles[indices,:]


if __name__ == '__main__':
    if create_movies:
        writer.setup(fig, 'animation_particle_filter.mp4', 100)
    test_object = localization_test(ENVIRONMENT_FILE, ROBOT_LOG)
    test_object.visualize()
    test_object.run_test()
    if create_movies:
        writer.finish()
    # filter_object.imshow()
