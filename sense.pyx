from __future__ import division

import numpy as np
cimport numpy as np


FTYPE = np.float
ITYPE = np.int

ctypedef np.int_t ITYPE_t
ctypedef np.float_t FTYPE_t

import types




def sense(cls, int particle_no, np.ndarray[FTYPE, ndim=2] temp_particles, np.ndarray[ITYPE, ndim=1 sensor_reading, occupancy_grid):
    
    cdef np.ndarray[FTYPE, ndim=1] robot_pose

    robot_pose = temp_particles[particle_no,:]

    sensor_pose = np.array([robot_pose[0] + 25*np.cos(robot_pose[2]), robot_pose[1] + 25*np.sin(robot_pose[2]), robot_pose[2]])

    point_pose_x = np.array(np.multiply(self._max_laser_reading, np.cos(sensor_pose[2] + np.pi*(np.array(range(0,180))-90.0)/180.0)))
    point_pose_y = np.array(np.multiply(self._max_laser_reading, np.sin(sensor_pose[2] + np.pi*(np.array(range(0,180))-90.0)/180.0)))

    point_pose_x = np.clip(np.array(((np.mat(point_pose_x).T)*self._linspace_array + sensor_pose[0])/10, dtype=int), 0, 799)
    point_pose_y = np.clip(np.array(((np.mat(point_pose_y).T)*self._linspace_array + sensor_pose[1])/10, dtype=int), 0, 799)

    #mask = np.multiply(np.multiply(point_pose_x >=0, point_pose_x < 800), np.multiply(point_pose_y >= 0, point_pose_y < 800))
    point_ranges = occupancy_grid[point_pose_x, point_pose_y]
    point_ranges[:, -1] = 1

    z_star = np.array([np.where(ranges > 0.01)[0][0] for ranges in point_ranges], dtype=int)
    z_star = (8183.0*z_star)/self._num_interp

    z_hit = np.array((self._z_hit_norm/np.sqrt(2*np.pi*self._z_sigma**2))*np.exp(-0.5*(np.array(z_star*10-np.array(sensor_reading))**2)/self._z_sigma**2))
    z_short = np.full_like(z_star, 0, dtype=float) #not implimented atm
    z_max = np.array(np.array(sensor_reading) == self._max_laser_reading, dtype=int)
    z_rand = np.full_like(z_star, 1.0/self._max_laser_reading, dtype=float)

    weights = np.array(self._z_hit*z_hit + self._z_short*z_short + self._z_max*z_max + self._z_rand*z_rand, dtype=float)

    return np.sum(np.log(weights))



