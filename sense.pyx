from __future__ import division

import numpy as np
"""
cimport numpy as np


FTYPE = np.float
ITYPE = np.int

ctypedef np.int_t ITYPE_t
ctypedef np.float_t FTYPE_t


def c_genLidar(np.ndarray[FTYPE_t, ndim=1] robot_pose, np.ndarray[FTYPE_t, ndim=2] occupancy_grid, int num_interp, FTYPE_t max_laser):
    
    cdef np.ndarray[FTYPE_t, ndim=1] sensor_pose

    cdef np.ndarray[FTYPE_t, ndim=2] point_sweep_x
    cdef np.ndarray[FTYPE_t, ndim=2] point_sweep_y

    cdef np.ndarray[ITYPE_t, ndim=2] point_pose_x
    cdef np.ndarray[ITYPE_t, ndim=2] point_pose_y

    cdef np.ndarray[FTYPE_t, ndim=2] point_ranges

    cdef np.ndarray[ITYPE_t, ndim=1] z_star    

    cdef np.ndarray[FTYPE_t, ndim=1] linspacer

    linspacer = np.linspace(0, 1, <unsigned int>num_interp)

    sensor_pose = np.array([robot_pose[<unsigned int>0] + 25*np.cos(robot_pose[<unsigned int>2]), robot_pose[<unsigned int>1] + 25*np.sin(robot_pose[<unsigned int>2]), robot_pose[<unsigned int>2]])

    point_sweep_x = np.array(np.multiply(max_laser, np.cos(sensor_pose[<unsigned int>2] + np.pi*(np.array(range(0,180))-90.0)/180.0)))
    point_sweep_y = np.array(np.multiply(max_laser, np.sin(sensor_pose[<unsigned int>2] + np.pi*(np.array(range(0,180))-90.0)/180.0)))

    point_pose_x = np.clip(np.array(((np.mat(point_sweep_x).T)*linspacer + sensor_pose[<unsigned int>0])/10, dtype=int), 0, 799)
    point_pose_y = np.clip(np.array(((np.mat(point_sweep_y).T)*linspacer + sensor_pose[<unsigned int>1])/10, dtype=int), 0, 799)

    point_ranges = occupancy_grid[point_pose_x, point_pose_y]
    point_ranges[:, <unsigned int>-1] = 1

    z_star = np.array([np.where(ranges > 0.01)[<unsigned int>0][<unsigned int>0] for ranges in point_ranges], dtype=int)

    return z_star

"""

def genLidar(robot_pose, occupancy_grid, num_interp, max_laser):
    
    linspacer = np.linspace(0, 1, num_interp)

    sensor_pose = np.array([robot_pose[0] + 25*np.cos(robot_pose[2]), robot_pose[1] + 25*np.sin(robot_pose[2]), robot_pose[2]])

    point_sweep_x = np.array(np.multiply(max_laser, np.cos(sensor_pose[2] + np.pi*(np.array(range(0,180))-90.0)/180.0)))
    point_sweep_y = np.array(np.multiply(max_laser, np.sin(sensor_pose[2] + np.pi*(np.array(range(0,180))-90.0)/180.0)))

    point_pose_x = np.clip(np.array(((np.mat(point_sweep_x).T)*linspacer + sensor_pose[0])/10, dtype=int), 0, 799)
    point_pose_y = np.clip(np.array(((np.mat(point_sweep_y).T)*linspacer + sensor_pose[1])/10, dtype=int), 0, 799)

    point_ranges = occupancy_grid[point_pose_x, point_pose_y]
    point_ranges[:, -1] = 1

    z_star = np.array([np.where(ranges > 0.01)[0][0] for ranges in point_ranges], dtype=int)

    return z_star
