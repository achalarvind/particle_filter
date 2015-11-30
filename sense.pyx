from __future__ import division

import numpy as np
cimport numpy as np


FTYPE = np.float
ITYPE = np.int

ctypedef np.int_t ITYPE_t
ctypedef np.float_t FTYPE_t


def c_genLidar(np.ndarray[FTYPE_t, ndim=1] robot_pose, np.ndarray[FTYPE_t, ndim=2] occupancy_grid, ITYPE_t num_interp, FTYPE_t max_laser):

    cdef np.ndarray[FTYPE_t, ndim=1] x_
    cdef np.ndarray[FTYPE_t, ndim=1] y_

    cdef np.ndarray[FTYPE_t, ndim=1] distances
    distances = np.empty(180)

    cdef FTYPE_t deg2rad
    deg2rad = np.pi/180.0

    cdef FTYPE_t rot
    rot = robot_pose[2]

    cdef FTYPE_t x
    x = (robot_pose[0] + 25*np.cos(rot))/10
    cdef FTYPE_t y
    y = (robot_pose[1] + 25*np.sin(rot))/10

    x_ = np.array(np.cos(rot + deg2rad*(np.array(range(0,180))-90)), dtype=float)
    y_ = np.array(np.sin(rot + deg2rad*(np.array(range(0,180))-90)), dtype=float)

    cdef unsigned int cur_x
    cdef unsigned int cur_y
    cdef unsigned int angle
    cdef unsigned int mag
    cdef FTYPE_t flo_x
    cdef FTYPE_t flo_y

    for angle in range(0,180):
        for mag in range(0,int(max_laser/10)):
            flo_x = x_[angle]*mag + x
            flo_y = y_[angle]*mag + x

            if(flo_x < 0 or flo_x > 799 or flo_y < 0 or flo_y> 799):
                distances[angle] = max_laser
                break
            
            cur_x = <unsigned int>flo_x
            cur_y = <unsigned int>flo_y

            if(occupancy_grid[cur_x, cur_y] > 0.01):
                distances[angle] = mag*10
                break


    return distances

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
    distances = (max_laser*z_star*1.0)/(num_interp-1.0)
    return distances
"""
