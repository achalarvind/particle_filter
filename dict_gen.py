import numpy as np
import os
import sys
import read_log
import json
import cv2 
import cPickle as pickle


global path
path=os.path.split(os.path.realpath(__file__))
path=path[0]

ENVIRONMENT_FILE = 'wean.dat'

class gen_dic:
    def __init__(self, env_file):
        # get environemnt information and robot log
        self._enviroment = open(os.path.join(path, 'map',env_file), 'r')
 
        self._max_laser_reading = 8183.0
        
        [self._robot_spec, self._occupancy_grid] = read_log.read_map(self._enviroment)
        self._occupancy_grid = 1 - self._occupancy_grid 
        self._occupancy_grid[self._occupancy_grid > 1.1] = -1

        self._num_interp = 820
        self._linspace_array = np.mat(np.linspace(0, 1, self._num_interp))


    def gen_entry(self, name):
        dictionary = {}
        cir_sweep = np.array(range(0, 360))
        occupancy_grid = self._occupancy_grid
        img = np.empty([800, 800])

        for x in range(0, 800):
            print " "
            print "row: ",x
#            print "col: ",
            for y in range(0, 800):
#                print y,",",
                if(occupancy_grid[x, y] > 0 and occupancy_grid[x,y] < 0.2):
                    point_pose_x = np.array(np.multiply(self._max_laser_reading, np.cos(np.pi*cir_sweep/180.0)))
                    point_pose_y = np.array(np.multiply(self._max_laser_reading, np.sin(np.pi*cir_sweep/180.0)))
               
                    point_pose_x = np.clip(np.array(((np.mat(point_pose_x).T)*self._linspace_array)/10 + x, dtype=int), 0, 799)
                    point_pose_y = np.clip(np.array(((np.mat(point_pose_y).T)*self._linspace_array)/10 + y, dtype=int), 0, 799)
              
                    #mask = np.multiply(np.multiply(point_pose_x >=0, point_pose_x < 800), np.multiply(point_pose_y >= 0, point_pose_y < 800))
                    point_ranges = occupancy_grid[point_pose_x, point_pose_y]
                    point_ranges[:, -1] = 1

                    z_star = np.array([np.where(ranges > 0.01)[0][0] for ranges in point_ranges], dtype=int) 
                    for angle in cir_sweep:
                        img[point_pose_x[angle, z_star[angle]],point_pose_y[angle, z_star[angle]]] = 1;
                    cv2.imshow("Points", img)
                    cv2.waitKey(1)
                    z_star = (self._max_laser_reading*z_star)/(self._num_interp-1)

                    dictionary[(x, y)] = z_star

        
        with open(name, 'wb') as fp:
            pickle.dump(dictionary, fp)

    def display(self, name):
        dictionary = {}

        cir_sweep = np.array(range(0, 360))
        img = np.empty([800, 800])
        
        with open(name, 'rb') as fp:
            dictionary = pickle.load(fp)
        

        for x in range(0, 800):
            for y in range(0, 800):
                if (x, y) not in dictionary:
                    continue
                z_star = dictionary[(x, y)] 
                point_pose_x = np.array(np.multiply(z_star, np.cos(np.pi*cir_sweep/180.0)/10) + x, dtype=int)
                point_pose_y = np.array(np.multiply(z_star, np.sin(np.pi*cir_sweep/180.0)/10) + y, dtype=int)
           
                for angle in cir_sweep:
                    if(point_pose_x[angle] < 0 or point_pose_x[angle] > 799 or point_pose_y[angle] < 0 or point_pose_y[angle] > 799):
                        continue
                    img[point_pose_x[angle], point_pose_y[angle]] = 1;
        cv2.imshow("Points", img)
        cv2.waitKey(0)


thing = gen_dic(ENVIRONMENT_FILE)


#thing.gen_entry("point_dic.dic")
thing.display("point_dic.dic")
