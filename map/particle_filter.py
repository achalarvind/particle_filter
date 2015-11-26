import numpy as np
import os
# import read_log

path=os.path.split(os.path.realpath(__file__))
path=path[0]

ENVIRONMENT_FILE = 'wean.dat'
ROBOT_LOG = 'robotdata1.log'

# get the training and test data and labels
enviroment = open(os.path.join(path, 'map',ENVIRONMENT_FILE), 'r')
log = open(os.path.join(path, 'log',), 'r')

[robot_spec, occupancy_grid]=read_log.read_map(enviroment)