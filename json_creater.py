#!/usr/bin/env python

import json

file_name="configuration.json"

json_data={
	"particle_count":10000,
	"max_laser_reading":8183,
	"alpha1":0.05,	#ratio motion to x/y std.dev
	"alpha2":0.001,	#ratio rotation to phi std.dev 
	"alpha3":5,	
	"alpha4":0.05,
	"min_std_xy":0.01,
	"min_std_theta":0.2,
	"resample_threshold":0.4
}

# Rz = np.array([[cos(tz), -sin(tz), 0], [sin(tz), cos(tz), 0], [0,0,1]])

if __name__=="__main__":
	with open(file_name, 'w') as outfile:
		json.dump(json_data, outfile, sort_keys = True, indent = 4, ensure_ascii=False)