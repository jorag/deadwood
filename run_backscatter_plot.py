#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 16:33:15 2019

@author: jorag
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os # Necessary for relative paths
import xml.etree.ElementTree as ET
import pickle
#import sys # To append paths
# My moduels
from mytools import *
from geopixpos import *
from visandtest import *
from dataclass import *


# Path to working directory 
dirname = os.path.realpath('.') # For parent directory use '..'

# Prefix for object filename
dataset_use = 'PGNLM3-C'
datamod_fprefix = 'Aug1-19'
                          
# Name of input object and file with satellite data path string
obj_in_name = datamod_fprefix + dataset_use + '.pkl'
                          
## Read DataModalities object with ground in situ vegetation data
with open(os.path.join(dirname, 'data', obj_in_name), 'rb') as input:
    all_data = pickle.load(input)
        
# Print points
#all_data.print_points()

# Get n_trees 
n_trees = all_data.read_data_points('n_trees') 
lai = all_data.read_data_points('lai') 
dai = all_data.read_data_points('dai') 

#trees = all_data.read_data_points('Tree') # Fails due to to all points having trees and hence no "Tree" attribute

# Get SAR data 
sar_data = all_data.read_data_points(dataset_use, modality_type='quad_pol') 
# Get OPT data
opt_data = all_data.read_data_points(dataset_use, modality_type='optical') 

# Remove singelton dimensions
sar_data = np.squeeze(sar_data)
opt_data = np.squeeze(opt_data)

fig = plt.figure()
plt.scatter(n_trees, sar_data[:,0], c='r')

fig = plt.figure()
plt.scatter(n_trees, sar_data[:,1], c='b')

fig = plt.figure()
plt.scatter(n_trees, sar_data[:,2], c='g')


fig = plt.figure()
plt.scatter(np.log(lai.astype(float)), sar_data[:,0], c='r')

fig = plt.figure()
plt.scatter(np.log(lai.astype(float)), sar_data[:,1], c='b')

fig = plt.figure()
plt.scatter(np.log(lai.astype(float)), sar_data[:,2], c='g')



fig = plt.figure()
plt.scatter(lai, sar_data[:,0], c='r')

fig = plt.figure()
plt.scatter(lai, sar_data[:,1], c='b')

fig = plt.figure()
plt.scatter(lai, sar_data[:,2], c='g')