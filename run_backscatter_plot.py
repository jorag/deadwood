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
datamod_fprefix = '19_nonorm'
                          
# Name of input object and file with satellite data path string
obj_in_name = datamod_fprefix + dataset_use + '.pkl'
                          
## Read DataModalities object with ground in situ vegetation data
with open(os.path.join(dirname, 'data', obj_in_name), 'rb') as input:
    all_data = pickle.load(input)
        
# Print points
#all_data.print_points()

# Get SAR data
sar_data, labels = all_data.read_data_array(['quad_pol'], 'all') 
# Get OPT data
opt_data, labels = all_data.read_data_array(['optical'], 'all') 

# Get n_trees data
n_trees = all_data.read_data_points('n_trees') 

fig = plt.figure()
plt.scatter(n_trees, sar_data[:,0])
