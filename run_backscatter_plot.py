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
from sklearn.cross_decomposition import CCA
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

# Read ground truth point measurements into a matrix
y_var_read = ['n_trees', 'plc', 'pdc']
n_obs_y = length(all_data.idx_list) # Number of observations
n_var_y = length(y_var_read) # Number of ecological variables read 
y_data = np.empty((n_obs_y, n_var_y))
# Loop through list of variables and add to Y mat out
for i_var_y in range(n_var_y):
    y = all_data.read_data_points(y_var_read[i_var_y])
    # Ensure that the data has the correct format and remove NaNs
    y = y.astype(float)
    y[np.isnan(y)] = 0
    y_data[:,i_var_y] = y

# Get n_trees 
n_trees = all_data.read_data_points('n_trees') 
plc = all_data.read_data_points('plc') 
pdc = all_data.read_data_points('pdc') 

# Convert to float
plc = plc.astype(float)
pdc = pdc.astype(float)

# for iter in range(length(plc)): print(plc[iter]) # print all values
#trees = all_data.read_data_points('Tree') # Fails due to to all points having trees and hence no "Tree" attribute

# Get SAR data 
sar_data = all_data.read_data_points(dataset_use, modality_type='quad_pol') 
# Get OPT data
opt_data = all_data.read_data_points(dataset_use, modality_type='optical') 

# Remove singelton dimensions
sar_data = np.squeeze(sar_data)
opt_data = np.squeeze(opt_data)


# Try canonical-correlation analysis (CCA)
cca = CCA(n_components=1)
cca.fit(sar_data, y_data )
# Print weights and scores
print(cca.x_scores_, cca.y_scores_)
print(cca.x_weights_, cca.y_weights_)
print(np.allclose(cca.x_scores_, np.matmul(sar_data, cca.x_weights_)))
print((cca.x_scores_ - np.matmul(sar_data, cca.x_loadings_))**2)

X_c, Y_c = cca.transform(sar_data, y_data )



# Plot number of trees vs. backscatter values
fig = plt.figure()
plt.scatter(n_trees, sar_data[:,0], c='r')

fig = plt.figure()
plt.scatter(n_trees, sar_data[:,1], c='b')

fig = plt.figure()
plt.scatter(n_trees, sar_data[:,2], c='g')

# Plot a combination of Proportion Live Crown (PLC) and Proportion Defoliated Crown (PDC) vs. backscatter values
plot_x = np.log(plc)

fig = plt.figure()
plt.scatter(plot_x, sar_data[:,0], c='r')

fig = plt.figure()
plt.scatter(plot_x, sar_data[:,1], c='b')

fig = plt.figure()
plt.scatter(plot_x, sar_data[:,2], c='g')


# Plot Y
plot_x = pdc
plot_y = 2*sar_data[:,1]/(sar_data[:,0]+sar_data[:,2])

fig = plt.figure()
plt.scatter(plot_x, plot_y, c='b')

fig = plt.figure()
plt.scatter(plot_y, plot_x, c='g')