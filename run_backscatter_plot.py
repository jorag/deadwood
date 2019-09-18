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
#dataset_use = 'PGNLM3-C'
dataset_use = 'Coh-A'
datamod_fprefix = 'Sept1-19'
                          
# Name of input object and file with satellite data path string
obj_in_name = datamod_fprefix + dataset_use + '.pkl'
                          
## Read DataModalities object with ground in situ vegetation data
with open(os.path.join(dirname, 'data', obj_in_name), 'rb') as input:
    all_data = pickle.load(input)
        
# Print points
#all_data.print_points()

# Read ground truth point measurements into a matrix
y_var_read = ['n_trees', 'plc', 'pdc', 'Height_GrassHerb', 'Height_Tallshrub', 'Height_Drfshrub']
n_obs_y = length(all_data.idx_list) # Number of observations
n_var_y = length(y_var_read) # Number of ecological variables read 
y_data = np.empty((n_obs_y, n_var_y))
# Loop through list of variables and add to Y mat out
for i_var_y in range(n_var_y):
    y = all_data.read_data_points(y_var_read[i_var_y])
    # Ensure that the data has the correct format and remove NaNs
    y = y.astype(float)
    y[np.isnan(y)] = 0 # Replace NaNs with zeros
    y_data[:,i_var_y] = y

# Get n_trees 
n_trees = y_data[:, y_var_read.index('n_trees')] 
plc = y_data[:, y_var_read.index('plc')] # Use index from y_var_read 
pdc = y_data[:, y_var_read.index('pdc')]

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
n_comp = 2
cca = CCA(n_components=n_comp)
cca.fit(sar_data, y_data)

# Print weights and scores
print(cca.x_scores_, cca.y_scores_)
print(cca.x_weights_, cca.y_weights_)
print(np.allclose(cca.x_scores_, np.matmul(sar_data, cca.x_weights_)))
print((cca.x_scores_ - np.matmul(sar_data, cca.x_loadings_))**2)

X_c, Y_c = cca.transform(sar_data, y_data )

print((X_c - np.matmul(sar_data, cca.x_weights_)))


# Plot number of trees vs. backscatter values
c_vec = mycolourvec()
fig = plt.figure()
for i_comp in range(n_comp):
    plt.scatter(X_c[:,i_comp], Y_c[:,i_comp], c=c_vec[i_comp])


# Plot a combination of Proportion Live Crown (PLC) and Proportion Defoliated Crown (PDC) vs. backscatter values
plot_x = np.log(plc)

fig = plt.figure()
plt.scatter(plot_x, sar_data[:,0], c='r')

fig = plt.figure()
plt.scatter(plot_x, sar_data[:,1], c='b')

fig = plt.figure()
plt.scatter(plot_x, sar_data[:,2], c='g')

# Plot multivariate linear regression for PLC and PDC
# Set transformation
transformation = np.log # identity # Loop over transformations?
for response in ['plc', 'pdc']:
    r = y_data[:, y_var_read.index(response)]  # Response
    r = transformation(r) # Do a transformation
    r[~np.isfinite(r)] = -100 # Replace NaNs 
    # Add column of ones
    X = np.hstack((np.ones((length(r),1)), sar_data))
    # Find regression parameters
    W = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T,X)),X.T), r)
    # Find predicted curve
    reg_curve = np.matmul(X,W)
    
    coeff_of_det = rsquare(r, reg_curve)
    print(coeff_of_det)
    
    # Create figure
    fig = plt.figure()
    # Plot data
    plt.plot(r, color='b')
    plt.plot(reg_curve, color='r')
    plt.legend(['Measured '+response.upper(), 'Regression '+response.upper()])
    plt.xlabel('Transect point nr.')
    plt.ylabel(response.upper())
    plt.title('Multivariate/multiple linear regression: R^2 = ' +'{:.3f}'.format(coeff_of_det))
    plt.show()  # display it


# Plot Y
plot_x = pdc
plot_y = 2*sar_data[:,1]/(sar_data[:,0]+sar_data[:,2])

fig = plt.figure()
plt.scatter(plot_x, plot_y, c='b')

fig = plt.figure()
plt.scatter(plot_y, plot_x, c='g')

# For analysing all datasets in object

# for dataset_use in dataset_list:
    # Calculate single number