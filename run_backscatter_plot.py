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
        
#all_data.print_points() # Print points

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
    y[np.isnan(y)] = 0 # Replace NaNs with zeros
    y_data[:,i_var_y] = y

#trees = all_data.read_data_points('Tree') # Fails due to to all points having trees and hence no "Tree" attribute

# Get SAR data 
sar_data = all_data.read_data_points(dataset_use, modality_type='quad_pol') 
# Get OPT data
opt_data = all_data.read_data_points(dataset_use, modality_type='optical') 

# Remove singelton dimensions
sar_data = np.squeeze(sar_data)
opt_data = np.squeeze(opt_data)


# Canonical-correlation analysis (CCA)
n_comp = 2
cca = CCA(n_components=n_comp)
cca.fit(sar_data, y_data)

# Get CCA transformation
U_c, V_c  = cca.x_scores_, cca.y_scores_#= cca.transform(sar_data, y_data)

# From: https://stackoverflow.com/questions/37398856/
rho_cca = np.corrcoef(U_c.T, V_c.T).diagonal(offset=n_comp)
#score = np.diag(np.corrcoef(cca.x_scores_, cca.y_scores_, rowvar=False)[:n_comp, n_comp:])

# Calculate Coefficient of Determination (COD) = R²
cod_cca = cca.score(sar_data, y_data)
print(cod_cca)

# Plot number of trees vs. backscatter values
c_vec = mycolourvec()
legend_list = []
fig = plt.figure()
for i_comp in range(n_comp):
    plt.scatter(U_c[:,i_comp], V_c[:,i_comp], c=c_vec[i_comp])
    legend_list.append('Comp. nr. '+str(i_comp)+ r' $\rho$ = ' +'{:.3f}'.format(rho_cca[i_comp]))
plt.title('CCA: R^2 = ' +'{:.3f}'.format(cod_cca))
plt.legend(legend_list)
plt.show()  # display it


# Plot multivariate linear regression for PLC and PDC
# Set transformation
transformation = identity # np.log # Loop over transformations?
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
    # Calculate Coefficient of Determination (COD) = R²
    cod_reg = rsquare(r, reg_curve)
    print(cod_reg)
    
    # Create figure
    fig = plt.figure()
    # Plot data
    plt.plot(r, color='b')
    plt.plot(reg_curve, color='r')
    plt.legend(['Measured '+response.upper(), 'Regression '+response.upper()])
    plt.xlabel('Transect point nr.')
    plt.ylabel(response.upper())
    plt.title('Multivariate/multiple linear regression: R^2 = ' +'{:.3f}'.format(cod_reg))
    plt.show()  # display 


# For analysing all datasets in object

# for dataset_use in dataset_list:
    # Calculate single number