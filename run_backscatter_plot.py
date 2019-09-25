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
datamod_fprefix = 'All-data-0919'
dataset_id = 'A'

# Name of input object and file with satellite data path string
#obj_in_name = datamod_fprefix + '-' + dataset_id + '.pkl'
obj_in_name = datamod_fprefix + '-' + '.pkl'

# List of plots
plot_list = ['pxcvsu'] # ['cca', 'pxcvsu', 'linreg']

## Read DataModalities object with ground in situ vegetation data
with open(os.path.join(dirname, 'data', obj_in_name), 'rb') as input:
    all_data = pickle.load(input)
        
#all_data.print_points() # Print points
print(all_data.all_modalities)
                      
# Read ground truth point measurements into a matrix 
y_var_read = ['plc', 'pdc'] 
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

# Get colour vector
c_vec = mycolourvec()

# Get SAR data 
quad_data = all_data.read_data_points('Quad-C', modality_type='Quad')
# Remove singelton dimensions
quad_data = np.squeeze(quad_data)
# Get SAR data 
pgnlm_data = all_data.read_data_points('PGNLM3-C', modality_type='PGNLM3')
# Remove singelton dimensions
pgnlm_data = np.squeeze(pgnlm_data)

# Plot
plt.figure()
plt.plot(quad_data[:,0])
plt.plot(pgnlm_data[:,0])

# Performance measures
linreg_pdc_r2 = dict()

# Analyse all data modalities in object
for dataset_type in all_data.all_modalities:
    # Calculate measures of performance for CCA and linear regression
    
    #if dataset_type.lower() in ['optical']:
    #   continue
                                   
    # Get SAR data
    try:
        dataset_use = dataset_type+'-'+dataset_id
        sar_data = all_data.read_data_points(dataset_use, modality_type=dataset_type)
        print(dataset_use)
        # Get OPT data
        #opt_data = all_data.read_data_points(dataset_use, modality_type='optical')
    except:
        continue
    
    # Remove singelton dimensions
    sar_data = np.squeeze(sar_data)
    #opt_data = np.squeeze(opt_data)
    print(sar_data)
    
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
    
    # Plot number of CCA U and V
    if 'CCA'.lower() in plot_list:
        legend_list = []
        fig = plt.figure()
        for i_comp in range(n_comp):
            plt.scatter(U_c[:,i_comp], V_c[:,i_comp], c=c_vec[i_comp])
            legend_list.append('Comp. nr. '+str(i_comp)+ r' $\rho$ = ' +'{:.3f}'.format(rho_cca[i_comp]))
        plt.title(dataset_use+' CCA: R^2 = ' +'{:.3f}'.format(cod_cca))
        plt.legend(legend_list)
        plt.show()  # display it
    
    
    # Plot number of CCA U and PDC
    if 'PxCvsU'.lower() in plot_list:
        legend_list = []
        fig = plt.figure()
        for i_comp in range(n_comp):
            plt.scatter(V_c[:,i_comp], y_data[:, y_var_read.index('plc')] , c=c_vec[i_comp])
            legend_list.append('Comp. nr. '+str(i_comp)+ r' $\rho$ = ' +'{:.3f}'.format(rho_cca[i_comp]))
        plt.title(dataset_use+' CCA: R^2 = ' +'{:.3f}'.format(cod_cca))
        plt.xlabel('U')
        plt.ylabel('PLC')
        plt.legend(legend_list)
        plt.show()  # display it
        
        # Plot number of CCA U and PDC
        legend_list = []
        fig = plt.figure()
        for i_comp in range(n_comp):
            plt.scatter(V_c[:,i_comp], y_data[:, y_var_read.index('pdc')] , c=c_vec[i_comp])
            legend_list.append('Comp. nr. '+str(i_comp)+ r' $\rho$ = ' +'{:.3f}'.format(rho_cca[i_comp]))
        plt.title(dataset_use+' CCA: R^2 = ' +'{:.3f}'.format(cod_cca))
        plt.xlabel('U')
        plt.ylabel('PDC')
        plt.legend(legend_list)
        plt.show()  # display it
    
    
    # Plot Multivariate/multiple linear linear regression for PLC and PDC
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
        # Add to output dict
        linreg_pdc_r2[dataset_use] = cod_reg
        
        # Create figure
        if 'linreg' in plot_list:
            fig = plt.figure()
            # Plot data
            plt.plot(r, color='b')
            plt.plot(reg_curve, color='r')
            plt.legend(['Measured '+response.upper(), 'Regression '+response.upper()])
            plt.xlabel('Transect point nr.')
            plt.ylabel(response.upper())
            plt.title(dataset_use+' regression: R^2 = ' +'{:.3f}'.format(cod_reg))
            plt.show()  # display 


# Plot summary statistics
n_datasets = len(linreg_pdc_r2)
plt.figure()
plt.bar(range(n_datasets), list(linreg_pdc_r2.values()), align='center')
plt.xticks(range(n_datasets), list(linreg_pdc_r2.keys()))

plt.show()