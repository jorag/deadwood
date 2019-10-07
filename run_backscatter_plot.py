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
id_list = ['A', 'B', 'C']

# Name of input object and file with satellite data path string
obj_in_name = datamod_fprefix + '-' + '.pkl'

# List of plots
plot_list = ['quad-pgnlm'] # ['cca', 'pxcvsu', 'linreg', 'cca_reg', 'quad-pgnlm']

# Parameters
y_var_read = ['plc', 'pdc']
n_cca_comp = 2

## Read DataModalities object with ground in situ vegetation data
with open(os.path.join(dirname, 'data', obj_in_name), 'rb') as input:
    all_data = pickle.load(input)
        
#all_data.print_points() # Print points
print(all_data.all_modalities)
                      
# Read ground truth point measurements into a matrix 
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

if 'quad-PGNLM'.lower() in plot_list:
    # Get SAR data 
    quad_data = all_data.read_data_points('Quad-C', modality_type='Quad')
    # Remove singelton dimensions
    quad_data = np.squeeze(quad_data)
    # Get SAR data 
    pgnlm_data = all_data.read_data_points('PGNLM3-C', modality_type='PGNLM3')
    # Remove singelton dimensions
    pgnlm_data = np.squeeze(pgnlm_data)
    
    # Plot
    plt.figure(); plt.title('HH')
    plt.plot(quad_data[:,0]); plt.plot(pgnlm_data[:,0]); plt.legend(['quad','PGNLM'])
    plt.xlabel('Transect point nr.'); plt.ylabel('Backscatter'); plt.show()
    plt.figure(); plt.title('HV')
    plt.plot(quad_data[:,1]); plt.plot(pgnlm_data[:,1]); plt.legend(['quad','PGNLM'])
    plt.xlabel('Transect point nr.'); plt.ylabel('Backscatter'); plt.show()
    plt.figure(); plt.title('VV')
    plt.plot(quad_data[:,3]); plt.plot(pgnlm_data[:,2]); plt.legend(['quad','PGNLM'])
    plt.xlabel('Transect point nr.'); plt.ylabel('Backscatter'); plt.show()

# Collect performance measures in dict
linreg_plc_r2 = dict()
linreg_pdc_r2 = dict()
cca_plc_r2 = dict()
cca_pdc_r2 = dict()

# Calculate measures of performance for CCA and linear regression
# Go through all satellite images and all data modalities in object
for dataset_id in id_list: 
    for dataset_type in all_data.all_modalities:
        print(dataset_type)            
        # Get satellite data
        try:
            dataset_use = dataset_type+'-'+dataset_id
            sat_data = all_data.read_data_points(dataset_use, modality_type=dataset_type)
            print(dataset_use)
        except:
            continue
        
        # Ensure that the output array has the proper shape (2 dimensions)
        if length(sat_data.shape) == 1:
            # If data is only a single column make it a proper vector
            sat_data = sat_data[:, np.newaxis]
        elif length(sat_data.shape) > 2:
            # Remove singelton dimensions
            sat_data = np.squeeze(sat_data)
        #print(sat_data)
        
        # TODO: 20190930 - assert that n_cca_comp >= sat_data.shape[1] ?
        # For now ignore NDVI
        if n_cca_comp > sat_data.shape[1]:
            continue
        
        # Canonical-correlation analysis (CCA)
        cca = CCA(n_components=n_cca_comp)
        cca.fit(sat_data, y_data)
        
        # Get CCA transformation
        U_c, V_c  = cca.x_scores_, cca.y_scores_ #= cca.transform(sat_data, y_data)
        
        # From: https://stackoverflow.com/questions/37398856/
        rho_cca = np.corrcoef(U_c.T, V_c.T).diagonal(offset=n_cca_comp)
        #score = np.diag(np.corrcoef(cca.x_scores_, cca.y_scores_, rowvar=False)[:n_cca_comp, n_cca_comp:])
        
        # Use function definition
        cod_cca2 = rsquare(U_c, y_data)
        print(cod_cca2)
        # Add to output dict
        cca_plc_r2[dataset_use] = cod_cca2[0] # TODO: set index programatically
        cca_pdc_r2[dataset_use] = cod_cca2[1] # TODO: set index programatically
        # Calculate Coefficient of Determination (COD) = R²
        cod_cca = cca.score(sat_data, y_data)
        print(cod_cca)
    
        # Plot number of CCA U and V
        if 'CCA'.lower() in plot_list:
            legend_list = []
            fig = plt.figure()
            for i_comp in range(n_cca_comp):
                plt.scatter(U_c[:,i_comp], V_c[:,i_comp], c=c_vec[i_comp])
                legend_list.append('Comp. nr. '+str(i_comp)+ r' $\rho$ = ' +'{:.3f}'.format(rho_cca[i_comp]))
            plt.title(dataset_use+' CCA: R^2 = ' +'{:.3f}'.format(cod_cca))
            plt.legend(legend_list)
            plt.show()  # display it
        
        # Plot number of CCA U and PLC
        if 'PxCvsU'.lower() in plot_list:
            legend_list = []
            fig = plt.figure()
            for i_comp in range(n_cca_comp):
                plt.scatter(V_c[:,i_comp], y_data[:, y_var_read.index('plc')] , c=c_vec[i_comp])
                legend_list.append('Comp. nr. '+str(i_comp)+ r' $\rho$ = ' +'{:.3f}'.format(rho_cca[i_comp]))
            plt.title(dataset_use+' CCA: R^2 = ' +'{:.3f}'.format(cod_cca))
            plt.xlabel('U'); plt.ylabel('PLC'); plt.legend(legend_list)
            plt.show()
            
            # Plot number of CCA U and PDC
            legend_list = []
            fig = plt.figure()
            for i_comp in range(n_cca_comp):
                plt.scatter(V_c[:,i_comp], y_data[:, y_var_read.index('pdc')] , c=c_vec[i_comp])
                legend_list.append('Comp. nr. '+str(i_comp)+ r' $\rho$ = ' +'{:.3f}'.format(rho_cca[i_comp]))
            plt.title(dataset_use+' CCA: R^2 = ' +'{:.3f}'.format(cod_cca))
            plt.xlabel('U'); plt.ylabel('PDC'); plt.legend(legend_list)
            plt.show()  # display it
            
        # Plot number of CCA U and PLC
        if 'CCA_reg'.lower() in plot_list:
            X_CCA = np.hstack((np.ones((sat_data.shape[0],1)), U_c))
            # Find regression parameters and regression curve
            W_CCA = np.matmul(np.matmul(np.linalg.inv(np.matmul(X_CCA.T,X_CCA)),X_CCA.T), y_data)
            r_CCA = np.matmul(X_CCA, W_CCA)
            # Calculate Coefficient of Determination (COD) = R²
            cca_reg = rsquare(y_data, r_CCA)
            print(r'CCA R^2')
            print(cca_reg)
            for i_col in range(r_CCA.shape[1]):
                fig = plt.figure()
                # Plot data
                plt.plot(y_data[:, i_col] , color='b')
                plt.plot(r_CCA[:, i_col], color='r')
                plt.legend(['Measured '+y_var_read[i_col].upper(), 'Regression '+y_var_read[i_col].upper()])
                plt.xlabel('Transect point nr.')
                plt.ylabel(y_var_read[i_col].upper())
                plt.title(dataset_use+' CCA regression: R^2 = ' +'{:.3f}'.format(cca_reg[i_col]))
                plt.show()  # display 
            
        # Linreg - TODO: 20190929 - Do some transformation??
        # Add column of ones
        X = np.hstack((np.ones((sat_data.shape[0],1)), sat_data))
        # Find regression parameters
        W = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T,X)),X.T), y_data)
        r_hat = np.matmul(X,W)
        # Calculate Coefficient of Determination (COD) = R²
        cod_reg = rsquare(y_data, r_hat)
        print(r'Regression R^2')
        print(cod_reg)
        
        # Add to output dict
        linreg_plc_r2[dataset_use] = cod_reg[0] # TODO: set index programatically
        linreg_pdc_r2[dataset_use] = cod_reg[1] # TODO: set index programatically
        # Plot Multivariate/multiple linear linear regression for PLC and PDC
        if 'linreg' in plot_list:
            for i_col in range(r_hat.shape[1]):
                fig = plt.figure()
                # Plot data
                plt.plot(y_data[:, i_col] , color='b')
                plt.plot(r_hat[:, i_col], color='r')
                plt.legend(['Measured '+y_var_read[i_col].upper(), 'Regression '+y_var_read[i_col].upper()])
                plt.xlabel('Transect point nr.')
                plt.ylabel(y_var_read[i_col].upper())
                plt.title(dataset_use+' regression: R^2 = ' +'{:.3f}'.format(cod_reg[i_col]))
                plt.show()  # display 


# Plot summary statistics
n_datasets = length(linreg_pdc_r2)
x_bars = np.arange(n_datasets) # range(n_datasets)
ofs = 0.15 # offset
alf = 0.7 # alpha
# Linreg
# # Try sorting dictionaries alphabetically
# From: 
#sorted(linreg_plc_r2, key=linreg_plc_r2.get, reverse=True)
#sorted(linreg_pdc_r2, key=linreg_pdc_r2.get, reverse=True)
plt.figure()
plt.bar(x_bars+ofs, list(linreg_plc_r2.values()), align='center', color='b', alpha=alf)
plt.bar(x_bars-ofs, list(linreg_pdc_r2.values()), align='center', color='r', alpha=alf)
plt.xticks(x_bars, list(linreg_pdc_r2.keys()))
plt.legend(['PLC - Live','PDC - Defoliated'])
plt.title('Linear regression: R^2')
plt.show()
# CCA
plt.figure()
plt.bar(x_bars+ofs, list(cca_plc_r2.values()), align='center', color='b', alpha=alf)
plt.bar(x_bars-ofs, list(cca_pdc_r2.values()), align='center', color='r', alpha=alf)
plt.xticks(x_bars, list(cca_pdc_r2.keys()))
plt.title('CCA: R^2')
plt.legend(['PLC - Live','PDC - Defoliated'])
plt.show()