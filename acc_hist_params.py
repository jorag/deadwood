#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 20:49:32 2020

@author: jorag
"""

import matplotlib.pyplot as plt
import numpy as np
import gdal
import pandas as pd
import os # Necessary for relative paths
import ast
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split # train/test set split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold # is deafault in cross-val?
from sklearn.metrics import confusion_matrix
# My moduels
from mytools import *
from geopixpos import *
from visandtest import *
from dataclass import *


#%% File with list of datasets and band info
# Path to PGNLM data
pgnlm_datadir = os.path.join(os.path.realpath('..'), 'gnlm34', 'data')
pgnlm_dataset_list = os.path.join(pgnlm_datadir, 'processed_list_v1.xls')

# Which Sentinel-2 bands to use
#opt_bands_include = ['b02','b03','b04','b05','b06','b07','b08','b08a','b11','b12']
opt_bands_include = ['b02','b03','b04','b05','b08'] # all 10 m resolution
    
# Path to working directory 
dirname = os.path.realpath('.') # For parent directory use '..'

#%% Classification parameters
knn_k = 5
rf_ntrees = 200 # Number of trees in the Random Forest algorithm
# Type of split of training and test data
split_type = 'crossval' # 'aoi_split'
# Cross validation parameters
crossval_split_k = 5
crossval_kfold = StratifiedKFold(n_splits=crossval_split_k, shuffle=True, random_state=crossval_split_k)

knn_mean_acc = dict()
rf_mean_acc = dict()

# %% Read defoliated AOI                          
# Get full data path from specified by input-path file
with open(os.path.join(dirname, 'input-paths', 'defo1-aoi-path')) as infile:
    defo_file = infile.readline().strip()
# Reading polygons into list
defo_aoi_list = read_wkt_csv(defo_file)

# %% Read live AOI                          
# Get full data path from specified by input-path file
with open(os.path.join(dirname, 'input-paths', 'live1-aoi-path')) as infile:
    live_file = infile.readline().strip()
# Reading polygons into list
live_aoi_list = read_wkt_csv(live_file)
                   
# %% Define band list (TODO: load from Excel file instead??)
lat_band = 9
lon_band = 10
sar_bands_use = [0, 1, 2, 3, 4]
# Features to use
c3_feature_type = 'c3pgnlm5feat'


datasets_xls = pd.ExcelFile(pgnlm_dataset_list)
df = pd.read_excel(datasets_xls)




# %% LOOP THROUGH SATELLITE DATA
for row in df.itertuples(index=True, name='Pandas'):
    print(row.Path, row.gamma)

    dataset_use = row.Time+'-'+row.Dataset_ID 
        
    # Set name of output object
    # Use path from Excel file
    try: 
        sat_file = row.Path

        #print(sat_file)
    except:
        # Something went wrong
        print('Error!')
        continue
      
    # %% Try reading the dataset
    try:
        # Load data
        dataset = gdal.Open(sat_file)
        #gdalinfo_log(dataset, log_type='default')
        
        # Read ALL bands - note that it will be zero indexed
        raster_data_array = dataset.ReadAsArray()
    except:
        # Something went wrong
        continue
    
            
    # %% Get array with SAR data
    sat_data_temp = raster_data_array[sar_bands_use,:,:]   
        
            
    #%% Read lat and lon bands
    lat = raster_data_array[lat_band,:,:]
    lon = raster_data_array[lon_band,:,:]
    
    #%% Get defo AOI pixels
    for i_defo, defo_aoi in enumerate(defo_aoi_list):
        # Get LAT and LONG for bounding rectangle and get pixel coordinates 
        lat_bounds, lon_bounds = defo_aoi.bounding_coords()
        row_ind, col_ind = geobox(lat_bounds, lon_bounds, lat, lon, margin=(0,0), log_type='default')
        x_min = row_ind[0]; x_max = row_ind[1]
        y_min = col_ind[0]; y_max = col_ind[1]

        # Get SAR data from area
        sat_data = sat_data_temp[:, x_min:x_max, y_min:y_max]
        
        # Extract SAR covariance mat features, reshape to get channels last
        sat_data = np.transpose(sat_data, (1,2,0))
        sat_data = get_sar_features(sat_data, feature_type=c3_feature_type, input_type='img')
        
        # Flatten, reshape to channels first due to ordering of reshape
        sat_data = np.transpose(sat_data, (2,0,1))
        c_first_shape = sat_data.shape
        sat_data = np.reshape(sat_data, (c_first_shape[0],c_first_shape[1]*c_first_shape[2]), order='C')
        
        # Create a new array or append to existing one
        if i_defo == 0:
            defo_data = np.copy(sat_data)
        else:
            defo_data = np.hstack((defo_data, sat_data))
            
    #%% Get live AOI pixels
    for i_live, live_aoi in enumerate(live_aoi_list):
        # Get LAT and LONG for bounding rectangle and get pixel coordinates 
        lat_bounds, lon_bounds = live_aoi.bounding_coords()
        row_ind, col_ind = geobox(lat_bounds, lon_bounds, lat, lon, margin=(0,0), log_type='default')
        x_min = row_ind[0]; x_max = row_ind[1]
        y_min = col_ind[0]; y_max = col_ind[1]

        # Get SAR data from area
        sat_data = sat_data_temp[:, x_min:x_max, y_min:y_max]
        
        # Extract SAR covariance mat features, reshape to get channels last
        sat_data = np.transpose(sat_data, (1,2,0))
        sat_data = get_sar_features(sat_data, feature_type=c3_feature_type, input_type='img')
        
        # Flatten, reshape to channels first due to ordering of reshape
        sat_data = np.transpose(sat_data, (2,0,1))
        c_first_shape = sat_data.shape
        sat_data = np.reshape(sat_data, (c_first_shape[0],c_first_shape[1]*c_first_shape[2]), order='C')
        
        # Create a new array or append to existing one
        if i_live == 0:
            live_data = np.copy(sat_data)
        else:
            live_data = np.hstack((live_data, sat_data))
    
    #%% Plot 3D backscatter values

    # Merge arrays with live and defoliated data
    #x = np.hstack((live_data, defo_data))
    x = np.hstack((defo_data, live_data))
    x = x.transpose((1,0))
    # Create labels
    #y = np.hstack((1*np.ones(length(live_data),dtype='int'), 0*np.ones(length(defo_data),dtype='int')))
    y = np.hstack((0*np.ones(length(defo_data),dtype='int'), 1*np.ones(length(live_data),dtype='int')))
    
    labels_dict = None # dict((['live', 'defo'], ['live', 'defo']))
    
    # Plot
    #modalitypoints3d('reciprocity', x, y, labels_dict=labels_dict, title=dataset_use)
    
    #%% Classify
    # Cross validate - kNN - All data
    knn_all = KNeighborsClassifier(n_neighbors=knn_k)
    knn_scores_all = cross_val_score(knn_all, x, y, cv=crossval_kfold)
    #print('kNN - ' + dataset_use + ' :')
    #print(np.mean(knn_scores_all))
    knn_mean_acc[dataset_use] = np.mean(knn_scores_all)
    
    rf_all = RandomForestClassifier(n_estimators=rf_ntrees, random_state=0)
    rf_scores_all = cross_val_score(rf_all, x, y, cv=crossval_kfold)
    print('Random Forest - ' + dataset_use + ' :')
    print(np.mean(rf_scores_all))
    rf_mean_acc[dataset_use] = np.mean(rf_scores_all)
        


#%% Plot classification summary
# Plot summary statistics
n_datasets = length(rf_mean_acc)

plt.figure()
plt.plot(list(rf_mean_acc.values()), 'b')
plt.plot(list(knn_mean_acc.values()), 'r')
