#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 20:49:32 2020

@author: jorag
"""

import matplotlib # For setting font size
import matplotlib.pyplot as plt
import numpy as np
import gdal
import pandas as pd
import os # Necessary for relative paths
import ast
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (train_test_split, 
                                     cross_val_score, 
                                     StratifiedKFold,
                                     GroupKFold)
from sklearn.metrics import confusion_matrix
# My moduels
from mytools import *
from geopixpos import *
from visandtest import *
from dataclass import *


#%% File with list of datasets and band info  
new_datalist_xls_file = 'SF_forest_subsets.xls' 
# Prefix for output datamodalities object filename
datamod_fprefix = 'PGNLM-SNAP_C3_20200116' #'PGNLM-SNAP_C3_geo_OPT_20200113'
base_obj_name = 'DiffGPS_FIELD_DATA'+'.pkl' # Name of the (pure) field data object everything is based on

# Run on grid data or homogeneous AOI
use_test_aois = False

pgnlm_set = 'PGNLM_19-2_v4'
# List of datasets to process
#dataset_list = ['refined_Lee_5x5_C3', 'boxcar_5x5_C3', 'IDAN_50_C3', 'NLSAR_1_1', pgnlm_set] 
#dataset_list = [pgnlm_set]
dataset_list = [pgnlm_set, 'NOOPT_1521patch', 'NOOPT_SARsort_64patch']
#dataset_keys = ['optical', 'boxcar',  'refined Lee', 'IDAN', 'NL-SAR', 'PGNLM']
id_list = ['A', 'C'] 

# Datasets to add optical bands from
opt_dataset_list = ['geo_opt', 'NDVI']

# Which Sentinel-2 bands to use
opt_bands_include = ['b02','b03','b04','b05','b08'] # all 10 m resolution
    
# Path to working directory 
dirname = os.path.realpath('.') 
parent_dir = os.path.realpath('..') # For parent directory 

#%% Classification parameters
knn_k = 5
rf_ntrees = 200 # Number of trees in the Random Forest algorithm
# Cross validation parameters
crossval_split_k = 4
crossval_kfold = StratifiedKFold(n_splits=crossval_split_k, 
                                 shuffle=True, random_state=crossval_split_k)
group_kfold = GroupKFold(n_splits=crossval_split_k)


#%% Output dicts
knn_mean_acc = dict()
knn_all_acc = dict()
rf_mean_acc = dict()
rf_all_acc = dict()

data_dict = dict()


# %% AOIs or grid data?
if use_test_aois:                        
    # Get full data path from specified by input-path files
    with open(os.path.join(dirname, 'input-paths', 'defo1-aoi-path')) as infile:
        defo_file = infile.readline().strip()
    with open(os.path.join(dirname, 'input-paths', 'live1-aoi-path')) as infile:
        live_file = infile.readline().strip()
    # Read AOI polygons into lists
    data_dict['dead'] = read_wkt_csv(defo_file)  
    data_dict['live'] = read_wkt_csv(live_file)
    # Set cross validator to use and groups param
    crossval_use = crossval_kfold
    groups = None
else:
    # Get full data path from specified by input-path files
    with open(os.path.join(dirname, 'input-paths', 'mixed_30m_latlon')) as infile:
        grid_file = infile.readline().strip()
    with open(os.path.join(dirname, 'input-paths', 'mixed_30m_classes')) as infile:
        class_file = infile.readline().strip()
    # Read layer
    grid_list = read_shp_layer(grid_file)
    # Read list of class IDs
    class_dict = read_class_csv(class_file)
    
    # Create lists for each class
    states_use = ['live', 'dead', 'damaged', 'other']
    #states_use = ['live', 'dead', 'other']
    #states_use = ['live', 'dead', 'damaged']
    #states_use = ['live', 'dead']
    for state_collect in states_use:
        data_dict[state_collect] = layer2roipoly_state(grid_list, state_collect)
        
    # Set cross validator to use and groups param
    crossval_use = group_kfold
    groups = []
    
                   
# %% Load band lists from Excel file
xls_fullpath = os.path.join(dirname, 'input-paths', new_datalist_xls_file)
datasets_xls = pd.ExcelFile(xls_fullpath)
df = pd.read_excel(datasets_xls)


# %% LOOP THROUGH SATELLITE DATA
for dataset_id in id_list:
    
    # %% Load optical data
    #dataset_id = 'A'
    dataset_in = pgnlm_set #'geo_opt'
    dataset_use = dataset_in +'-'+dataset_id
    sat_file = df.loc[(df['Dataset_key'] == dataset_id) & (df['Processing_key'] == dataset_in), 'Path'].values[0]
    lat_band = ast.literal_eval(df.loc[(df['Dataset_key'] == dataset_id) & (df['Processing_key'] == dataset_in), 'Lat_band'].values[0])[0]
    lon_band = ast.literal_eval(df.loc[(df['Dataset_key'] == dataset_id) & (df['Processing_key'] == dataset_in), 'Lon_band'].values[0])[0]
    opt_bands_use = ast.literal_eval(df.loc[(df['Dataset_key'] == dataset_id) 
        & (df['Processing_key'] == dataset_in), 'OPT_bands'].values[0])
    # List of optical band names for PGNLM
    #opt_band_names = ['b02','b03','b04','b05','b06','b07','b08','b08a','b11','b12']
    opt_band_names = ['b02','b03','b04','b08']
    opt_bands_include = ['b02','b03','b04','b08'] # all 10 m resolution
    
    # Add OPT bands
    opt_bands_dict = dict()
    opt_bands_dict[dataset_use] = dict(zip(opt_band_names , opt_bands_use))
    
    # Load data
    dataset = gdal.Open(sat_file)
    # Read ALL bands - note that it will be zero indexed
    raster_data_array = dataset.ReadAsArray()
    
    
    opt_bands_use = [] # Check which of the available bands should be included 
    for key in opt_bands_include:
        opt_bands_use.append(opt_bands_dict[dataset_use][key])
    opt_data_temp = raster_data_array[opt_bands_use,:,:]
    
    # Read lat and lon bands
    lat_opt = raster_data_array[lat_band,:,:]
    lon_opt = raster_data_array[lon_band,:,:]
    
    # %% Read satellite data
    for dataset_in in dataset_list:
        dataset_use = dataset_in +'-'+dataset_id 
            
        # Set name of output object
        # Use path from Excel file
        try: 
            sat_file = df.loc[(df['Dataset_key'] == dataset_id) & (df['Processing_key'] == dataset_in), 'Path'].values[0]
            # Get indices for bands, lat, lon, SAR, and optical
            lat_band = ast.literal_eval(df.loc[(df['Dataset_key'] == dataset_id) & (df['Processing_key'] == dataset_in), 'Lat_band'].values[0])[0]
            lon_band = ast.literal_eval(df.loc[(df['Dataset_key'] == dataset_id) & (df['Processing_key'] == dataset_in), 'Lon_band'].values[0])[0]
            sar_bands_use = ast.literal_eval(df.loc[(df['Dataset_key'] == dataset_id) 
                & (df['Processing_key'] == dataset_in), 'SAR_bands'].values[0])

        except:
            # Something went wrong
            continue
          
        # %% Try reading the dataset
        try:
            # Load data
            dataset = gdal.Open(sat_file)
            # Read ALL bands - note that it will be zero indexed
            raster_data_array = dataset.ReadAsArray()
        except:
            # Something went wrong
            continue
        
        # %% Check input data type for feature extraction
        if dataset_in.lower()[0:5] in ['pgnlm', 'noopt']:
            if dataset_in.lower()[6:10] in ['2019', 'best']:
                c3_feature_type = 'iq2c3'
            else:
                c3_feature_type = 'c3pgnlm5feat'
        elif dataset_in.lower()[0:5] in ['nlsar']:
            c3_feature_type = 'all' # 5 features
        elif dataset_in.lower()[-2:] in ['c3']:
            c3_feature_type = 'c3snap5feat'
        else:
            c3_feature_type = 'all'
                
        # %% If SAR data should be added
        if sar_bands_use:
            # Get array with SAR data
            sat_data_temp = raster_data_array[sar_bands_use,:,:]   

                
        #%% Read lat and lon bands
        lat = raster_data_array[lat_band,:,:]
        lon = raster_data_array[lon_band,:,:]
        
        #%% Get AOI pixels

        # Initialise 
        x = None
        y = None
        #groups = np.array(())
        i_class_label = int(0)
        i_group = 0
        
        # Go through all classes
        for aoi_state, aoi_list in data_dict.items():
            # Loop through all AOIs for each class
            for i_aoi, aoi in enumerate(aoi_list):
                # Get LAT and LONG for bounding rectangle and get pixel coordinates 
                lat_bounds, lon_bounds = aoi.bounding_coords()
                row_ind, col_ind = geobox(lat_bounds, lon_bounds, lat, lon, margin=(0,0), log_type='default')
                x_min = row_ind[0]; x_max = row_ind[1]
                y_min = col_ind[0]; y_max = col_ind[1]
        
                # Get SAR data from area
                sat_data = sat_data_temp[:, x_min:x_max, y_min:y_max]
                
                # Extract SAR covariance mat features, reshape to get channels last
                sat_data = np.transpose(sat_data, (1,2,0))
                sat_data = get_sar_features(sat_data, feature_type=c3_feature_type, input_type='img')
                
                # Get OPT indices
                row_ind, col_ind = geobox(lat_bounds, lon_bounds, lat_opt, lon_opt, margin=(0,0), log_type='default')
                x_min = row_ind[0]; x_max = row_ind[1]
                y_min = col_ind[0]; y_max = col_ind[1]
        
                # Get OPT data from area
                opt_data = np.copy(opt_data_temp[:, x_min:x_max, y_min:y_max])
                
                
                # Flatten, reshape to channels first due to ordering of reshape
                sat_data = np.transpose(sat_data, (2,0,1))
                c_first_shape = sat_data.shape
                sat_data = np.reshape(sat_data, (c_first_shape[0],c_first_shape[1]*c_first_shape[2]), order='C')
                
                # Reshape optical data
                opt_data = np.reshape(opt_data, (opt_data.shape[0], opt_data.shape[1]*opt_data.shape[2]), order='C')
                
                # Stack with SAR data
                sat_data = np.vstack((sat_data, opt_data))
                
                # Create a new array or append to existing one
                if i_aoi == 0:
                    state_data = np.copy(sat_data)
                    # Check if groups array should initialized
                    if i_group == 0:
                        groups = i_group*np.ones(sat_data.shape[1],)
                    else:
                        groups = np.hstack((groups, i_group*np.ones(sat_data.shape[1],)))
                else:
                    state_data = np.hstack((state_data, sat_data))
                    groups = np.hstack((groups, i_group*np.ones(sat_data.shape[1],)))
                    
                i_group += 1
            
            # Add to merged data array
            if i_class_label == 0:
                x = np.copy(state_data)
                # TODO: 20201029 change to use x.shape instead of length
                y = i_class_label*np.ones( (length(state_data), ) ,dtype='int')
            else:
                x = np.hstack((x, state_data))
                y = np.hstack((y, i_class_label*np.ones( (length(state_data), ) ,dtype='int')))
            # Update class label
            i_class_label += int(1)
            
            
        # Transpose
        x = x.transpose((1,0))

        #%% Plot 3D backscatter values

        # Plot
        #labels_dict = None # dict((['live', 'defo'], ['live', 'defo']))
        #modalitypoints3d('reciprocity', x, y, labels_dict=labels_dict, title=dataset_use)
        
        #%% Classify
        group_kfold.get_n_splits(X=x, y=y, groups=groups)
        # Cross validate - kNN - All data
        knn_all = KNeighborsClassifier(n_neighbors=knn_k)
        knn_scores_all = cross_val_score(knn_all, x, y, groups=groups, 
                                         cv=crossval_use)
        #knn_scores_all = cross_val_score(knn_all, x, y, cv=crossval_kfold)
        print('kNN - ' + dataset_use + ' :')
        print(np.mean(knn_scores_all))
        knn_mean_acc[dataset_use] = np.mean(knn_scores_all)
        knn_all_acc[dataset_use] = knn_scores_all
        
        rf_all = RandomForestClassifier(n_estimators=rf_ntrees, random_state=0)
        rf_scores_all = cross_val_score(rf_all, x, y, groups=groups, 
                                        cv=crossval_use)
        #rf_scores_all = cross_val_score(rf_all, x, y, cv=crossval_kfold)
        print('Random Forest - ' + dataset_use + ' :')
        print(np.mean(rf_scores_all))
        rf_mean_acc[dataset_use] = np.mean(rf_scores_all)
        rf_all_acc[dataset_use] = rf_scores_all
        
#%% Plot classification summary
# Plot summary statistics
n_datasets = length(rf_mean_acc)
x_bars = np.arange(n_datasets) # range(n_datasets)
ofs = 0.25 # offset
alf = 0.7 # alpha

class_n, class_counts = np.unique(y, return_counts = True)
print(rf_mean_acc)
