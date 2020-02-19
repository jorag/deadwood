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
import xml.etree.ElementTree as ET
import pickle
import ast
# My moduels
from mytools import *
from geopixpos import *
from visandtest import *
from dataclass import *


# Check if the new band list format, where everything is stored in a .xls, is used
new_datalist_xls_file = '2020_C3_dataset_overview.xls' # '2019_reprocess_dataset_overview.xls'
# Prefix for output datamodalities object filename
datamod_fprefix = 'PGNLM-SNAP_C3_20200116' #'PGNLM-SNAP_C3_geo_OPT_20200113'
base_obj_name = 'DiffGPS_FIELD_DATA'+'.pkl' # Name of the (pure) field data object everything is based on 

# List of datasets to process
#dataset_list = ['iq', 'C3', 'cloude_3x3', 'genFD_3x3', 'vanZyl_3x3', 'yamaguchi_3x3', 'collocate_iq', 'collocate_C3', 'pgnlm_iq'] 
#dataset_list = ['C3', 'refined_Lee_5x5_C3', 'boxcar_5x5_C3', 'IDAN_50_C3'] 
#dataset_list = ['geo_opt']
dataset_list = ['IDAN_50_C3']
id_list = ['A', 'C'] #['A', 'B', 'C'] # TODO: 20190909 Consider changing this a date string
add_ndvi = True

# Datasets to add optical bands from
opt_dataset_list = ['geo_opt']

# Which Sentinel-2 bands to use
#opt_bands_include = ['b02','b03','b04','b05','b06','b07','b08','b08a','b11','b12']
opt_bands_include = ['b02','b03','b04','b05','b08'] # b02, b03, b04, b08, all 10 m resolution
    
# Path to working directory 
dirname = os.path.realpath('.') # For parent directory use '..'

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
                   
# %% Load band lists from Excel file
xls_fullpath = os.path.join(dirname, 'input-paths', new_datalist_xls_file)
datasets_xls = pd.ExcelFile(xls_fullpath)
df = pd.read_excel(datasets_xls)


# %% LOOP THROUGH SATELLITE DATA
for dataset_id in id_list:
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
            opt_bands_use = ast.literal_eval(df.loc[(df['Dataset_key'] == dataset_id) 
                & (df['Processing_key'] == dataset_in), 'OPT_bands'].values[0])

            # If optical bands part of dataset, assume all are present and create dict
            # TODO 20200113 - Fix this assumption
            if opt_bands_use:
                # List of optical band names (added zero for correct alphabetical sorting)
                opt_band_names = ['b02','b03','b04','b05','b06','b07','b08','b08a','b11','b12']
                
                # Add OPT bands
                opt_bands_dict = dict()
                opt_bands_dict[dataset_use] = dict(zip(opt_band_names , opt_bands_use))
                
            print(sat_file)
        except:
            # Something went wrong
            continue
          
        # %% Try reading the dataset
        try:
            # Load data
            dataset = gdal.Open(sat_file)
            gdalinfo_log(dataset, log_type='default')
            
            # Read ALL bands - note that it will be zero indexed
            raster_data_array = dataset.ReadAsArray()
        except:
            # Something went wrong
            continue
        
        # %% Check input data type for feature extraction
        if dataset_in.lower()[0:5] in ['pgnlm']:
            if dataset_in.lower()[6:10] in ['2019', 'best']:
                c3_feature_type = 'iq2c3'
            else:
                c3_feature_type =  'c3_pgnlm2intensities'
        elif dataset_in.lower()[-2:] in ['c3']:
            c3_feature_type = 'c3_snap_intensities'
        else:
            print('No feature type found for: '+dataset_type)
            c3_feature_type = 'NA'
                
        # %% If SAR data should be added
        if sar_bands_use:
            # Get array with SAR data
            sar_data_temp = raster_data_array[sar_bands_use,:,:]   
            # Convert to 2D array
            sar_data_temp, n_rows, n_cols = imtensor2array(sar_data_temp)
            # Reshape to 3D image tensor (3 channels)
            sar_data_temp = np.reshape(sar_data_temp, (n_rows, n_cols, sar_data_temp.shape[1]))

        
        # %% If MULTISPECTRAL OPTICAL data should be added
        if dataset_in in opt_dataset_list:
            opt_bands_use = [] # Check which of the available bands should be included 
            for key in opt_bands_include:
                opt_bands_use.append(opt_bands_dict[dataset_use][key])
            opt_data_temp = raster_data_array[opt_bands_use,:,:]
            # Convert to 2D array
            opt_data_temp, n_rows, n_cols = imtensor2array(opt_data_temp)
            # Reshape to 3D image tensor (3 channels)
            opt_data_temp = np.reshape(opt_data_temp, (n_rows, n_cols, opt_data_temp.shape[1]))

            # Get OPT pixels
            opt_pixels = opt_data_temp
            # Add NDVI
            if add_ndvi:
                # TODO 20200113 - Clean up this!
                nir_pixels = opt_data_temp[:, :, opt_bands_include.index('b08')] 
                red_pixels = opt_data_temp[:, :, opt_bands_include.index('b04')]
                ndvi_pixels = (nir_pixels-red_pixels)/(nir_pixels+red_pixels)
        # %% Read lat and lon bands
        lat = raster_data_array[lat_band,:,:]
        lon = raster_data_array[lon_band,:,:]
        
        # Get defo AOI
        for i_defo, defo_aoi in enumerate(defo_aoi_list):
            # Get LAT and LONG for bounding rectangle and get pixel coordinates 
            lat_bounds, lon_bounds = defo_aoi.bounding_coords()
            row_ind, col_ind = geobox(lat_bounds, lon_bounds, lat, lon, margin=(0,0), log_type='default')
            x_min = row_ind[0]; x_max = row_ind[1]
            y_min = col_ind[0]; y_max = col_ind[1]
    
            # Get SAR data from area
            sat_data = raster_data_array[sar_bands_use, x_min:x_max, y_min:y_max]
            print(sat_data.shape)
            
            # Extract SAR covariance mat features, reshape to get channels last
            sat_data = np.transpose(sat_data, (1,2,0))
            sat_data = get_sar_features(sat_data, feature_type=c3_feature_type, input_type='img')
            
            # Flatten, reshape to channels first due to ordering of reshape
            sat_data = np.transpose(sat_data, (2,0,1))
            c_first_shape = sat_data.shape
            sat_data = np.reshape(sat_data, (c_first_shape[0],c_first_shape[1]*c_first_shape[2]), order='C')
            print(sat_data.shape)
            
            # Create a new array or append to existing one
            if i_defo == 0:
                defo_data = np.copy(sat_data)
                # Check that reshaping is correct 
                org_data = raster_data_array[[0,5,8], x_min:x_max, y_min:y_max]
                ref_data = defo_data.reshape(c_first_shape)
                print(org_data.shape)
                print(ref_data.shape)
                print(np.allclose(org_data, ref_data))
            else:
                defo_data = np.hstack((defo_data, sat_data))
        
        #%% Plot 3D backscatter values
        # Merge arrays with live and defoliated data
        
        # Plot
        modalitypoints3d('reciprocity', sat_data.transpose((1,0)), np.ones(length(sat_data),dtype='int'), labels_dict=None)
    
        
    

