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
dataset_list = ['geo_opt']
id_list = ['A', 'C'] #['A', 'B', 'C'] # TODO: 20190909 Consider changing this a date string
add_ndvi = True

# Datasets to add optical bands from
opt_dataset_list = ['geo_opt']

# Which Sentinel-2 bands to use
#opt_bands_include = ['b02','b03','b04','b05','b06','b07','b08','b08a','b11','b12']
opt_bands_include = ['b02','b03','b04','b05','b08'] # b02, b03, b04, b08, all 10 m resolution
    
# Path to working directory 
dirname = os.path.realpath('.') # For parent directory use '..'

                      
# Load band lists from Excel file
xls_fullpath = os.path.join(dirname, 'input-paths', new_datalist_xls_file)
datasets_xls = pd.ExcelFile(xls_fullpath)
df = pd.read_excel(datasets_xls)



# ADD SATELLITE DATA
# Loop through all satellite images
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
          
        # Try reading the dataset
        try:
            # Load data
            dataset = gdal.Open(sat_file)
            gdalinfo_log(dataset, log_type='default')
            
            # Read ALL bands - note that it will be zero indexed
            # 20191127: IF-test here to switch between reading data from file, 
            # or use output (possibly from memory) from filtering
            raster_data_array = dataset.ReadAsArray()
        except:
            # Something went wrong
            continue
        
        # If SAR data should be added
        if sar_bands_use:
            # Get array with SAR data
            sar_data_temp = raster_data_array[sar_bands_use,:,:]   
            # Convert to 2D array
            sar_data_temp, n_rows, n_cols = imtensor2array(sar_data_temp)
            # Reshape to 3D image tensor (3 channels)
            sar_data_temp = np.reshape(sar_data_temp, (n_rows, n_cols, sar_data_temp.shape[1]))

        
        # If MULTISPECTRAL OPTICAL data should be added
        if dataset_in in opt_dataset_list:
            opt_bands_use = [] # Check which of the available bands should be included 
            for key in opt_bands_include:
                opt_bands_use.append(opt_bands_dict[dataset_use][key])
            opt_data_temp = raster_data_array[opt_bands_use,:,:]
            # Convert to 2D array
            opt_data_temp, n_rows, n_cols = imtensor2array(opt_data_temp)
            # Reshape to 3D image tensor (3 channels)
            opt_data_temp = np.reshape(opt_data_temp, (n_rows, n_cols, opt_data_temp.shape[1]))
            

        # If MULTISPECTRAL OPTICAL data should be added
        if dataset_in in opt_dataset_list:
            # Get OPT pixels
            opt_pixels = opt_data_temp
            # Add NDVI
            if add_ndvi:
                # TODO 20200113 - Clean up this!
                nir_pixels = opt_data_temp[:, :, opt_bands_include.index('b08')] 
                red_pixels = opt_data_temp[:, :, opt_bands_include.index('b04')]
                ndvi_pixels = (nir_pixels-red_pixels)/(nir_pixels+red_pixels)
    

