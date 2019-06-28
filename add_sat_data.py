# -*- coding: utf-8 -*-
"""
@author: jorag
"""

import matplotlib.pyplot as plt
import numpy as np
import gdal
#from gdalconst import *
import tkinter
from tkinter import filedialog
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

# New structure:
    #-read field data object (perhaps try to read object with other sat data already added first)
    #-read satellite data
    #-loop through with: for point in DataModalities.point_name:
        #--read GPS coords
        #--get sat data for that coord
        #--store in DataModalities object for that point
    #-save updated object 


# List of datasets to process
#dataset_list = ['Coh-A', 'Coh-B', 'Coh-C', 'vanZyl-A', 'vanZyl-B', 'vanZyl-C']
#dataset_list = ['19-vanZyl-A', '19-Coh-A', '19-Quad-A']
#dataset_list = ['19-Quad-A']
dataset_list = ['PGNLM3-C']

# Prefix for output datamodalities object filename
datamod_fprefix = '19_nonorm'

# Normalization
opt_norm_type = 'none' # 'local' #   
sar_norm_type = 'none' # 'global'  #     

# Which Sentinel-2 bands to use
#opt_bands_include = ['b02','b03','b04','b05','b06','b07','b08','b08a','b11','b12']
opt_bands_include = ['b02','b03','b04']
    
# Path to working directory 
dirname = os.path.realpath('.') # For parent directory use '..'

# Name of input object and file with satellite data path string
obj_in_name = 'TEST_FIELD_DATA' + '.pkl'
                          
## Read data object
# Load DataModalities object
with open(os.path.join(dirname, 'data', obj_in_name), 'rb') as input:
    all_data = pickle.load(input)
                      

# ADD SATELLITE DATA
# Loop through all satellite images
for dataset_use in dataset_list:
    
    # Set name of output object
    # TODO: 20190628 Should objects now contain multiple different SAR data
    # - or, should each object contain only one data type?
    obj_out_name = datamod_fprefix + dataset_use + '.pkl'
    sat_pathfile_name = dataset_use + '-path'
    
    # Load data bands dictionary object
    with open(os.path.join(dirname, 'data', 'band_dicts'), 'rb') as input:
        sar_bands_dict, opt_bands_dict, geo_bands_dict = pickle.load(input)
                              
    # Geo bands
    lat_band = geo_bands_dict[dataset_use]['lat']
    lon_band = geo_bands_dict[dataset_use]['lon']
    # SAR bands
    sar_bands_use = sar_bands_dict[dataset_use]
    # OPT bands
    opt_bands_use = []
    for key in opt_bands_include:
        opt_bands_use.append(opt_bands_dict[dataset_use][key])
    
    # Read satellite data
    try:
        # Read predefined file
        with open(os.path.join(dirname, 'input-paths', sat_pathfile_name)) as infile:
            sat_file = infile.readline().strip()
            logit('Read file: ' + sat_file, log_type = 'default')
        
        # Load data
        dataset = gdal.Open(sat_file)
        gdalinfo_log(dataset, log_type='default')
    except:
        logit('Error, promt user for file.', log_type = 'default')
        # Predefined file failed for some reason, promt user
        root = tkinter.Tk() # GUI for file selection
        root.withdraw()
        sat_file = tkinter.filedialog.askopenfilename(title='Select input .tif file')
        # Load data
        dataset = gdal.Open(sat_file)
        gdalinfo_log(dataset, log_type='default')
    
    # Read ALL bands - note that it will be zero indexed
    raster_data_array = dataset.ReadAsArray()
    
    # Get array with SAR data
    sar_data_temp = raster_data_array[sar_bands_use,:,:]   
    # Convert to 2D array
    sar_data_temp, n_rows, n_cols = imtensor2array(sar_data_temp)
    # Normalize data
    sar_data_temp = norm01(sar_data_temp, norm_type=sar_norm_type, log_type='print')
    # Reshape to 3D image tensor (3 channels)
    sar_data_temp = np.reshape(sar_data_temp, (n_rows, n_cols, sar_data_temp.shape[1]))
    # SAR info to add to object
    kw_sar = dict([['bands_use', sar_bands_use]])
    
    
    # Get array with MULTISPECTRAL OPTICAL data
    opt_data_temp = raster_data_array[opt_bands_use,:,:]
    # Convert to 2D array
    opt_data_temp, n_rows, n_cols = imtensor2array(opt_data_temp)
    # Normalize data
    opt_data_temp = norm01(opt_data_temp, norm_type=opt_norm_type, log_type='print')
    # Reshape to 3D image tensor (3 channels)
    opt_data_temp = np.reshape(opt_data_temp, (n_rows, n_cols, opt_data_temp.shape[1]))

    
    # OPT info to add to object
    kw_opt = dict([['bands_use', opt_bands_use]])
        
    # Read GPS coord
    for point in all_data.point_name:
        coord = all_data.read_point(point, 'gps_coordinates')
        print(coord)
        # Get pixel positions from my geopixpos module
        x_p, y_p = geocoords2pix(raster_data_array[lat_band,:,:], 
                                 raster_data_array[lon_band,:,:], 
                                 lat=coord[0], lon=coord[1], pixels_out ='npsingle')
        print(x_p, y_p)
        # Get SAR pixels
        sar_pixels = sar_data_temp[x_p.T, y_p.T, :] 
        # Add SAR modality
        all_data.add_modality(point, 'quad_pol', sar_pixels.tolist(), **kw_sar)
        
        # Get OPT pixels
        opt_pixels = opt_data_temp[x_p.T, y_p.T, :] 
        # Add OPT modality
        all_data.add_modality(point, 'optical', opt_pixels.tolist(), **kw_opt)
    
    ## Print points
    all_data.print_points()
    
    # Save DataModalities object
    with open(os.path.join(dirname, 'data', obj_out_name), 'wb') as output:
        pickle.dump(all_data, output, pickle.HIGHEST_PROTOCOL)
