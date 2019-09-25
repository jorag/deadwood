# -*- coding: utf-8 -*-
"""
@author: jorag
"""

import matplotlib.pyplot as plt
import numpy as np
import gdal
import tkinter
from tkinter import filedialog
import tkinter.messagebox as tkmb
import pandas as pd
import os # Necessary for relative paths
import xml.etree.ElementTree as ET
import pickle
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

# Prefix for output datamodalities object filename
datamod_fprefix = 'All-data-0919'

# List of datasets to process
#dataset_list = ['Coh-A', 'Coh-B', 'Coh-C', 'vanZyl-A', 'vanZyl-B', 'vanZyl-C']
#dataset_list = ['19-vanZyl-A', '19-Coh-A', '19-Quad-A']
dataset_list = ['19-Quad', 'PGNLM3', 'Coh', 'vanZyl', 'Quad', 'GNLM']
#dataset_list = ['Quad']
#dataset_id = 'C' # TODO: 20190909 Consider changing this a date string

# Datasets to add optical bands from
opt_dataset_list = ['vanZyl']

# Which Sentinel-2 bands to use
opt_bands_include = ['b02','b03','b04','b05','b06','b07','b08','b08a','b11','b12']
    
# Path to working directory 
dirname = os.path.realpath('.') # For parent directory use '..'

# Name of input object and file with satellite data path string
obj_in_name = 'NEW_FIELD_DATA' + '.pkl'
obj_out_name = datamod_fprefix + '-' + '.pkl'  

# Add to existing object or create from scratch
root = tkinter.Tk()
root.withdraw()
result1 = tkmb.askquestion('Create new object?', 'If not, data will be added to existing one', icon='warning')
if result1 == 'yes':
    obj_in_name = 'NEW_FIELD_DATA' + '.pkl'
else:
    obj_in_name = obj_out_name
#root.destroy()   
                          
## Read DataModalities object with ground in situ vegetation data
with open(os.path.join(dirname, 'data', obj_in_name), 'rb') as input:
    all_data = pickle.load(input)
                      
# Load data bands dictionaries
with open(os.path.join(dirname, 'data', 'band_dicts'), 'rb') as input:
    sar_bands_dict, opt_bands_dict, geo_bands_dict = pickle.load(input)

# ADD SATELLITE DATA
# Loop through all satellite images
for dataset_id in ['A', 'B', 'C']:
    for dataset_in in dataset_list:
        
        # Set name of output object
        dataset_use = dataset_in + '-' + dataset_id 
        sat_pathfile_name = dataset_use + '-path'
        print(sat_pathfile_name)
        # Try reading the dataset
        try:
            # Get indices for bands, lat, lon, SAR, and optical
            lat_band = geo_bands_dict[dataset_use]['lat']
            lon_band = geo_bands_dict[dataset_use]['lon']
            sar_bands_use = sar_bands_dict[dataset_use]
            
            # Read satellite data specified by input-path file
            with open(os.path.join(dirname, 'input-paths', sat_pathfile_name)) as infile:
                sat_file = infile.readline().strip()
                logit('Read file: ' + sat_file, log_type = 'default')
            
            # Load data
            dataset = gdal.Open(sat_file)
            gdalinfo_log(dataset, log_type='default')
            
            # Read ALL bands - note that it will be zero indexed
            raster_data_array = dataset.ReadAsArray()
        except:
            # Something went wrong, zero out some variables to ensure no follow up errors
            sar_bands_use = []
            continue
        
        # Get array with SAR data
        sar_data_temp = raster_data_array[sar_bands_use,:,:]   
        # Convert to 2D array
        sar_data_temp, n_rows, n_cols = imtensor2array(sar_data_temp)
        # Reshape to 3D image tensor (3 channels)
        sar_data_temp = np.reshape(sar_data_temp, (n_rows, n_cols, sar_data_temp.shape[1]))
        # SAR info to add to object
        kw_sar = dict([['bands_use', sar_bands_use]])
        
        # Get array with MULTISPECTRAL OPTICAL data
        # Check to see if optical data should be included
        if dataset_in in opt_dataset_list:
            opt_bands_use = [] # Check which of the available bands should be included 
            for key in opt_bands_include:
                opt_bands_use.append(opt_bands_dict[dataset_use][key])
            opt_data_temp = raster_data_array[opt_bands_use,:,:]
            # Convert to 2D array
            opt_data_temp, n_rows, n_cols = imtensor2array(opt_data_temp)
            # Reshape to 3D image tensor (3 channels)
            opt_data_temp = np.reshape(opt_data_temp, (n_rows, n_cols, opt_data_temp.shape[1]))
            # OPT info to add to object
            kw_opt = dict([['bands_use', opt_bands_use]])
            
        
        # Read GPS coord and add data from that coordinate
        for point in all_data.point_name:
            coord = all_data.read_point(point, 'gps_coordinates')
            # Get pixel positions from my geopixpos module
            x_p, y_p = geocoords2pix(raster_data_array[lat_band,:,:], 
                                     raster_data_array[lon_band,:,:], 
                                     lat=coord[0], lon=coord[1], pixels_out ='npsingle')
    
            # Get SAR pixels
            sar_pixels = sar_data_temp[x_p, y_p, :] 
            # Add SAR modality
            # TODO: 20190911 - change dataset_use to dataset_id to keep multiple datasets in a single object:
            all_data.add_modality(point, dataset_in, sar_pixels.tolist(), dataset_use, **kw_sar)
            
            if dataset_in in opt_dataset_list:
                # Get OPT pixels
                opt_pixels = opt_data_temp[x_p, y_p, :] 
                # Add OPT modality
                all_data.add_modality(point, 'optical', opt_pixels.tolist(), dataset_use, **kw_opt)
    

## Print points
#all_data.print_points()

# Save DataModalities object
with open(os.path.join(dirname, 'data', obj_out_name), 'wb') as output:
    pickle.dump(all_data, output, pickle.HIGHEST_PROTOCOL)
