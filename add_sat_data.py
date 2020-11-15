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
import ast
# My moduels
from mytools import *
from geopixpos import *
from visandtest import *
from dataclass import *


# Check if the new band list format, where everything is stored in a .xls, is used
new_datalist_format = True
new_datalist_xls_file = 'SF_forest_subsets.xls' # '2020_C3_dataset_overview.xls' # '2019_reprocess_dataset_overview.xls'
# Prefix for output datamodalities object filename
datamod_fprefix = 'PGNLM-NLSAR_C3_20201115' #'PGNLM-SNAP_C3_geo_OPT_20200113'
base_obj_name = 'DiffGPS_FIELD_DATA'+'.pkl' # Name of the (pure) field data object everything is based on 

# List of datasets to process
#dataset_list = ['iq', 'C3', 'cloude_3x3', 'genFD_3x3', 'vanZyl_3x3', 'yamaguchi_3x3', 'collocate_iq', 'collocate_C3', 'pgnlm_iq'] 
#dataset_list = ['C3', 'refined_Lee_5x5_C3', 'boxcar_5x5_C3', 'IDAN_50_C3', 'geo_opt'] 
dataset_list = ['PGNLM_19-2_v4', 'NLSAR_1_1', 'refined_Lee_5x5_C3', 'boxcar_5x5_C3', 'IDAN_50_C3', 'geo_opt']
id_list = ['A', 'C'] #['A', 'B', 'C'] # TODO: 20190909 Consider changing this a date string
add_ndvi = True

# Datasets to add optical bands from
opt_dataset_list = ['geo_opt']

# Which Sentinel-2 bands to use
#opt_bands_include = ['b02','b03','b04','b05','b06','b07','b08','b08a','b11','b12']
opt_bands_include = ['b02','b03','b04','b05','b08'] # b02, b03, b04, b08, all 10 m resolution
    
# Path to working directory 
dirname = os.path.realpath('.') # For parent directory use '..'

# Name of input object and file with satellite data path string
obj_out_name = datamod_fprefix+'.pkl' 

# Create object from scratch OR add to existing object
root = tkinter.Tk()
root.withdraw()
result1 = tkmb.askquestion('Create new object?', 'If not, data will be added to existing one', icon='warning')
if result1 == 'yes':
    obj_in_name = base_obj_name
else:
    obj_in_name = obj_out_name
#root.destroy()   
                          
## Read DataModalities object with ground in situ vegetation data
with open(os.path.join(dirname, 'data', obj_in_name), 'rb') as input:
    all_data = pickle.load(input)
                      
if new_datalist_format:
    # Load band lists from Excel file
    xls_fullpath = os.path.join(dirname, 'input-paths', new_datalist_xls_file)
    datasets_xls = pd.ExcelFile(xls_fullpath)
    df = pd.read_excel(datasets_xls)
else:
    # Load data bands dictionaries
    with open(os.path.join(dirname, 'data', 'band_dicts'), 'rb') as input:
        sar_bands_dict, opt_bands_dict, geo_bands_dict = pickle.load(input)


# ADD SATELLITE DATA
# Loop through all satellite images
for dataset_id in id_list:
    for dataset_in in dataset_list:
        dataset_use = dataset_in +'-'+dataset_id 
            
        # Set name of output object
        if new_datalist_format:
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
        else:
            sat_pathfile_name = dataset_use + '-path'
            print(sat_pathfile_name)

            # Get indices for bands, lat, lon, SAR, and optical
            lat_band = geo_bands_dict[dataset_use]['lat']
            lon_band = geo_bands_dict[dataset_use]['lon']
            sar_bands_use = sar_bands_dict[dataset_use]
            
            # Read satellite data specified by input-path file
            with open(os.path.join(dirname, 'input-paths', sat_pathfile_name)) as infile:
                sat_file = infile.readline().strip()
                logit('Read file: ' + sat_file, log_type = 'default')
          
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
            # SAR info to add to object
            kw_sar = dict([['bands_use', sar_bands_use]])
        
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
            # OPT info to add to object
            kw_opt = dict([['bands_use', opt_bands_use]])
            
        
        # Read GPS coord and add data from that coordinate
        for point in all_data.point_name:
            coord = all_data.read_point(point, 'gps_coordinates')
            # Get pixel positions from my geopixpos module
            x_p, y_p = geocoords2pix(raster_data_array[lat_band,:,:], 
                                     raster_data_array[lon_band,:,:], 
                                     lat=coord[0], lon=coord[1], pixels_out ='npsingle')
#            # DEBUG - 20191113: Check offset:
#            coord2 = all_data.read_point(point, 'waypoint_coordinates')
#            # Get pixel positions from my geopixpos module
#            x_p2, y_p2 = geocoords2pix(raster_data_array[lat_band,:,:], 
#                                     raster_data_array[lon_band,:,:], 
#                                     lat=coord2[0], lon=coord2[1], pixels_out ='npsingle')
#            print(point, x_p-x_p2, y_p-y_p2)
            
            # If SAR data should be added
            if sar_bands_use:
                # Get SAR pixels
                sar_pixels = sar_data_temp[x_p, y_p, :] 
                # Add SAR modality
                all_data.add_modality(point, dataset_in, sar_pixels.tolist(), dataset_use, **kw_sar)
            
            # If MULTISPECTRAL OPTICAL data should be added
            if dataset_in in opt_dataset_list:
                # Get OPT pixels
                opt_pixels = opt_data_temp[x_p, y_p, :] 
                # Add OPT modality
                all_data.add_modality(point, 'optical', opt_pixels.tolist(), 'optical-'+dataset_id, **kw_opt)
                # Add NDVI
                if add_ndvi:
                    # TODO 20200113 - Clean up this!
                    kw_ndvi = dict([['bands_use', ['b04', 'b08']]])
                    nir_pixels = opt_data_temp[x_p, y_p, opt_bands_include.index('b08')] 
                    red_pixels = opt_data_temp[x_p, y_p, opt_bands_include.index('b04')]
                    ndvi_pixels = (nir_pixels-red_pixels)/(nir_pixels+red_pixels)
                    all_data.add_modality(point, 'NDVI', ndvi_pixels.tolist(), 'NDVI-'+dataset_id, **kw_ndvi)
    

## Print points
#all_data.print_points()

# Save DataModalities object
with open(os.path.join(dirname, 'data', obj_out_name), 'wb') as output:
    pickle.dump(all_data, output, pickle.HIGHEST_PROTOCOL)
