#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 12:43:21 2019

@author: jorag

# Quick script to store index for bands to use from GeoTIFF products
"""
import numpy as np
import pickle
import os # Necessary for relative paths
#import sys # To append paths
# My moduels
from mytools import *


# Path to working directory 
dirname = os.path.realpath('.') # For parent directory use '..'

# Initialize dictionaries
sar_bands_dict = dict()
opt_bands_dict = dict()
geo_bands_dict = dict()

# NOTE: WHEN ALL BANDS ARE READ, PYTHON'S 0 BASED INDEXING MUST BE USED IN ARRAY

# Add SAR bands
# van Zyl decomposition
sar_bands_dict['vanZyl-A'] = [0, 1, 2]
sar_bands_dict['vanZyl-B'] = [0, 1, 2]
sar_bands_dict['vanZyl-C'] = [0, 1, 2]
sar_bands_dict['19-vanZyl-A'] = [0, 1, 2]
# Coherence bands: t11 = 11, t22 = 16, t33 = 19
sar_bands_dict['Coh-A'] = [34,35,36,37,38,39,40,41,42] # [34, 39, 42] # 
sar_bands_dict['Coh-B'] = [10,11,12,13,14,15,16,17,18] # [10, 15, 18] # 
sar_bands_dict['Coh-C'] = [10,11,12,13,14,15,16,17,18] # [10, 15, 18] # 
sar_bands_dict['19-Coh-A'] = [0,1,2,3,4,5,6,7,8] # [34, 39, 42] # 
# Calibrated backscatter bands
sar_bands_dict['19-Quad-A'] = [0, 1, 2, 3]
sar_bands_dict['19-Quad-C'] = [2, 5, 8, 11]
sar_bands_dict['Quad-C'] = [2, 5, 8, 11]
# Filtered and raw HH/HV/VH/VV
sar_bands_dict['GNLM-A'] = [0, 1, 2, 3, 17, 18, 19, 20]
sar_bands_dict['GNLM-B'] = [0, 1, 2, 3, 17, 18, 19, 20]
# PGNLM test
sar_bands_dict['PGNLM1-C'] = [0, 1, 2]
sar_bands_dict['PGNLM3-C'] = [0, 1, 2]


# List of optical band names (added zero for correct alphabetical sorting)
opt_band_names = ['b02','b03','b04','b05','b06','b07','b08','b08a','b11','b12']

# Add OPT bands
opt_bands_dict['vanZyl-A'] = dict(zip(opt_band_names , [3,4,5,6,7,8,9,10,11,12]))
opt_bands_dict['vanZyl-B'] = dict(zip(opt_band_names , [3,4,5,6,7,8,9,10,11,12]))
opt_bands_dict['vanZyl-C'] = dict(zip(opt_band_names , [3,4,5,6,7,8,9,10,11,12]))
opt_bands_dict['19-vanZyl-A'] = dict(zip(opt_band_names , [0,0,0,1,1,1,2,2,2,2])) # CHANGE THIS WHEN OPT BANDS ADDED TO PRODUCT
opt_bands_dict['Coh-A'] = dict(zip(opt_band_names , [0,1,2,3,4,5,6,7,8,9]))
opt_bands_dict['Coh-B'] = dict(zip(opt_band_names , [0,1,2,3,4,5,6,7,8,9]))
opt_bands_dict['Coh-C'] = dict(zip(opt_band_names , [0,1,2,3,4,5,6,7,8,9]))
opt_bands_dict['19-Coh-A'] = dict(zip(opt_band_names , [0,1,2,3,4,5,6,7,8,0])) # CHANGE THIS WHEN OPT BANDS ADDED TO PRODUCT
opt_bands_dict['19-Quad-A'] = dict(zip(opt_band_names , [0,0,0,1,1,1,2,2,3,3])) # CHANGE THIS WHEN OPT BANDS ADDED TO PRODUCT
opt_bands_dict['19-Quad-C'] = dict(zip(opt_band_names , [0,1,2,3,4,5,6,7,8,0])) # CHANGE THIS WHEN OPT BANDS ADDED TO PRODUCT
opt_bands_dict['Quad-C'] = dict(zip(opt_band_names , [14,15,16,17,18,19,20,21,22,23])) 
opt_bands_dict['GNLM-A'] = dict(zip(opt_band_names , [6,7,8,9,10,11,12,13,14,15]))
opt_bands_dict['GNLM-B'] = dict(zip(opt_band_names , [6,7,8,9,10,11,12,13,14,15]))
opt_bands_dict['PGNLM1-C'] = dict(zip(opt_band_names , [3,4,5,3,4,5,3,4,5,5])) # CHANGE THIS WHEN all OPT BANDS ADDED TO PRODUCT
opt_bands_dict['PGNLM3-C'] = dict(zip(opt_band_names , [3,4,5,3,4,5,3,4,5,5])) # CHANGE THIS WHEN all OPT BANDS ADDED TO PRODUCT
              
# Add GEO bands
geo_bands_dict['vanZyl-A'] = dict(zip(['lat', 'lon'], [38, 39]))
geo_bands_dict['vanZyl-B'] = dict(zip(['lat', 'lon'], [14, 15]))
geo_bands_dict['vanZyl-C'] = dict(zip(['lat', 'lon'], [14, 15]))
geo_bands_dict['19-vanZyl-A'] = dict(zip(['lat', 'lon'], [3, 4]))
geo_bands_dict['Coh-A'] = dict(zip(['lat', 'lon'], [44, 45]))
geo_bands_dict['Coh-B'] = dict(zip(['lat', 'lon'], [20, 21]))
geo_bands_dict['Coh-C'] = dict(zip(['lat', 'lon'], [20, 21]))
geo_bands_dict['19-Coh-A'] = dict(zip(['lat', 'lon'], [9, 10]))
geo_bands_dict['19-Quad-A'] = dict(zip(['lat', 'lon'], [4, 5]))
geo_bands_dict['19-Quad-C'] = dict(zip(['lat', 'lon'], [12, 13]))
geo_bands_dict['Quad-C'] = dict(zip(['lat', 'lon'], [12, 13]))
geo_bands_dict['GNLM-A'] = dict(zip(['lat', 'lon'], [4, 5]))
geo_bands_dict['GNLM-B'] = dict(zip(['lat', 'lon'], [4, 5]))
geo_bands_dict['PGNLM1-C'] = dict(zip(['lat', 'lon'], [6, 7]))
geo_bands_dict['PGNLM3-C'] = dict(zip(['lat', 'lon'], [6, 7]))

# Save DataModalities object
with open(os.path.join(dirname, 'data', 'band_dicts'), 'wb') as output:
    pickle.dump([sar_bands_dict, opt_bands_dict, geo_bands_dict], output, pickle.HIGHEST_PROTOCOL)