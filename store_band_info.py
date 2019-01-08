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

# PARAMETERS
# Geo bands: A lat = 45, lon = 46. Dataset B & C: lat = 21, lon = 22
lat_band = 20 # 21-1
lon_band = 21 # 22-1

# SAR bands: t11 = 11, t22 = 16, t33 = 19
# NOTE: WHEN ALL BANDS ARE READ, PYTHON'S 0 BASED INDEXING MUST BE USED IN ARRAY
sar_bands_use = [10, 15, 18]



# Initialize dictionaries
sar_bands_dict = dict()
opt_bands_dict = dict()
geo_bands_dict = dict()

# Add SAR bands
sar_bands_dict['vanZyl-A'] = [0, 1, 2]
sar_bands_dict['vanZyl-B'] = [0, 1, 2]
sar_bands_dict['vanZyl-C'] = [0, 1, 2]


# Add OPT bands
opt_bands_dict['vanZyl-A'] = dict(zip(['b02','b03','b04','b05','b06','b07','b08','b08a','b11','b12'], [3,4,5,6,7,8,9,10,11,12]))
opt_bands_dict['vanZyl-B'] = dict(zip(['b02','b03','b04','b05','b06','b07','b08','b08a','b11','b12'], [3,4,5,6,7,8,9,10,11,12]))
opt_bands_dict['vanZyl-C'] = dict(zip(['b02','b03','b04','b05','b06','b07','b08','b08a','b11','b12'], [3,4,5,6,7,8,9,10,11,12]))

# Add GEO bands
geo_bands_dict['vanZyl-A'] = dict(zip(['lat', 'lon'], [38, 39]))
geo_bands_dict['vanZyl-B'] = dict(zip(['lat', 'lon'], [14, 15]))
geo_bands_dict['vanZyl-C'] = dict(zip(['lat', 'lon'], [14, 15]))

# Save DataModalities object
with open(os.path.join(dirname, 'data', 'band_dicts'), 'wb') as output:
    pickle.dump([sar_bands_dict, opt_bands_dict, geo_bands_dict], output, pickle.HIGHEST_PROTOCOL)