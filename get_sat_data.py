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
#import sys # To append paths
# My moduels
from mytools import *
from geopixpos import *
from visandtest import *
from dataclass import *


# Intialize data object
all_data = DataModalities('Polmak')

dirname = os.path.realpath('.') # For parent directory use '..'

# Classify LIVE FOREST vs. DEAD FOREST vs. OTHER
# This function: Return data array? 
# Files could be loaded using SNAPPY import product, but for now assuming that the input is .tiff is ok


# Define class numbers


# Read satellite data
try:
    # Read predefined file
    with open(os.path.join(dirname, "data", "sat-data-path")) as infile:
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
            


# Get georeference info
geotransform = dataset.GetGeoTransform()
    
# Load a band, 1 based
band = dataset.GetRasterBand(1)
bandinfo_log(band, log_type='default')

# Read multiple bands
all_sat_bands = dataset.ReadAsArray()

# Show point of Interest
point_lat = 70.0
point_lon = 27.0
showimpoint(all_sat_bands, geotransform, point_lat, point_lon, n_pixel_x=500, n_pixel_y=500, bands=[0,1,2])

# Show entire image
showimage(all_sat_bands, bands=[0,1,2])
    
# Read .gpx file with coordinates of transect points
try:
    # Read predefined file
    with open(os.path.join(dirname, "data", "gps-data-path")) as infile:
        gps_file = infile.readline().strip()
        logit('Read file: ' + gps_file, log_type = 'default')
    
    # Load data
    tree = ET.parse(gps_file)
except:      
    logit('Error, promt user for file.', log_type = 'default')
    # Predefined file failed for some reason, promt user
    root = tkinter.Tk() # GUI for file selection
    root.withdraw()
    gps_file = tkinter.filedialog.askopenfilename(title='Select input .gpx file')
    # Load data
    tree = ET.parse(gps_file)

# Get lat and long
pos_array = []
for elem in tree.findall("{http://www.topografix.com/GPX/1/1}wpt"):
    lon, lat = elem.attrib['lon'], elem.attrib['lat']
    pos_array.append([float(lat), float(lon)])
# Get name of waypoints
gps_id = []
for elem in tree.findall("//{http://www.topografix.com/GPX/1/1}name"):
    gps_id.append(elem.text)

# Merge names and positions
gps_points = list(zip(gps_id, pos_array))
# Convert to numpy array
#pos_array2 = np.asarray(pos_array)
pos_array2 = np.asarray([item[1] for item in gps_points])
gps_id2 = [item[0] for item in gps_points]


# Read Excel file with vegetation types
try:
    # Read predefined file
    with open(os.path.join(dirname, "data", "vegetation-data-path")) as infile:
        veg_file = infile.readline().strip()
        logit('Read file: ' + veg_file, log_type = 'default')
    
    # Load data
    xls = pd.ExcelFile(veg_file)
except:      
    logit('Error, promt user for file.', log_type = 'default')
    # Predefined file failed for some reason, promt user
    root = tkinter.Tk() # GUI for file selection
    root.withdraw()
    veg_file = tkinter.filedialog.askopenfilename(title='Select input .csv/.xls(x) file')
    xls = pd.ExcelFile(veg_file)


   
# Categorize points


# Get pixel positions from my geopixpos module
pix_lat, pix_long = pos2pix(geotransform, lon=pos_array2[:,1], lat=pos_array2[:,0], pixels_out = 'npsingle', verbose=True)

# Extract pixels from area
data_out = all_sat_bands[0:3, pix_lat, pix_long]

# Go through all sheets in Excel sheet
point_info = []
for i_sheet in range(1,7):
    print(i_sheet)
    # Get pandas dataframe
    df = pd.read_excel(xls, str(i_sheet))
    point_id = list(df['GPSwaypoint'])
    # Go through the list of points
    for id in point_id:
        print(id)
        #print(gps_id.index(id), df['LCT1_2017'][point_id.index(id)], pos_array[gps_id.index(id)])
        point_info.append([gps_id.index(id), df['LCT1_2017'][point_id.index(id)], pos_array[gps_id.index(id)]])