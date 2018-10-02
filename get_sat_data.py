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


dirname = os.path.realpath('.') # For parent directory use '..'

# Classify LIVE FOREST vs. DEAD FOREST vs. OTHER
# This function: Return data array? 
# Files could be loaded using SNAPPY import product, but for now assuming that the input is .tiff is ok

# Global options
refine_pixpos = False # using lat/long bands 

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
print("Band Type={}".format(gdal.GetDataTypeName(band.DataType)))

# Get pixel info
pixel_min = band.GetMinimum()
pixel_max = band.GetMaximum()
if not pixel_min or not pixel_max:
    (pixel_min,pixel_max) = band.ComputeRasterMinMax(True)
print("Min={:.3f}, Max={:.3f}".format(pixel_min, pixel_max))
      
if band.GetOverviewCount() > 0:
    print("Band, number of overviews:")
    print(band.GetOverviewCount())
      
if band.GetRasterColorTable():
    print("Band, number of colour table with entries:")
    print(band.GetRasterColorTable().GetCount())

    
# Read raster data as array
# From https://www.gis.usu.edu/~chrisg/python/2009/lectures/ospy_slides4.pdf
xOffset = 1000
yOffset = 1000
data = band.ReadAsArray(xOffset, yOffset, 10, 10)

# Point of Interest
point_lat = 70.0
point_lon = 27.0

# Use the pos2pix function from my geopixpos module to find pixel indice
pix_lat, pix_long = pos2pix(geotransform, lon=point_lon, lat=point_lat, pixels_out = 'single', verbose=True)


# Read multiple bands
all_data = dataset.ReadAsArray()

# Extract pixels around image
n_pixel_x = 500
n_pixel_y = 500

# Extract pixels from area
im_generate = all_data[0:3, int(pix_lat-n_pixel_x/2):int(pix_lat+n_pixel_x/2), int(pix_long-n_pixel_y/2):int(pix_long+n_pixel_y/2)]

# Rearrage dimensions to x,y,RGB format
im_generate = np.transpose(im_generate, (1,2,0))
plt.figure()
plt.imshow(im_generate) 
plt.show()  # display it

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

# Go through all sheets in Excel sheet
for i_sheet in range(1,7):
    print(i_sheet)
    # Get pandas dataframe
    df = pd.read_excel(xls, str(i_sheet))
    point_id = list(df['GPSwaypoint'])
    # Go through the list of points
    for id in point_id:
        #print(id)
        print(gps_id.index(id), df['LCT1_2017'][point_id.index(id)], pos_array[gps_id.index(id)])
   
# Categorize points


# Get pixel positions
pix_lat, pix_long = pos2pix(geotransform, lon=pos_array2[:,1], lat=pos_array2[:,0], pixels_out = 'npsingle', verbose=True)
