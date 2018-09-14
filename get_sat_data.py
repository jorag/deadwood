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
# My moduels
from mytools import *
from geopixpos import *


# Classify LIVE FOREST vs. DEAD FOREST vs. OTHER
# This function: Return data array? 
# Files could be loaded using SNAPPY import product, but for now assuming that the input is .tiff is ok


# Global options
refine_pixpos = False # using lat/long bands 
lat_band_i = 5

# GUI for file selection
root = tkinter.Tk()
root.withdraw()
file_path = tkinter.filedialog.askopenfilename(title='Select input .tif file')
#root.destroy() #is this needed?

# Load data
dataset = gdal.Open(file_path)


# Print information - can also use command line: !gdalinfo file_path 
print("Driver: {}/{}".format(dataset.GetDriver().ShortName,
                             dataset.GetDriver().LongName))
print("Size is {} x {} x {}".format(dataset.RasterXSize,
                                    dataset.RasterYSize,
                                    dataset.RasterCount))
print("Projection is {}".format(dataset.GetProjection()))

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


# Read Excel file with coordinates
terrain_class_file = tkinter.filedialog.askopenfilename(title='Select input .tif file')
#df = pd.read_excel(terrain_class_file)
xls = pd.ExcelFile(terrain_class_file)
df1 = pd.read_excel(xls, '1')

# Read raster data as array
# From https://www.gis.usu.edu/~chrisg/python/2009/lectures/ospy_slides4.pdf
xOffset = 1000
yOffset = 1000
data = band.ReadAsArray(xOffset, yOffset, 10, 10)

# Point of Interest
point_lat = 70.0
point_lon = 27.0

# Try using the pos2pix function from my geopixpos module
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