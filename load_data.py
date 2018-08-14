# -*- coding: utf-8 -*-
"""
@author: jorag
"""

import matplotlib.pyplot as plt
import numpy as np
import gdal
#from gdalconst import *
import os, sys, time
#from src import dispms
import math
import tkinter
from tkinter import filedialog
import struct # for converting scanline raster data to numeric

# GUI for file selection
root = tkinter.Tk()
root.withdraw()
file_path = tkinter.filedialog.askopenfilename(title='Select input .tif file')
#root.destroy() #is this needed?

# Files could be loaded using SNAPPY import product, but for now assuming that the input is .tiff is ok

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
ul_lat = geotransform[3] # lat or y??
ul_lon = geotransform[0] # long or x??
pixel_width_lat = geotransform[5] # lat or y??
pixel_width_lon = geotransform[1] # long or x??

if geotransform:
    print("Origin = ({}, {})".format(geotransform[0], geotransform[3]))
    print("Pixel Size = ({}, {})".format(geotransform[1], geotransform[5]))

    
# Load a band
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
xOffset = 0
yOffset = 0
data = band.ReadAsArray(xOffset, yOffset, 10, 10)

# Point of Interest
point_lat = 70.0
point_lon = 27.0

# Estimate pixel position
point_pixpos_lat = (point_lat - ul_lat)/pixel_width_lat # Better way of ensuring correct sign? 
point_pixpos_lon = (ul_lon - point_lon)/pixel_width_lon # Better way of ensuring correct sign?

# Idea: Use the Geo-transform info to get a rough estimate of position, then look up "exact" position using LAT/LONG bands
# Look at sign of band minus coordinate of interest
