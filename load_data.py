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
file_path = tkinter.filedialog.askopenfilename()
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
geotransform = dataset.GetGeoTransform()
if geotransform:
    print("Origin = ({}, {})".format(geotransform[0], geotransform[3]))
    print("Pixel Size = ({}, {})".format(geotransform[1], geotransform[5]))

    
# Load a band
band = dataset.GetRasterBand(1)
print("Band Type={}".format(gdal.GetDataTypeName(band.DataType)))
      
min = band.GetMinimum()
max = band.GetMaximum()
if not min or not max:
    (min,max) = band.ComputeRasterMinMax(True)
print("Min={:.3f}, Max={:.3f}".format(min,max))
      
if band.GetOverviewCount() > 0:
    print("Band, number of overviews:")
    print(band.GetOverviewCount())
      
if band.GetRasterColorTable():
    print("Band, number of colour table with entries:")
    print(band.GetRasterColorTable().GetCount())


# Read raster data
scanline = band.ReadRaster(xoff=0, yoff=0,
                           xsize=band.XSize, ysize=1,
                           buf_xsize=band.XSize, buf_ysize=1,
                           buf_type=gdal.GDT_Float32)

tuple_of_floats = struct.unpack('f' * b2.XSize, scanline)
