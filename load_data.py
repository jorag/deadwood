import matplotlib.pyplot as plt
import numpy as np
#import gdal
#from gdalconst import *
import os, sys, time
#from src import dispms
import math
import tkinter
from tkinter import filedialog


root = tkinter.Tk()
root.withdraw()

file_path = tkinter.filedialog.askopenfilename()

# Files could be loaded using SNAPPY import product, but for now assuming that the input is .tiff is ok

#filename1 = 'test.tif'
#dataset1 = gdal.Open(filename1)

#!gdalinfo filename1

# Load data