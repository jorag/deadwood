import matplotlib.pyplot as plt
import numpy as np
import gdal
#from gdalconst import *
import os, sys, time
#from src import dispms
import math
import tkinter
from tkinter import filedialog

# GUI for file selection
root = tkinter.Tk()
root.withdraw()
file_path = tkinter.filedialog.askopenfilename()
#root.destroy() #is this needed?

# Files could be loaded using SNAPPY import product, but for now assuming that the input is .tiff is ok

#!gdalinfo file_path 


# Load data
dataset1 = gdal.Open(file_path)