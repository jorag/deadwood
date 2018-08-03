import matplotlib.pyplot as plt
import numpy as np
import gdal
from gdalconst import *
import os, sys, time
from src import dispms
import math

filename1 = 'test.tif'
dataset1 = gdal.Open(filename1)

!gdalinfo filename1