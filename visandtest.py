#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visandtest - Visualize and Test
Created on Wed Oct  3 12:58:32 2018

@author: jorag
"""

import numpy as np
import matplotlib.pyplot as plt
from mytools import *
from geopixpos import *

def showimpoint(all_data, geotransform, lat, lon, n_pixel_x=500, n_pixel_y=500, bands=[0,1,2]):
    # Use the pos2pix function from my geopixpos module to find pixel indice
    pix_lat, pix_long = pos2pix(geotransform, lat=lat, lon=lon, pixels_out = 'single', verbose=True)
        
    # Extract pixels from area
    im_generate = all_data[bands, int(pix_lat-n_pixel_x/2):int(pix_lat+n_pixel_x/2), int(pix_long-n_pixel_y/2):int(pix_long+n_pixel_y/2)]
    
    # Rearrage dimensions to x,y,RGB format
    im_generate = np.transpose(im_generate, (1,2,0))
    plt.figure()
    plt.imshow(im_generate) 
    plt.show()  # display it
    return

def showimage(all_data, bands=[0,1,2]):
    # Extract pixels from area
    im_generate = all_data[bands, :, :]
    
    # Rearrage dimensions to x,y,RGB format
    im_generate = np.transpose(im_generate, (1,2,0))
    plt.figure()
    plt.imshow(im_generate) 
    plt.show()  # display it
    return
