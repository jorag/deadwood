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


def showallbands(dataset_array):
    # Show all bands in GDAL image
    
    # Rearrage dimensions to x,y,channel format
    im_generate = np.transpose(dataset_array, (1,2,0))
    
    for i_band in range(im_generate.shape[2]):
        # Get statistics
        band_min = np.nanmin(im_generate[:,:,i_band])
        band_mean = np.nanmean(im_generate[:,:,i_band])
        band_max = np.nanmax(im_generate[:,:,i_band])
        plt.figure()
        plt.imshow(im_generate[:,:,i_band])
        plt.title('BAND '+ str(i_band) +' Min = '+str(band_min)+' Mean = '+str(band_mean)+' Max = '+str(band_max))
        plt.show()  # display it
        
                
    return
