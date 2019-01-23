#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visandtest - Visualize and Test
Created on Wed Oct  3 12:58:32 2018

@author: jorag
"""

import numpy as np
import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D # noqa: F401 unused import
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
    # Show all bands in GDAL image array
    
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


def showpoints3d(dataset_array, labels_array):
    # Show all (transect) points in a 3D plot with colour and annotation
    # Sources:
        # https://matplotlib.org/gallery/mplot3d/scatter3d.html
        # https://stackoverflow.com/questions/1985856/how-to-make-a-3d-scatter-plot-in-python
    
    fig = plt.figure()
    ax = Axes3D(fig)
    
    # TODO: Use mytools.py to define a standard colour/plotstyle vector
    colour_vec = mycolourvec()
    
    ax.scatter(dataset_array[:,0], dataset_array[:,1], dataset_array[:,3], c='r', marker='o')
#    for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
#        xs = randrange(n, 23, 32)
#        ys = randrange(n, 0, 100)
#        zs = randrange(n, zlow, zhigh)
#        ax.scatter(xs, ys, zs, c=c, marker=m)
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    plt.show()
    
    return
