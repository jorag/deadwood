#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""visandtest - Visualize and Test
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
    """Show all bands in GDAL image array"""
    
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


def modalitypoints3d(modality_type, dataset_array, labels_array, labels_dict=None):
    """Show all (transect) points in a 3D plot with colour and annotation
    
    Different annotations and channels used for different modalities.
    Sources:
        # https://matplotlib.org/gallery/mplot3d/scatter3d.html
        # https://stackoverflow.com/questions/1985856/how-to-make-a-3d-scatter-plot-in-python
    """
    
    if modality_type.lower() in ['sar_quad', 'quad', 'hhhvvv']:
        xlabel = 'HH'; xs = dataset_array[:,0]
        ylabel = 'HV'; ys = dataset_array[:,1]
        zlabel = 'VV'; zs = dataset_array[:,3]
    
    
    # Get standard colour/plotstyle vector
    colour_vec = mycolourvec()
    # Convert to numpy array for indexation
    colour_vec = np.asarray(colour_vec)
    
    # Plot
    fig = plt.figure()
    ax = Axes3D(fig)
    
    ax.scatter(xs, ys, zs, c=colour_vec[labels_array], marker='o')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    
    ax.set_title(modality_type)
    # TODO: Check if dict instead
    if labels_dict is not None:
        # Get legend text
        legend_text = []
        vals = list(labels_dict.values())
        # Go through them in sorted order
        for i_class in np.unique(vals):
            print(i_class)
            print(list(labels_dict.keys())[vals.index(i_class)])
            legend_text.append(list(labels_dict.keys())[vals.index(i_class)])
        plt.legend(legend_text)
        
    plt.show()
    
    return
