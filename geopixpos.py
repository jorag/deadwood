#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jorag
"""

from mytools import *

def pos2pix(geotransform, lat='default', lon='default', pixels_out = 'single', verbose=False):
    """Find pixel position bt geotranform-info from GDAL.
    
    Input: geotranform, lat = , lon = , verbose = False 
    """
    # Get upper left corner and pixel width
    ul_lat = geotransform[3] # lat or y??
    ul_lon = geotransform[0] # long or x??
    pixel_width_lat = geotransform[5] # lat or y??
    pixel_width_lon = geotransform[1] # long or x??
    if verbose:
        print("Origin = ({}, {})".format(geotransform[0], geotransform[3]))
        print("Pixel Size = ({}, {})".format(geotransform[1], geotransform[5]))
                                  
    
    # Point of Interest
    point_lat = lat
    point_lon = lon
    if verbose:
        logit('\n lat =' + str(point_lat) + '\n lon =' + str(point_lon))
    
    # Estimate pixel position
    exact_pixpos_lat = (point_lat - ul_lat)/pixel_width_lat # Better way of ensuring correct sign? 
    exact_pixpos_lon = (ul_lon - point_lon)/pixel_width_lon # Better way of ensuring correct sign?
    
    # Return pixel position of closest the pixel or the four closest pixels  
    if pixels_out == 'single':
        pixpos_lat = int(round(exact_pixpos_lat))
        pixpos_lon = int(round(exact_pixpos_lon))
    elif pixels_out == 'quad':
        pixpos_lat = [int(np.floor(exact_pixpos_lat)), int(np.ceil(exact_pixpos_lat))]
        pixpos_lon = [int(np.floor(exact_pixpos_lon)), int(np.ceil(exact_pixpos_lon))]
        logit(str(pixpos_lat) + ' ' + str(pixpos_lon))
        
    # Return pixel position
    return pixpos_lat, pixpos_lon


def refinepos(a):
    """TODO: Implement this!
    
    Idea: Use the Geo-transform info to get a rough estimate of position, then look up "exact" position using LAT/LONG bands
    Look at sign of band minus coordinate of interest
    Use mean(abs(diff)) as sanity check, shuld be much smaller than 1 if correct (lat/long) band
    
    Needs GDAL (import). 
    Needs numpy as np
    Needs indice of lat/long bands.
    """
    lat_band = dataset.GetRasterBand(lat_band_i)
    print("Band Type={}".format(gdal.GetDataTypeName(lat_band.DataType)))
    
    # Offset in pixels from estimated position so that the estimated position is at the centre
    # Read this many extra pixels in each direction
    pix_offset_x = 2
    pix_offset_y = 2
    
    # Read geoposition band for fine search of position
    data = lat_band.ReadAsArray(pixpos_lon - pix_offset_x, pixpos_lat - pix_offset_y, pix_offset_x*2+1, pix_offset_y*2+1)
    #data = lat_band.ReadAsArray(int(exact_pixpos_lon), int(exact_pixpos_lat), 3, 3)
    print(data)
    
    diff_data = data-point_lat
    abs_diff = abs(diff_data)
    print(diff_data)
    a = np.where(abs_diff == np.min(abs_diff))
    b = np.argmin(abs_diff)
    
    # In the original example, there point of interests latitude is equal distance between 69.99997 lat and 70.00003 lat
    # In this case, there should be no refinement to the original estimate
    # Or should perhaps both be considered?
    
    print(data[a]) 
    # Does not work for b - this is because of how armin works in Python:
    # In case of multiple occurrences of the minimum values, the indices corresponding to the first occurrence are returned.
