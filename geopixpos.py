#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jorag
"""

from mytools import *


def pos2pix(geotransform, lat='default', lon='default', pixels_out = 'single', verbose=False):
    """Find pixel position by geotranform-info from GDAL.
    
    Input: geotranform, lat = , lon = , verbose = False 
    """
    func_log_id = "In pos2pix: "
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
    elif pixels_out == 'npsingle':
        pixpos_lat = np.round(exact_pixpos_lat)
        pixpos_lat = pixpos_lat.astype(int)
        pixpos_lon = np.round(exact_pixpos_lon)
        pixpos_lon = pixpos_lon.astype(int)
    elif pixels_out == 'quad':
        # LAT: Check that the pixel indice are different
        lat_floor = int(np.floor(exact_pixpos_lat))
        lat_ceil = int(np.ceil(exact_pixpos_lat))
        if lat_floor != lat_ceil:
            pixpos_lat = [lat_floor, lat_ceil]
        else:
            pixpos_lat = lat_floor
            logit(func_log_id + 'pixels_out = ' + str(pixels_out) + ' found exact match (1 pixel) for lat = ' + str(lat))
            
        # LONG: Check that the pixel indice are different
        lon_floor = int(np.floor(exact_pixpos_lon))
        lon_ceil = int(np.ceil(exact_pixpos_lon))
        if lon_floor != lon_ceil:
            pixpos_lon = [lon_floor, lon_ceil]
        else:
            pixpos_lon = lon_floor
            logit(func_log_id + 'pixels_out = ' + str(pixels_out) + ' found exact match (1 pixel) for long = ' + str(lon))
        
    # Return pixel position
    return pixpos_lat, pixpos_lon


def geobands2pix(lat_band, lon_band, lat='default', lon='default', pixels_out = 'single', verbose=False):
    """Find pixel position by geotranform-info from GDAL.
    
    Input: geotranform, lat = , lon = , verbose = False 
    """
    
    # Subtract position from band
    lat_diff = np.abs(lat_band - lat)
    lon_diff = np.abs(lon_band - lon)
    
    # Find minimum (zero crossing) in each direction for each band
    lat_indice = np.where(lat_diff == np.min(lat_diff))
    lon_indice = np.where(lon_diff == np.min(lon_diff))
    
    print(lat_indice)
    print(lon_indice)
    print(np.min(lat_diff), np.min(lon_diff))
    print(length(lat_indice[0]), length(lat_indice[1]), length(lon_indice[0]), length(lon_indice[1]))
    
    # Find where indice overlap
    # Must be a match in BOTH row and column indice at the same time
    row_candidates = 1
    
    # Return pixel position
    return
    #return pixpos_lat, pixpos_lon


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
    
    
def gdalinfo_log(dataset, log_type='default'):
    """Log dataset info from GDAL.
    
    Input: dataset, log_type='default'
    """
    # Print information - can also use command line: !gdalinfo file_path 
    logit("Driver: {}/{}".format(dataset.GetDriver().ShortName,
                                 dataset.GetDriver().LongName), log_type)
    logit("Size is {} x {} x {}".format(dataset.RasterXSize,
                                        dataset.RasterYSize,
                                        dataset.RasterCount), log_type)
    logit("Projection is {}".format(dataset.GetProjection()), log_type)
    
    
def bandinfo_log(band, log_type='default'):
    """Log band info from GDAL.
    
    Input: band, log_type='default'
    """
    # To avoid importing GDAL in this module...
    logit("Band Type={}".format(band.DataType), log_type)
    #logit("Band Type={}".format(gdal.GetDataTypeName(band.DataType)), log_type)

    # Get pixel info
    pixel_min = band.GetMinimum()
    pixel_max = band.GetMaximum()
    if not pixel_min or not pixel_max:
        (pixel_min,pixel_max) = band.ComputeRasterMinMax(True)
    logit("Min={:.3f}, Max={:.3f}".format(pixel_min, pixel_max), log_type)
          
    if band.GetOverviewCount() > 0:
        logit("Band, number of overviews:", log_type)
        logit(band.GetOverviewCount(), log_type)
          
    if band.GetRasterColorTable():
        logit("Band, number of colour table with entries:", log_type)
        logit(band.GetRasterColorTable().GetCount(), log_type)
