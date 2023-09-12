#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ISRM Data Object

@author: libbykoolik
last modified: 2023-09-12
"""

# Import Libraries
import pandas as pd
import geopandas as gpd
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import netcdf_file as nf
import os
from os import path
import sys
import concurrent.futures
from inspect import currentframe, getframeinfo
sys.path.append('./scripts')
from tool_utils import *

#%% Define the ISRM Object
class isrm:
    '''
    Defines a new object for storing and manipulating ISRM data.
    
    INPUTS:
        - isrm_path: a string representing the folder containing all ISRM data
        - output_region: a geodataframe of the region for results to be output, 
          as calculated by get_output_region in tool_utils.py
        - region_of_interest: the name of the region contained in the output_region
        - run_parallel: a Boolean indicating whether or not to run in parallel
        - load_file: a Boolean indicating whether or not the file should be 
          loaded (for debugging)
        - verbose: a Boolean indicating whether or not detailed logging statements 
          should be printed
        - debug_mode: a Boolean indicating whether or not to output debug statements
          
    CALCULATES:
        - receptor_IDs: the IDs associated with ISRM receptors within the output_region
        - receptor_geometry: the geospatial information associated with the ISRM 
          receptors within the output_region
        - PM25, NH3, NOx, SOX, VOC: the ISRM matrices for each of the primary 
          pollutants
        
    EXTERNAL FUNCTIONS:
        - get_pollutant_layer: returns the ISRM matrix for a single pollutant
        - map_isrm: simple function for mapping the ISRM grid cells
    
    '''
    def __init__(self, isrm_path, output_region, region_of_interest, run_parallel, debug_mode, load_file=True, verbose=False):
        ''' Initializes the ISRM object'''        
        
        # Initialize paths and check that they are valid
        sys.path.append(os.path.realpath('..'))
        self.isrm_path = isrm_path
        self.nh3_path, self.nox_path, self.pm25_path, self.sox_path, self.voc_path, self.geo_file_path = self.get_isrm_files()
        self.output_region = output_region
        self.region_of_interest = region_of_interest
        self.run_parallel = run_parallel
        self.valid_file, self.valid_geo_file = self.check_path()
        
        # Grab other meta-parameters
        self.debug_mode = debug_mode
        self.load_file = load_file
        self.verbose = verbose
        
        # Return a starting statement
        verboseprint(self.verbose, '- [ISRM] Loading a new ISRM object.',
                     self.debug_mode, frameinfo=getframeinfo(currentframe()))
        
        # If the files do not exist, quit before opening
        if not self.valid_file:
            logging.info('\n<< [ISRM] ERROR: The folder provided for the ISRM files is not correct or the correct files are not present. Please correct and retry. >>')
            sys.exit()
        elif not self.valid_geo_file:
            logging.info('\n<< [ISRM] ERROR: The folder provided for the ISRM files is not correct or the correct boundary file is not present. Please correct and retry. >>')
            sys.exit()
        else:
            verboseprint(self.verbose, '- [ISRM] Filepaths and files found. Proceeding to import ISRM data.',
                         self.debug_mode, frameinfo=getframeinfo(currentframe()))
        
        # Read ISRM data and geographic information
        if self.valid_file == True and self.load_file == True and self.valid_geo_file == True:
            verboseprint(self.verbose, '- [ISRM] Beginning to import ISRM geographic data. This step may take some time.',
                         self.debug_mode, frameinfo=getframeinfo(currentframe()))            
            
            # Import the geographic data for the ISRM
            self.geodata = self.load_geodata()
            verboseprint(self.verbose, '- [ISRM] ISRM geographic data imported.',
                         self.debug_mode, frameinfo=getframeinfo(currentframe()))

            # Import numeric ISRM layers - if running in parallel, this will occur 
            # while the geodata file is also loading. 
            self.PM25, self.NH3, self.NOX, self.SOX, self.VOC = self.load_isrm()
            verboseprint(self.verbose, '- [ISRM] ISRM data imported. Five pollutant variables created',
                         self.debug_mode, frameinfo=getframeinfo(currentframe()))
            
            # Pull a few relevant layers
            self.crs = self.geodata.crs
            self.ISRM_ID = self.geodata['ISRM_ID']
            self.geometry = self.geodata['geometry']
            self.receptor_IDs, self.receptor_geometry = self.clip_isrm()
            
    
    def __str__(self):
        return 'ISRM object'

    def __repr__(self):
        return '< ISRM object >'

    
    def get_isrm_files(self):
        ''' Defines ISRM file paths from the ISRM_Path input '''
        nh3_path = path.join(self.isrm_path,'ISRM_NH3.npy')
        nox_path = path.join(self.isrm_path,'ISRM_NOX.npy')
        pm25_path = path.join(self.isrm_path,'ISRM_PM25.npy')
        sox_path = path.join(self.isrm_path,'ISRM_SOX.npy')
        voc_path = path.join(self.isrm_path,'ISRM_VOC.npy')
        geo_file_path = path.join(self.isrm_path, 'isrm_geo.feather')
        
        return nh3_path, nox_path, pm25_path, sox_path, voc_path, geo_file_path

    def check_path(self):
        ''' Checks if file exists at the path specified '''
        # Use the os library to check the path and the file
        # First, check ISRM layers exist
        good_paths = 0
        good_files = 0
        
        for f in [self.nh3_path, self.nox_path, self.pm25_path, self.sox_path, self.voc_path]:            
            good_paths += path.exists(f)
            good_files += path.isfile(f)
        
        # Get Boolean path_exists and file_exists
        path_exists = good_paths == 5
        file_exists = good_files == 5
        
        # Second, check ISRM geodata exists
        geo_path_exists = path.exists(self.geo_file_path)
        geo_file_exists = path.isfile(self.geo_file_path)
            
        return (path_exists and file_exists, geo_path_exists and geo_file_exists)
    
    def load_and_cut(self, path):
        ''' Loads and cuts the ISRM numeric layer '''
        # Load in the file
        pollutant = np.load(path)
        
        if self.region_of_interest != 'CA':
            # Trim the columns of each ISRM layer to just the necessary IDs
            indices = self.receptor_IDs.values
            pollutant = pollutant[:,:,indices]
        
        return pollutant
    
    def load_isrm(self):
        ''' Loads ISRM from numpy files '''
        # Route to pollutant paths
        pollutant_paths = [self.pm25_path, self.nh3_path, self.nox_path,
                           self.sox_path, self.voc_path]
 
        # Create a storage list
        pollutants = []
        
        # Run clip_isrm to get the appendices
        self.receptor_IDs, self.receptor_geometry = self.clip_isrm()
        
        # Iterate through each path
        for path in pollutant_paths:
            pollutants.append(self.load_and_cut(path))
        
        return pollutants
    
    def load_geodata(self):
        ''' Loads feather into geopandas dataframe '''
        isrm_gdf = gpd.read_feather(self.geo_file_path)
        isrm_gdf.columns = ['ISRM_ID', 'geometry']
        
        return isrm_gdf
    
    def clip_isrm(self):
        ''' Clips the ISRM receptors to only the relevant ones '''
        if self.region_of_interest != 'CA':
            # Make a copy of the output_region geodataframe
            output_region = self.output_region.copy()
            output_region_prj = output_region.to_crs(self.geodata.crs)
            
            # Select rows of isrm_geodata that are within the output_region
            isrm_geodata = self.geodata.copy()
            isrm_region = gpd.sjoin(isrm_geodata, output_region_prj)
            receptor_IDs = isrm_region['ISRM_ID']
            receptor_geometry = isrm_region['geometry']
        
        else: # Return all indices
            receptor_IDs = self.geodata['ISRM_ID']
            receptor_geometry = self.geodata['geometry']
            
        self.receptor_geometry = receptor_geometry.copy()
        
        return receptor_IDs, receptor_geometry
    
    def get_pollutant_layer(self, pol_name):
        ''' Returns pollutant layer '''
        # Define a pollutant dictionary for convenience
        pollutant_dict = {'PM25':self.PM25,
                         'NH3':self.NH3,
                         'VOC':self.VOC,
                         'NOX':self.NOX,
                         'SOX':self.SOX}
        
        # Confirm pol_name is valid
        assert pol_name in pollutant_dict.keys()
        
        # Return pollutant layer
        return pollutant_dict[pol_name]
    
    def map_isrm(self):
        ''' Creates map of ISRM grid  '''
        # Note to build this out further at some point in the future, works for now
        fig, ax = plt.subplots(1,1)
        self.geodata.plot(ax = ax, edgecolor='black', facecolor='none')
        ax.set_title('ISRM Grid')
        fig.tight_layout()
        return fig