#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ISRM Data Object

@author: libbykoolik
last modified: 2026-04-10
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
        - dpm: a Boolean indicating whether or not the DPM ISRM should be loaded
        - nox_conc: a Boolean indicating whether or not the NOx concentration ISRM should be loaded
          
    CALCULATES:
        - receptor_IDs: the IDs associated with ISRM receptors within the output_region
        - receptor_geometry: the geospatial information associated with the ISRM 
          receptors within the output_region
        - PM25, NH3, NOx, SOX, VOC: the ISRM matrices for each of the primary 
          pollutants
        - DPM, NOX_CONC: the ISRM matrices for each of the optional pollutants
        
    EXTERNAL FUNCTIONS:
        - get_pollutant_layer: returns the ISRM matrix for a single pollutant
        - map_isrm: simple function for mapping the ISRM grid cells
    
    '''
    def __init__(self, isrm_path, output_region, region_of_interest, run_parallel, debug_mode, dpm, nox_conc = False, LA_flag=True, LB_flag=True, LC_flag=True, load_file=True, verbose=False):
        ''' Initializes the ISRM object'''        
        
        # Initialize paths and check that they are valid
        sys.path.append(os.path.realpath('..'))
        self.isrm_path = isrm_path
        self.dpm = dpm
        self.nox_conc = nox_conc
        self.pollutant_names = ['PM25', 'NH3', 'VOC', 'NOX', 'SOX']
        print('**** '+','.join(self.pollutant_names))
        if self.dpm:
            self.pollutant_names.append('DPM')
        if self.nox_conc:
            self.pollutant_names.append('NOX_CONC')

        # 2. Get the flat list of paths
        all_paths = self.get_isrm_files()

        # 3. Dynamically assign attributes (self.pm25_la_path, etc.)
        # We use all_paths[:-1] to skip the geo_file at the end of the list
        path_idx = 0
        for pol in self.pollutant_names:
            for layer in ['LA', 'LB', 'LC']:
                attr_name = f"{pol.lower()}_{layer}_path"
                setattr(self, attr_name, all_paths[path_idx])
                path_idx += 1
            
        # 4. Assign the geo path (the last item)
        self.geo_file_path = all_paths[-1]

        
        # Store the region definitions and control flags for later use
        self.output_region = output_region # GeoDataFrame defining where concentrations run
        self.region_of_interest = region_of_interest # Name (e.g., 'SAN FRANCISCO BAY') used for logging/outputs
        self.run_parallel = run_parallel  #Boolean: whether to parallelize inner loops
        self.debug_mode = debug_mode # Boolean: print extra debug statements
        self.load_file = load_file # Boolean: if False, skip loading .npy’s (just check existence)
        self.verbose = verbose # Boolean: if True, log more progress messages

        # Flags for which vertical layers to load
        #   LA_flag: include layer A (ground level) in any concentration calculations
        #   LB_flag: include layer B (mid level)
        #   LC_flag: include layer C (upper level)
        self.LA_flag = LA_flag
        self.LB_flag = LB_flag
        self.LC_flag = LC_flag

        # Validate that all file paths exist and can be opened
        self.valid_file, self.valid_geo_file = self.check_path()


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
            self.pollutants = self.load_isrm()
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
        ''' Defines ISRM file paths based on active pollutants '''
        paths = []
    
        # Iterate through each active pollutant and each required layer
        for pol in self.pollutant_names:
            for layer in ['LA', 'LB', 'LC']:
                file_name = f'ISRM_{pol}_{layer}.npy'
                paths.append(os.path.join(self.isrm_path, file_name))
            
        # Always append the geo file at the end
        paths.append(os.path.join(self.isrm_path, 'isrm_geo.feather'))
        print('******* get_isrm_files: '+','.join(paths))
        return tuple(paths)

    def check_path(self):
        ''' Checks if ISRM layer files and geo file exist at the paths specified '''
        # Use the os library to check the path and the file
        # First, check ISRM layers exist
        good_paths = 0
        good_files = 0

        # List out all pollutant–layer paths. The base amount of paths is 15. 3 paths are added for each additional pollutant

        pollutant_files = [
            self.pm25_LA_path, self.pm25_LB_path, self.pm25_LC_path,
            self.nh3_LA_path,  self.nh3_LB_path,  self.nh3_LC_path,
            self.voc_LA_path,  self.voc_LB_path,  self.voc_LC_path,
            self.nox_LA_path,  self.nox_LB_path,  self.nox_LC_path,
            self.sox_LA_path,  self.sox_LB_path,  self.sox_LC_path,
        ]

        if self.dpm:
            pollutant_files.append(self.dpm_LA_path)
            pollutant_files.append(self.dpm_LB_path)
            pollutant_files.append(self.dpm_LC_path)

        if self.nox_conc:
            pollutant_files.append(self.nox_conc_LA_path)
            pollutant_files.append(self.nox_conc_LB_path)
            pollutant_files.append(self.nox_conc_LC_path)


        # Count how many of those exist and are files
        for f in pollutant_files:
            good_paths += path.exists(f)
            good_files += path.isfile(f)

        # We expect exactly 15, 18, or 21 layer files
        if self.dpm and self.nox_conc:
            path_exists = (good_paths == 21)
            file_exists = (good_files == 21)
        elif self.dpm or self.nox_conc:
            path_exists = (good_paths == 18)
            file_exists = (good_files == 18)
        else:
            path_exists = (good_paths == 15)
            file_exists = (good_files == 15)

        # Second, check ISRM geodata exists
        geo_path_exists = path.exists(self.geo_file_path)
        geo_file_exists = path.isfile(self.geo_file_path)

        # Return a tuple: (all pollutant layers OK?, geo-file OK?)
        return (path_exists and file_exists,
                geo_path_exists and geo_file_exists)

    
    def load_and_cut(self, path):
        ''' Loads and cuts the ISRM numeric layer '''
        # Load in the file
        pollutant = np.load(path)

        if self.region_of_interest != 'CA':
            # Trim the columns of each ISRM layer to just the necessary IDs
            indices = self.receptor_IDs.values
            pollutant = pollutant[:,indices]
        
        return pollutant

    def load_isrm(self):
        ''' Loads ISRM from numpy files, but only for flagged layers '''
        # now each pollutant has 3 layer‐paths
        pollutant_paths = [
            [self.pm25_LA_path, self.pm25_LB_path, self.pm25_LC_path],
            [self.nh3_LA_path,  self.nh3_LB_path,  self.nh3_LC_path],
            [self.voc_LA_path,  self.voc_LB_path,  self.voc_LC_path],
            [self.nox_LA_path,  self.nox_LB_path,  self.nox_LC_path],
            [self.sox_LA_path,  self.sox_LB_path,  self.sox_LC_path],
            ]
        print('*** paths being loaded as PM25, NH3, NOx, SOx, VOC')
        
        if self.dpm:
            pollutant_paths.append([self.dpm_LA_path, self.dpm_LB_path, self.dpm_LC_path])
        if self.nox_conc:
            pollutant_paths.append([self.nox_conc_LA_path, self.nox_conc_LB_path, self.nox_conc_LC_path])

        # Create a storage list
        pollutants = []
            
        # Run clip_isrm to get the appendices
        self.receptor_IDs, self.receptor_geometry = self.clip_isrm()

        # for each pollutant, for each of the 3 layers, load-or-zero
        for paths in pollutant_paths:
            for layer_path, flag in zip(paths, (self.LA_flag, self.LB_flag, self.LC_flag)):
                if flag:
                    arr = self.load_and_cut(layer_path)
                else:
                    arr = np.array([], dtype=float)
                pollutants.append(arr)
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
    
    def get_pollutant_layer(self):
        """
        Return a nested dict of ISRM layers:
        """

        # the order you built your flat pollutants list in load_isrm()
        pollutant_names =  self.pollutant_names
        layer_names     = ['LA',   'LB',   'LC']

        # assume you saved your arrays in self.pollutants
        flat = self.pollutants
        if self.dpm and self.nox_conc:
            assert len(flat) == 7 * 3, "* [ISRM] Expected 21 arrays in self.pollutants"
        elif self.dpm or self.nox_conc:
            assert len(flat) == 6 * 3, "* [ISRM] Expected 18 arrays in self.pollutants"
        else:  
            assert len(flat) == 5 * 3, "* [ISRM] Expected 15 arrays in self.pollutants"

        # build the empty outer dict
        layers = { lvl: {} for lvl in layer_names }

        # p_idx goes 0…4, l_idx goes 0…2
        for p_idx, pol in enumerate(pollutant_names):
            for l_idx, lvl in enumerate(layer_names):
                idx = p_idx * 3 + l_idx
                layers[lvl][pol] = flat[idx]
        return layers

    
    def map_isrm(self):
        ''' Creates map of ISRM grid  '''
        # Note to build this out further at some point in the future, works for now
        fig, ax = plt.subplots(1,1)
        self.geodata.plot(ax = ax, edgecolor='black', facecolor='none')
        ax.set_title('ISRM Grid')
        fig.tight_layout()
        return fig
