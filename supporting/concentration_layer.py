#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Concentration Layer Data Object

@author: libbykoolik
last modified: 2023-09-12
"""

# Import Libraries
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import netcdf_file as nf
import os
from os import path
import logging
import sys
from inspect import currentframe, getframeinfo
sys.path.append('./supporting')
from isrm import isrm
from emissions import emissions
sys.path.append('./scripts')
from tool_utils import *
import concurrent.futures

#%% Define the Concentration Layer Object
class concentration_layer:
    '''
    Defines a new object for storing and manipulating concentration data for a single layer of the ISRM.
    
    INPUTS:
        - emis_obj: an emissions object
        - isrm_obj: an ISRM object
        - layer: the vertical layer of the ISRM grid to use
        - run_parallel: a Boolean indicating whether or not to run in parallel
        - run_calcs: whether calculations should be run or just checked
        - debug_mode: a Boolean indicating whether or not to output debug statements
        - verbose: whether the tool should return more logging statements
        
    CALCULATES:
        - PM25e, NH3e, VOCe, NOXe, SOXe: geodataframes of the emissions (for each pollutant) 
          from that layer re-allocated onto the ISRM grid
        - pPM25, pNH4, pVOC, pNO3, pSO4: geodataframes of the concentrations from each primary 
          pollutant from the emissions of that pollutant in that layer
        - detailed_conc: geodataframe containing columns for each primary pollutant's 
          contribution to the total ground-level PM2.5 concentrations
        
    '''
    def __init__(self, emis_obj, isrm_obj, layer, output_dir, run_parallel, debug_mode, run_calcs=True, verbose=False):
        ''' Initializes the Concentration object'''        
        # Initialize concentration object by reading in the emissions and isrm 
        self.emissions = emis_obj
        self.isrm = isrm_obj
        
        # Get a few other metadata
        self.layer = layer
        self.run_parallel = run_parallel
        self.isrm_id = self.isrm.ISRM_ID
        self.receptor_id = self.isrm.receptor_IDs
        self.isrm_geom = self.isrm.geometry
        self.crs = self.isrm.crs
        self.name = self.emissions.emissions_name
        self.debug_mode = debug_mode
        self.verbose = verbose
        self.output_dir = output_dir
        
        # Print a few things for logging purposes
        logging.info('- [CONCENTRATION] Estimating concentrations from layer {} of the ISRM.'.format(self.layer))
        #verboseprint = logging.info if self.verbose else lambda *a, **k:None # for logging
        verboseprint(self.verbose, '   - [CONCENTRATION] Creating a new concentration object for layer {}'.format(self.layer),
                     self.debug_mode, frameinfo=getframeinfo(currentframe()))
        
        # Run concentration calculations
        if run_calcs:
            # Allocate emissions to the ISRM grid
            verboseprint(self.verbose, '   - [CONCENTRATION] Reallocating emissions to the ISRM grid.',
                         self.debug_mode, frameinfo=getframeinfo(currentframe()))
            self.PM25e, self.NH3e, self.VOCe, self.NOXe, self.SOXe = self.process_emissions(self.emissions, self.isrm, self.verbose, self.output_dir)
            
            # Save the allocated emissions
            ...
            
            # Estimate concentrations
            verboseprint(self.verbose, '   - [CONCENTRATION] Calculating concentrations of PM25 from each pollutant.',
                         self.debug_mode, frameinfo=getframeinfo(currentframe()))
            self.pPM25 = self.get_concentration(self.PM25e, self.isrm.get_pollutant_layer('PM25'), self.layer)
            verboseprint(self.verbose, '      - [CONCENTRATION] Concentrations estimated from primary PM2.5.',
                         self.debug_mode, frameinfo=getframeinfo(currentframe()))
            self.pNH4 = self.get_concentration(self.NH3e, self.isrm.get_pollutant_layer('NH3'), self.layer)
            verboseprint(self.verbose, '      - [CONCENTRATION] Concentrations estimated from NH3.',
                         self.debug_mode, frameinfo=getframeinfo(currentframe()))
            self.pVOC = self.get_concentration(self.VOCe, self.isrm.get_pollutant_layer('VOC'), self.layer)
            verboseprint(self.verbose, '      - [CONCENTRATION] Concentrations estimated from VOCs.',
                         self.debug_mode, frameinfo=getframeinfo(currentframe()))
            self.pNO3 = self.get_concentration(self.NOXe, self.isrm.get_pollutant_layer('NOX'), self.layer)
            verboseprint(self.verbose, '      - [CONCENTRATION] Concentrations estimated from NOx.',
                         self.debug_mode, frameinfo=getframeinfo(currentframe()))
            self.pSO4 = self.get_concentration(self.SOXe, self.isrm.get_pollutant_layer('SOX'), self.layer)
            verboseprint(self.verbose, '      - [CONCENTRATION] Concentrations estimated from SOx.',
                         self.debug_mode, frameinfo=getframeinfo(currentframe()))
    
            # Add these together at each ISRM grid cell
            self.detailed_conc = self.combine_concentrations(self.pPM25,
                                                              self.pNH4,
                                                              self.pVOC,
                                                              self.pNO3,
                                                              self.pSO4)
            verboseprint(self.verbose, '   - [CONCENTRATION] Detailed concentrations are estimated from layer {}.'.format(self.layer),
                         self.debug_mode, frameinfo=getframeinfo(currentframe()))
            
    def __str__(self):
        return 'Concentration layer object created from the emissions from '+self.name + ' and the ISRM grid.'

    def __repr__(self):
        return '< Concentration layer object created from '+self.name + ' and the ISRM grid.>'

    @staticmethod
    def allocate_emissions(emis_layer, isrm_geography, pollutant, verbose, debug_mode):    
        ''' Reallocates the emissions into the ISRM geography using a spatial intersect '''
        
        ## Pre-Process Slightly for Easier Functioning Downstream
        # Deep copy the emissions layer and add an ID field
        verboseprint(verbose, '- [CONCENTRATION] Allocating {} emissions to grid for ISRM layer.'.format(pollutant),
                     debug_mode, frameinfo=getframeinfo(currentframe()))
        emis = emis_layer.copy(deep=True)
        emis['EMIS_ID'] = 'EMIS_'+emis.index.astype(str)
        
        # Re-project the emissions layer into the ISRM coordinate reference system
        emis = emis.to_crs(isrm_geography.crs)
        
        # Store the total emissions from the raw emissions data for later comparison
        old_total = emis['EMISSIONS_UG/S'].sum()
        
        ## Perform Intersect to Reallocate Emissions
        # Get total area of each emissions cell
        emis['area_km2'] = emis.geometry.area/(1000*1000)
        
        # Create intersect object between emis and ISRM grid
        intersect = gpd.overlay(emis, isrm_geography, how='intersection')
        emis_totalarea = intersect.groupby('EMIS_ID').sum()['area_km2'].to_dict()
        
        # Add a total area and area fraction to the intersect object
        intersect['area_total'] = intersect['EMIS_ID'].map(emis_totalarea)
        intersect['area_frac'] = intersect['area_km2'] / intersect['area_total']
        
        # Update the EMISSIONS_UG/S field to scale emissions by the area fraction
        intersect['EMISSIONS_UG/S'] = intersect['area_frac'] * intersect['EMISSIONS_UG/S']  
            
        # Sum over ISRM grid cell
        reallocated_emis = intersect.groupby('ISRM_ID')[['EMISSIONS_UG/S']].sum().reset_index()
        
        ## Preserve all ISRM grid cells for consistent shapes
        reallocated_emis = isrm_geography[['ISRM_ID','geometry']].merge(reallocated_emis,
                                                          how='left',
                                                          left_on='ISRM_ID',
                                                          right_on='ISRM_ID')
        reallocated_emis['EMISSIONS_UG/S'].fillna(0, inplace=True)
        
        ## Confirm that the total has not changed
        assert np.isclose(reallocated_emis['EMISSIONS_UG/S'].sum(), old_total)
        
        return reallocated_emis
    
    def cut_emissions(self, pol_obj, height_min, height_max):
        ''' Cuts an emissions pollutant object based on the height column '''
        tmp = pol_obj.copy()
        tmp_cut = tmp[(tmp['HEIGHT_M']>=height_min) & (tmp['HEIGHT_M']<height_max)]
        
        return tmp_cut
    
    def process_emissions(self, emis, isrm_obj, verbose, output_dir):
        ''' Processes emissions before calculating concentrations '''
        # Define pollutant names
        pollutants = ['PM25', 'NH3', 'VOC', 'NOX', 'SOX']
        
        # Define height_min and height_max for each layer
        height_bounds_dict = {0:(0.0, 57.0),
                              1:(57.0, 140.0),
                              2:(760.0, 99999.0),
                              'linear':(140.0, 760.0)}
        height_min = height_bounds_dict[self.layer][0]
        height_max = height_bounds_dict[self.layer][1]
        
        # Set up a dictionary for more intuitive storage
        tmp_dct = {}
        
        # Estimate results for each pollutant
        if self.run_parallel: # In parallel
            with concurrent.futures.ProcessPoolExecutor(max_workers=5) as cl_executor:
                futures = {}
                for pollutant in pollutants:
                    # Grab the pollutant layer (e.g., PM25)
                    emis_slice = emis.get_pollutant_layer(pollutant)
    
                    # Cut the pollutant layer based on the height
                    emis_slice = emis_slice[(emis_slice['HEIGHT_M']>=height_min) & (emis_slice['HEIGHT_M']<height_max)]
    
                    # verboseprint(self.verbose, f'- Estimating concentrations of PM2.5 from {pollutant}')
                    futures[pollutant] = cl_executor.submit(self.allocate_emissions, emis_slice, isrm_obj.geodata, pollutant, verbose, self.debug_mode)
                    
                verboseprint(verbose, '- [CONCENTRATION] Waiting for all allocations to complete',
                             self.debug_mode, frameinfo=getframeinfo(currentframe()))
                concurrent.futures.wait(futures.values()) # Waits for all calculations to finish
                verboseprint(verbose, '- [CONCENTRATION] All allocations complete.',
                             self.debug_mode, frameinfo=getframeinfo(currentframe()))
                
                # Creates a dict of the values
                tmp_dct = {x: futures[x].result() for x in pollutants}
                
        else: # If linear, loop through pollutants
        
            for pollutant in pollutants:
                # Grab the pollutant layer (e.g., PM25)
                emis_slice = emis.get_pollutant_layer(pollutant)
                
                # Cut the pollutant layer based on the height
                emis_slice = emis_slice[(emis_slice['HEIGHT_M']>=height_min) & (emis_slice['HEIGHT_M']<height_max)]
                
                tmp_dct[pollutant] = self.allocate_emissions(emis_slice, isrm_obj.geodata, 
                                                             pollutant, verbose, self.debug_mode)
                
        # TEMP
        self.save_allocated_emis(tmp_dct, output_dir)
            
        return tmp_dct['PM25'], tmp_dct['NH3'], tmp_dct['VOC'], tmp_dct['NOX'], tmp_dct['SOX']
    
    def save_allocated_emis(self, tmp_dct, output_dir):
        ''' Temporary function for outputting allocated emissions '''
        
        # Set up a dataframe based on the PM25 one
        aloc_emis = tmp_dct['PM25'].copy()
        
        # Grab just geometry
        geodata = aloc_emis[['ISRM_ID', 'geometry']].copy()
        
        # Remove geometry from aloc_emis
        aloc_emis = aloc_emis.drop('geometry', axis=1)
        
        # Rename column
        aloc_emis.rename(columns={'EMISSIONS_UG/S':'PM25_UG/S'}, inplace=True)
        
        # Loop through other pollutants
        for pol in ['NH3', 'VOC', 'NOX', 'SOX']:
            # Copy the geodataframe
            tmp = tmp_dct[pol].copy()
            
            # Fix column names
            tmp.drop('geometry', axis=1, inplace=True)
            tmp.rename(columns={'EMISSIONS_UG/S':'{}_UG/S'.format(pol)}, inplace=True)
            
            # Merge with aloc_emis
            aloc_emis = pd.merge(aloc_emis, tmp, on='ISRM_ID')
            
        # Add the geodata back in
        aloc_emis = pd.merge(aloc_emis, geodata, on='ISRM_ID')
            
        # Create a file name
        fname_tmp = 'allocated_emis'
        
        # Output
        aloc_emis.to_file(path.join(output_dir, 'shapes', fname_tmp))
            
        return
    
    def get_concentration(self, pol_emis, pol_isrm, layer):
        ''' For a given pollutant layer, get the resulting PM25 concentration '''
        # Slice off just the appropriate layer of the ISRM
        pol_isrm_slice = pol_isrm[layer, :, :]
        
        # Concentration is the dot product of emissions and ISRM
        conc = np.dot(pol_emis['EMISSIONS_UG/S'], pol_isrm_slice)
        
        # Convert into a geodataframe
        conc_df = pd.DataFrame(conc, columns=['CONC_UG/M3'], index=self.receptor_id)#pol_emis.index)
        conc_gdf = pol_emis.merge(conc_df, left_index=True, right_index=True)
        
        return conc_gdf
    
    def combine_concentrations(self, pPM25, pNH4, pVOC, pNO3, pSO4):
        ''' Combines concentration from each pollutant into one geodataframe '''
        # Merge to combine into one dataframe
        pol_gdf = pd.merge(pPM25, pNH4, left_on=['ISRM_ID','geometry'], 
                           right_on=['ISRM_ID','geometry'],
                           suffixes=('_PM25','_NH3'))
        
        pol_gdf = pol_gdf.merge(pVOC, left_on=['ISRM_ID','geometry'], 
                           right_on=['ISRM_ID','geometry'],
                           suffixes=('','_VOC'))
                
        pol_gdf = pol_gdf.merge(pNO3, left_on=['ISRM_ID','geometry'], 
                           right_on=['ISRM_ID','geometry'],
                           suffixes=('','_NOX'))
        
                
        pol_gdf = pol_gdf.merge(pSO4, left_on=['ISRM_ID','geometry'], 
                           right_on=['ISRM_ID','geometry'],
                           suffixes=('','_SOX'))
        
        # Quick ugly fix to add the pollutant back onto VOC (otherwise it is dropped)
        pol_gdf.rename(columns={'EMISSIONS_UG/S':'EMISSIONS_UG/S_VOC',
                                'CONC_UG/M3':'CONC_UG/M3_VOC'}, inplace=True)
        
        # Reorder columns for prettiness
        pol_gdf = pol_gdf[['ISRM_ID', 'geometry', 'EMISSIONS_UG/S_PM25',
                           'EMISSIONS_UG/S_NH3', 'EMISSIONS_UG/S_VOC',  
                           'EMISSIONS_UG/S_NOX', 'EMISSIONS_UG/S_SOX', 
                           'CONC_UG/M3_PM25','CONC_UG/M3_NH3', 'CONC_UG/M3_VOC',
                           'CONC_UG/M3_NOX', 'CONC_UG/M3_SOX']]
    
        pol_gdf['TOTAL_CONC_UG/M3'] = pol_gdf['CONC_UG/M3_PM25'] \
                                        + pol_gdf['CONC_UG/M3_NH3'] \
                                        + pol_gdf['CONC_UG/M3_VOC'] \
                                        + pol_gdf['CONC_UG/M3_NOX'] \
                                        + pol_gdf['CONC_UG/M3_SOX']
                                    
        return pol_gdf