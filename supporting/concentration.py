#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Total Concentration Data Object

@author: libbykoolik
last modified: 2024-06-11
"""

# Import Libraries
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import netcdf_file as nf
import logging
import os
from os import path
import sys
from inspect import currentframe, getframeinfo
sys.path.append('./supporting')
from isrm import isrm
from emissions import emissions
from concentration_layer import concentration_layer
sys.path.append('./scripts')
from tool_utils import *
import concurrent.futures
from matplotlib_scalebar.scalebar import ScaleBar

#%% Define the Concentration Object
class concentration:
    '''
    Defines a new object for storing and manipulating concentration data.
    
    INPUTS:
        - emis_obj: the emissions object, as defined by emissions.py
        - isrm_obj: the ISRM object, as defined by isrm.py
        - detailed_conc_flag: a Boolean indicating whether concentrations should be output
          at a detailed level or not
        - run_parallel: a Boolean indicating whether or not to run in parallel
        - output_dir: a string pointing to the output directory
        - output_emis_flag: a Boolean indicating whether ISRM-allocated emissions should be output
        - debug_mode: a Boolean indicating whether or not to output debug statements
        - shp_path: data variable file path for the boarder
        - output_region: a geodataframe containing only the region of interest
        - emis_change_only: a Boolean indicating whether this is only emissions change
        
    CALCULATES:
        - detailed_conc: geodataframe of the detailed concentrations at ground-level 
          combined from all three vertical layers
        - detailed_conc_clean: simplified geodataframe of the detailed concentrations 
          at ground-level combined from all three vertical layers
        - total_conc: geodataframe with total ground-level PM2.5 concentrations 
          across the ISRM grid
          
    EXTERNAL FUNCTIONS:
        - visualize_concentrations: draws a map of concentrations for a variable
          and exports it as a PNG into an output directory of choice
        - export_concentrations: exports concentrations as a shapefile into an output
          directory of choice

    '''
        
    def __init__(self, emis_obj, isrm_obj, detailed_conc_flag, run_parallel, output_dir, output_emis_flag, debug_mode, ca_shp_path, output_region, output_geometry_fps, emis_change_only, output_resolution='ISRM', run_calcs=True, verbose=False):

        ''' Initializes the Concentration object'''        
        
        # Initialize concentration object by reading in the emissions and isrm 
        self.emissions = emis_obj
        self.isrm = isrm_obj
        
        # Get a few other metadata
        self.detailed_conc_flag = detailed_conc_flag
        self.run_parallel = run_parallel
        self.isrm_id = self.isrm.ISRM_ID
        self.isrm_geom = self.isrm.geometry
        self.crs = self.isrm.crs
        self.name = self.emissions.emissions_name
        self.output_resolution = output_resolution
        self.debug_mode = debug_mode
        self.shp_path = ca_shp_path
        self.emis_change_only = emis_change_only
        self.output_region = output_region
        self.output_geometry_fps = output_geometry_fps
        self.verbose = verbose
        self.run_calcs = run_calcs
        self.output_dir = output_dir
        self.output_emis_flag = output_emis_flag
        
        #verboseprint = logging.info if self.verbose else lambda *a, **k:None # for logging
        verboseprint(self.verbose, '- [CONCENTRATION] Creating a new concentration object',
                     self.debug_mode, frameinfo=getframeinfo(currentframe()))
                
        # Run concentration calculations
        if self.run_calcs:
            self.detailed_conc, self.detailed_conc_clean, self.total_conc = self.combine_concentrations()
            self.summary_conc, self.crosswalk = self.get_summary_conc()
                
            verboseprint(self.verbose, '- [CONCENTRATION] Total concentrations are now ready.',
                         self.debug_mode, frameinfo=getframeinfo(currentframe()))
            logging.info('\n')
            
    def __str__(self):
        return 'Concentration object created from the emissions from '+self.name + ' and the ISRM grid.'

    def __repr__(self):
        return '< Emissions object created from '+self.name + ' and the ISRM grid.>'

    def run_layer(self, layer):
        ''' Estimates concentratiton for a single layer '''
        # Creates a concentration_layer object for the given layer
        conc_layer = concentration_layer(self.emissions, self.isrm, layer, self.output_dir, self.output_emis_flag, self.emis_change_only, self.run_parallel, self.shp_path, self.output_region, debug_mode = self.debug_mode, run_calcs=True, verbose=self.verbose)
        
        # Copies out just the detailed_conc object and adds the LAYER column
        detailed_conc_layer = conc_layer.detailed_conc.copy()
        detailed_conc_layer['LAYER'] = layer
        
        return detailed_conc_layer
 
    
    def combine_concentrations(self):
        ''' 
        Creates a concentration_layer object for each valid layer and then 
        combines them all into three sets of concentration data
        '''
        # Define a concentration layer list for easier appending
        conc_layers = []
        
        # Run each layer if the layer flag is True
        if self.emissions.L0_flag: conc_layers.append(self.run_layer(0))
        if self.emissions.L1_flag: conc_layers.append(self.run_layer(1))
        if self.emissions.L2_flag: conc_layers.append(self.run_layer(2))
        if self.emissions.isrm_hole_flag: conc_layers.append(self.run_layer('hole'))
        
        # Concatenate these detailed concentration dataframes
        detailed_concentration = pd.concat(conc_layers)
        
        ## Sum each concentration field across ISRM_ID
        # First, need to get rid of unnecessary columns
        detailed_concentration_clean = detailed_concentration[detailed_concentration.columns.drop(list(detailed_concentration.filter(regex='EMISSIONS')))]
        detailed_concentration_clean = detailed_concentration_clean.drop(columns='geometry').copy()
        
        # Add across ISRM IDs
        detailed_concentration_clean = detailed_concentration_clean.groupby(['ISRM_ID']).sum().reset_index()
        
        # Merge back in the geodata
        geodata = self.isrm.geodata.copy()
        detailed_concentration_clean = pd.merge(detailed_concentration_clean, geodata, 
                                                left_on='ISRM_ID', right_on='ISRM_ID')
        
        # Make a final version that is very simple
        total_concentration = detailed_concentration_clean[['ISRM_ID','geometry', 'TOTAL_CONC_UG/M3']].copy()
        
        return detailed_concentration, detailed_concentration_clean, total_concentration
    
    def visualize_concentrations(self, var, output_region, output_dir, f_out, ca_shp_fp, export=False):
        ''' Creates map of concentrations using simple chloropleth '''
        # Note to build this out further at some point in the future, works for now
        if self.verbose:
            logging.info('- Drawing map of total PM2.5 concentrations.')
        
        # Read in CA boundary
        ca_shp = gpd.read_feather(ca_shp_fp)
        ca_prj = ca_shp.to_crs(self.crs)
        
        # Reproject output_region
        output_region = output_region.to_crs(self.crs)
        
        # Create necessary labels and strings
        if var[0:10] == 'CONC_UG/M3':
            pol = 'Emissions of '+var.split('_')[-1]
        else:
            pol = 'All Emissions'
                
        # A few things vary on the output resolution
        if self.output_resolution in ['AB','AD','C']:
            st_str = '* Area-Weighted Average'
            fname = f_out + '_' + pol + '_area_wtd_concentrations.png'
            t_str = r'PM$_{2.5}$ Concentrations* '+'from {}'.format(pol)
            c_to_plot = self.summary_conc[['NAME', 'geometry', var]].copy()
           
        else:
            t_str = r'PM$_{2.5}$ Concentrations '+'from {}'.format(pol)
            fname = f_out + '_' + pol + '_concentrations.png'
            c_to_plot = self.detailed_conc_clean[['ISRM_ID', 'geometry', var]].copy()
            
        # Tie things together
        fname = str.lower(fname)
        fpath = os.path.join(output_dir, fname)
        
        # Clip to output region
        c_to_plot = gpd.clip(c_to_plot, output_region)
        
        sns.set_theme(context="notebook", style="whitegrid", font_scale=1.25)
        
        fig, ax = plt.subplots(1,1)
        c_to_plot.plot(column=var,
                              figsize=(20,10),
                              legend=True,
                              legend_kwds={'label':r'Concentration of PM$_{2.5}$ ($\mu$g/m$^3$)'},
                              cmap='mako_r',
                              edgecolor='none',
                              antialiased=False,
                              ax = ax)
        
        ca_prj.plot(edgecolor='black', facecolor='none', ax=ax)
        
        # Clip to output_region
        minx, miny, maxx, maxy = output_region.total_bounds
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

        # Calculates the longitude and latitude of the center
        center_lon, center_lat = (minx + maxx) / 2, (miny + maxy) / 2
        
        angle_to_north = calculate_true_north_angle(center_lon, center_lat, self.crs)
        add_north_arrow(ax,float(angle_to_north))
        
        # Add scale bar
        scalebar = ScaleBar(1, location='lower left', border_pad=0.5)  # 1 pixel = 1 unit
        ax.add_artist(scalebar)

        # Add formatting
        ax.set_title(t_str)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        
        # If output region is used, add a footnote
        if self.output_resolution in ['AB','AD','C']:
            ax.text(minx-(maxx-minx)*0.1, miny-(maxy-miny)*0.1, st_str, fontsize=12)
        
        fig.tight_layout()
        
        if export:
            verboseprint(self.verbose, '   - [CONCENTRATION] Exporting a map of total PM2.5 concentrations as a png.',
                         self.debug_mode, frameinfo=getframeinfo(currentframe()))
            fig.savefig(fpath, dpi=200)
            logging.info('- [CONCENTRATION] Map of concentrations output as {}'.format(fname))
        return 
    
    def export_concentrations(self, output_dir, f_out):
        ''' Exports concentration as a shapefile (detailed or total) '''
        verboseprint(self.verbose, '- [CONCENTRATION] Exporting concentrations as a shapefile.',
                     self.debug_mode, frameinfo=getframeinfo(currentframe()))
        # If detailed flag is True, export detailed shapefile
        if self.detailed_conc_flag:
            if self.emis_change_only: 
                fname = f_out + '_change_detailed_concentration.shp' # File Name
            else: 
                fname = f_out + '_detailed_concentration.shp' # File Name
            fpath = os.path.join(output_dir, fname)
            
            # Make a copy and change column names to meet shapefile requirements
            gdf_export = self.detailed_conc.copy()
            gdf_export.columns = ['ISRM_ID', 'geometry', 'PM25_UG_S', 'NH3_UG_S',
                                  'VOC_UG_S', 'NOX_UG_S', 'SOX_UG_S', 'fPM_UG_M3', 
                                  'fNH3_UG_M3', 'fVOC_UG_M3', 'fNOX_UG_M3',
                                  'fSOX_UG_M3', 'PM25_UG_M3', 'LAYER']
            
            # Export
            gdf_export.to_file(fpath)
            logging.info('   - [CONCENTRATION] Detailed concentrations output as {} >>'.format(fname))
            
        # If detailed flag is False, export only total concentration shapefile
        else:
            if self.emis_delta:
                fname = str.lower(f_out + '_change_total_concentration.shp') # File Name
            else:
                fname = str.lower(f_out + '_total_concentration.shp') # File Name
            fpath = os.path.join(output_dir, fname)
            
            # Make a copy and change column names to meet shapefile requirements
            gdf_export = self.summary_conc.copy()
            gdf_export.columns = ['NAME', 'geometry', 'PM25_UG_M3']
            
            # Export
            gdf_export.to_file(fpath)
            logging.info('   - [CONCENTRATION] Total concentrations output as {}'.format(fname))
        
        return 
    
    def get_summary_conc(self):
        ''' Creates the summary concentration object if the output resolution is coarser
            than the ISRM grid '''
        
        # This function will take two different approaches based on the output resolution
        if self.output_resolution in ['AB','AD','C']:
            
            # Load the output resolution data
            boundary = gpd.read_feather(self.output_geometry_fps[self.output_resolution]).to_crs(self.crs)
            
            # Make a copy of the ISRM data
            tmp = self.total_conc.copy()
            
            # Intersect these two dataframes
            intersect = gpd.overlay(tmp, boundary, keep_geom_type=False, how='intersection')

            # Add the area column for the intersected data
            intersect['area_km2'] = intersect.geometry.area/(1000.0*1000.0)    
            total_area = intersect.groupby('NAME').sum()['area_km2'].to_dict()
            
            # Add a total area and area fraction to the intersect object
            intersect['area_total'] = intersect['NAME'].map(total_area)
            intersect['area_frac'] = intersect['area_km2'] / intersect['area_total']

            # Update the concentration to scale by the fraction
            intersect['TOTAL_CONC_UG/M3'] = intersect['area_frac'] * intersect['TOTAL_CONC_UG/M3']  
                
            # Remove any null variables
            intersect['TOTAL_CONC_UG/M3'] = intersect['TOTAL_CONC_UG/M3'].fillna(0)
         
            # Sum up for each larger shape
            summary_conc = intersect.groupby(['NAME'])[['TOTAL_CONC_UG/M3']].sum().reset_index()
            
            ## Clean up
            summary_conc = summary_conc[['NAME','TOTAL_CONC_UG/M3']].copy()
                        
            # Clean up
            summary_conc = summary_conc.reset_index(drop=True)
            
            # Merge with boundary data
            summary_conc = pd.merge(boundary, summary_conc, on='NAME')
            
            # Also, save a crosswalk
            crosswalk = intersect[['NAME','ISRM_ID','area_frac', 'area_total', 'geometry']].copy()
            crosswalk = crosswalk[~crosswalk['NAME'].isna()].copy()
            crosswalk = pd.merge(crosswalk, tmp[['ISRM_ID','TOTAL_CONC_UG/M3']], on='ISRM_ID', how='left')
             
        # If not, create summary_conc from total_conc
        else:
            # Just copy the total concentration
            summary_conc = self.total_conc.copy()
            
            # Change the column names
            summary_conc.rename(columns={'ISRM_ID':'NAME'}, inplace=True)
            
            # Create the crosswalk
            crosswalk = summary_conc[['NAME']].copy()
            crosswalk['ISRM_ID'] = crosswalk['NAME']
            crosswalk['area_frac'], crosswalk['area_total'] = (1,1) # placeholder values
        
        return summary_conc, crosswalk
    
    def output_concentrations(self, output_region, output_dir, f_out, ca_shp_path, shape_out):
        ''' Function for outputting concentration data '''
    
        # Draw the map
        self.visualize_concentrations('TOTAL_CONC_UG/M3', output_region, output_dir, f_out, ca_shp_path, export=True)
        
        # Export the shapefiles
        self.export_concentrations(shape_out, f_out)
        
        return

    