#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EJ Functions

@author: libbykoolik
last modified: 2024-01-24
"""

# Import Libraries
import pandas as pd
import geopandas as gpd
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow
from scipy.io import netcdf_file as nf
import os
from os import path
import sys
from inspect import currentframe, getframeinfo
sys.path.append('./scripts')
from tool_utils import *
import concurrent.futures

#%%
def create_exposure_df(conc, isrm_pop_alloc, verbose, debug_mode):
    ''' 
    Create an exposure geodataframe from concentration and population.
    
    INPUTS:
        - conc: concentration object
        - isrm_pop_alloc: population object re-allocated to the ISRM grid cell 
          geometry
        - verbose: a Boolean indicating whether or not detailed logging statements 
          should be printed
        - debug_mode: a Boolean indicating whether or not to output debug statements
          
    OUTPUTS:
        - exposure_gdf: a geodataframe with the exposure concentrations and allocated 
          population by racial group
    
    '''
    # Pull the total concentration from the conc object
    conc_gdf = conc.total_conc.copy()
    conc_gdf.columns = ['ISRM_ID', 'geometry', 'PM25_UG_M3']
    
    # Pull only relevant columns from isrm_pop_alloc
    groups = ['TOTAL', 'ASIAN', 'BLACK', 'HISLA', 'INDIG', 'PACIS', 'WHITE','OTHER']
    isrm_pop_alloc = isrm_pop_alloc[['ISRM_ID']+groups].copy()
    
    # Merge concentration and population data based on ISRM_ID
    exposure_gdf = pd.merge(conc_gdf, isrm_pop_alloc, left_on='ISRM_ID', right_on='ISRM_ID')
    
    # Get PWM columns per group
    verboseprint(verbose, '- [EJ] Estimating population weighted mean exposure for each demographic group.', 
                 debug_mode, frameinfo=getframeinfo(currentframe()))
    for group in groups:
        exposure_gdf = add_pwm_col(exposure_gdf, group)
        
    return exposure_gdf

def add_pwm_col(exposure_gdf, group):
    ''' 
    Adds an intermediate column that multiplies population by exposure.
    
    INPUTS:
        - conc: concentration object from `concentration.py`
        - isrm_pop_alloc: population object (from `population.py`) re-allocated to the ISRM 
          grid cell geometry
        - verbose: a Boolean indicating whether or not detailed logging statements should be 
          printed
          
    OUTPUTS:
        - exposure_gdf: a geodataframe with the exposure concentrations and allocated population
          by racial group
    
    '''
    # Create a string for the PWM column name
    pwm_col = group+'_PWM'
    
    # Create a column for each ISRM cell that is the group total exposure
    exposure_gdf[pwm_col] = exposure_gdf[group]*exposure_gdf['PM25_UG_M3']
    
    return exposure_gdf

def get_pwm(exposure_gdf, group):
    ''' 
    Estimates the population weighted mean exposure for a given group 
    
    INPUTS:
        - exposure_gdf: a geodataframe with the exposure concentrations and allocated population 
          by racial group
        - group: the racial/ethnic group name
        
    OUTPUTS: 
        - PWM_group: the group-level population weighted mean exposure concentration (float)
    '''
    # Create a string for the PWM column name
    pwm_col = group+'_PWM'
    
    # Estimate the total group-level PWM
    PWM_group = exposure_gdf[pwm_col].sum()/exposure_gdf[group].sum()
    
    return PWM_group

def get_overall_disparity(exposure_gdf):
    ''' 
    Returns a table of overall disparity metrics 
    
    INPUTS:
        - exposure_gdf: a geodataframe with the exposure concentrations and allocated population 
          by racial group
        
    OUTPUTS: 
        - pwm_df: a dataframe containing the PWM, absolute disparity, and relative disparity
          of each group
    
    '''
    # Define racial/ethnic groups of interest
    groups = ['TOTAL', 'ASIAN', 'BLACK', 'HISLA', 'INDIG', 'PACIS', 'WHITE','OTHER']
    
    # Create a dataframe to store information
    pwm_df = pd.DataFrame({'Group':groups}, columns=['Group'])
    
    # Use predefined function to get the group PWMs
    pwm_df['Group PWM'] = pwm_df.apply(lambda x: get_pwm(exposure_gdf, x['Group']), axis=1)
    
    # Calculate Absolute and Relative Disparities 
    pwm_df['Absolute Disparity'] = pwm_df['Group PWM'] - pwm_df.loc[0,'Group PWM']
    pwm_df['Relative Disparity'] = pwm_df['Absolute Disparity']/pwm_df.loc[0,'Group PWM']
    
    return pwm_df

def estimate_exposure_percentile(exposure_gdf, verbose):
    ''' 
    Creates a dataframe of percentiles
    
    INPUTS:
        - exposure_gdf: a geodataframe with the exposure concentrations and allocated population 
          by racial group
        - verbose: a Boolean indicating whether or not detailed logging statements should be printed
        
    OUTPUTS:
        - df_pctl: a dataframe of exposure concentrations by percentile of population exposed 
          by group
    
    '''
    if verbose:
        logging.info('- Estimating the exposure level for each percentile of each demographic group population.')
    
    # Define racial/ethnic groups of interest
    groups = ['TOTAL', 'ASIAN', 'BLACK', 'HISLA', 'INDIG', 'PACIS', 'WHITE','OTHER']
    
    # Create a copy to avoid overwriting, then sort based on PM25 concentration
    df_pctl = exposure_gdf.copy()
    df_pctl.sort_values(by='PM25_UG_M3', inplace=True)
    df_pctl.reset_index(drop=True, inplace=True)
    
    # Iterate through each group to estimate the percentile of exposure
    for group in groups:
        # Create a slice of the percentile dataframe
        df_slice = df_pctl[['PM25_UG_M3',group]].copy()
        
        # Add the cumulative sum of the population
        df_slice.loc[:,'Cumulative_Sum_Pop'] = df_slice.loc[:, group].cumsum()
        
        # Estimate the total population in that group, then divide the cumulative sum
        # to get the percentile
        total_pop_group = df_slice[group].sum()
        df_slice.loc[:, 'Percentile_'+group] = df_slice['Cumulative_Sum_Pop']/total_pop_group
        
        # Add the Percentile column into the main percentile dataframe
        df_pctl.loc[:, group] = df_slice.loc[:, 'Percentile_'+group]
    
    return df_pctl

def run_exposure_calcs(conc, pop_alloc, verbose, debug_mode):
    ''' 
    Run the exposure EJ calculations from one script 
    
    INPUTS:
        - conc: concentration object from `concentration.py`
        - isrm_pop_alloc: population object (from `population.py`) re-allocated to the 
          ISRM grid cell geometry
        - verbose: a Boolean indicating whether or not detailed logging statements should
          be printed
        - debug_mode: a Boolean indicating whether or not to output debug statements
        
    OUTPUTS: 
        - exposure_gdf: a dataframe containing the exposure concentrations and population
          estimates for each group
        - exposure_pctl: a dataframe of exposure concentrations by percentile of population
          exposed by group
        - exposure_disparity: a dataframe containing the PWM, absolute disparity, and relative
          disparity of each group
    
    '''
    # Call each of the functions in series
    exposure_gdf = create_exposure_df(conc, pop_alloc, verbose, debug_mode)
    exposure_disparity = get_overall_disparity(exposure_gdf)
    exposure_pctl = estimate_exposure_percentile(exposure_gdf, verbose)
    
    return exposure_gdf, exposure_pctl, exposure_disparity 

def export_exposure_gdf(exposure_gdf, shape_out, f_out):
    ''' 
    Exports the exposure_gdf dataframe as a shapefile 
    
    INPUTS:
        - exposure_gdf: a dataframe containing the exposure concentrations and population
          estimates for each group
        - shape_out: a filepath string of the location of the shapefile output directory
        - f_out: the name of the file output category (will append additional information)
        
    OUTPUTS:
        - None (fname is surrogate for completion)
    
    '''
    
    # Return a log statement
    logging.info('- [EJ] Exporting exposure geodataframe as a shapefile.')
    
    # Create the file name and path
    fname = str.lower(f_out + '_exposure_concentrations.shp') # File Name
    fpath = os.path.join(shape_out, fname)
    
    # Update the columns slightly
    exposure_gdf = exposure_gdf[['ISRM_ID', 'PM25_UG_M3', 'TOTAL', 'ASIAN', 'BLACK',
                                 'HISLA', 'INDIG', 'PACIS', 'WHITE', 'OTHER',
                                 'geometry']].copy()

    # Export to file
    exposure_gdf.to_file(fpath)
    logging.info('   - [EJ] Exposure concentrations output as {}'.format(fname))

    return fname #placeholder for parallelization

def export_exposure_csv(exposure_gdf, output_dir, f_out):
    ''' 
    Exports the exposure_gdf dataframe as a CSV file 
    
    INPUTS:
        - exposure_gdf: a dataframe containing the exposure concentrations and population
          estimates for each group
        - output_dir: a filepath string of the location of the output directory
        - f_out: the name of the file output category (will append additional information)
        
    OUTPUTS:
        - None (fname is surrogate for completion)
    
    '''
    
    # Return a log statement
    logging.info('- [EJ] Exporting exposure geodataframe as a comma separated value text file.')
    
    # Create the file name and path
    fname = str.lower(f_out + '_exposure_concentrations.csv') # File Name
    fpath = os.path.join(output_dir, fname)
    
    # Update the columns slightly
    exposure_gdf = exposure_gdf[['ISRM_ID', 'PM25_UG_M3', 'TOTAL', 'ASIAN', 'BLACK',
                                 'HISLA', 'INDIG', 'PACIS', 'WHITE', 'OTHER']].copy()

    # Change column names
    rename_dict = create_rename_dict()
    rename_dict = {k: v + ' (# People)' for k, v in rename_dict.items()} # Add units to population
    exposure_gdf.rename(columns=rename_dict, inplace=True)
    exposure_gdf.rename(columns={'PM25_UG_M3':'PM2.5 Concentration (ug/m3)'}, inplace=True)

    # Export to file
    exposure_gdf.to_csv(fpath, index=False)
    logging.info('   - [EJ] Exposure concentrations output as {}'.format(fname))

    return fname #placeholder for parallelization


def export_exposure_disparity(exposure_disparity, output_dir, f_out):
    ''' 
    Exports the exposure_disparity dataframe as a CSV file 
    
    INPUTS:
        - exposure_disparity: a dataframe containing the PWM, absolute disparity, and relative
          disparity of each group
        - output_dir: a filepath string of the location of the output directory
        - f_out: the name of the file output category (will append additional information)
        
    OUTPUTS:
        - None (fname is surrogate for completion)
    
    '''
    
    # Return a log statement
    logging.info('- [EJ] Exporting population-weighted mean exposures for each racial/ethnic group.')
    
    # Create the file name and path
    fname = str.lower(f_out + '_exposure_disparity.csv') # File Name
    fpath = os.path.join(output_dir, fname)
    
    # Update the values slightly
    rename_dict = create_rename_dict()
    exposure_disparity['Group'] = exposure_disparity['Group'].map(rename_dict)
    exposure_disparity['Relative Disparity'] = exposure_disparity['Relative Disparity'] * 100.0
    
    # Fix the columns for clarity of units
    exposure_disparity.rename(columns={'Group PWM':'Group PWM (ug/m3)',
                                       'Absolute Disparity':'Absolute Disparity (ug/m3)',
                                       'Relative Disparity':'Relative Disparity (%)'},
                              inplace=True)

    # Export to file
    exposure_disparity.to_csv(fpath, index=False)
    logging.info('   - [EJ] Exposure concentrations output as {}'.format(fname))

    return fname

def plot_percentile_exposure(output_dir, f_out, exposure_pctl, verbose, debug_mode):
    ''' 
    Creates a percentile plot by group 
    
    INPUTS:
        - output_dir: a filepath string of the location of the output directory
        - f_out: the name of the file output category (will append additional information)
        - exposure_pctl: a dataframe of exposure concentrations by percentile of population
          exposed by group
        - verbose: a Boolean indicating whether or not detailed logging statements should
          be printed
        - debug_mode: a Boolean indicating whether or not to output debug statements
        
    OUTPUTS:
        - None (fname is surrogate for completion)
    
    '''
    verboseprint(verbose, '- [EJ] Drawing plot of exposure by percentile of each racial/ethnic group.', 
                 debug_mode, frameinfo=getframeinfo(currentframe()))
    # Define racial/ethnic groups of interest
    groups = ['TOTAL', 'ASIAN', 'BLACK', 'HISLA', 'INDIG', 'PACIS', 'WHITE','OTHER']
    
    # Melt the dataframe for easier use of seaborn
    pctl_melt = pd.melt(exposure_pctl, id_vars='PM25_UG_M3',
                        value_vars=groups,var_name='Racial/Ethnic Group', 
                        value_name='Percentile')
    
    # Adjust formatting for a prettier plot
    pctl_melt['Percentile'] = pctl_melt['Percentile']*100
    rename_dict = create_rename_dict()
    pctl_melt['Racial/Ethnic Group'] = pctl_melt['Racial/Ethnic Group'].map(rename_dict)
    sns.set_theme(context="notebook", style="whitegrid", font_scale=1.75)

    # Initialize the figure
    fig, ax = plt.subplots(figsize=(10,8))
    sns.lineplot(data=pctl_melt, x='Percentile', y='PM25_UG_M3', hue='Racial/Ethnic Group', ci=None, 
                 linewidth=3, palette='deep', ax=ax)
    ax.set(ylabel='PM$_{2.5}$ Exposure ($\mu$g/m$^3$)')
    ax.set_xticks(ticks=[5,25,50,75,95], 
                  labels=['5th','25th','50th','75th','95th'])
    
    # Save the file
    fname =f_out+'_PM25_Exposure_Percentiles.png' # File Name
    fpath = os.path.join(output_dir, fname)
    fig.savefig(fpath, dpi=200)
    logging.info('- [EJ] Exposure concentration by percentile figure output as {}'.format(fname))
    
    return fname

def export_exposure(exposure_gdf, exposure_disparity, exposure_pctl, shape_out, output_dir, f_out, verbose, run_parallel, debug_mode):
    ''' 
    Calls each of the exposure output functions in parallel
    
    INPUTS:
        - exposure_gdf: a dataframe containing the exposure concentrations and population 
          estimates for each group
        - exposure_disparity: a dataframe containing the population-weighted mean exposure 
          concentrations for each group
        - exposure_pctl: a dataframe of exposure concentrations by percentile of population 
          exposed by group
        - shape_out: a filepath string of the location of the shapefile output directory
        - output_dir: a filepath string of the location of the output directory
        - f_out: the name of the file output category (will append additional information)
        - verbose: a Boolean indicating whether or not detailed logging statements should be 
          printed 
        - run_parallel: a Boolean indicating whether or not to run in parallel
        - debug_mode: a Boolean indicating whether or not to output debug statements
        
    OUTPUTS:
        - None
    
    '''
    
    # Return a log statements
    logging.info('- [EJ] Exporting exposure outputs.')

    if run_parallel:
        # Call export functions in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=5) as ej_executor:
            
            # Submit each export function to the executor
            gdf_export_future = ej_executor.submit(export_exposure_gdf, exposure_gdf, shape_out, f_out)
            csv_export_future = ej_executor.submit(export_exposure_csv, exposure_gdf, output_dir, f_out)
            disp_export_future = ej_executor.submit(export_exposure_disparity, exposure_disparity, output_dir, f_out)
            plot_export_future = ej_executor.submit(plot_percentile_exposure, output_dir, f_out, exposure_pctl, verbose, 
                                                    debug_mode)
            
            # Wait for all to finish
            (tmp, tmp, tmp, tmp) = (gdf_export_future.result(), csv_export_future.result(),
                                    disp_export_future.result(), plot_export_future.result())
    else:
        # Call export functions linearly
        export_exposure_gdf(exposure_gdf, shape_out, f_out)
        export_exposure_csv(exposure_gdf, output_dir, f_out)
        export_exposure_disparity(exposure_disparity, output_dir, f_out)
        plot_percentile_exposure(output_dir, f_out, exposure_pctl, verbose,
                                 debug_mode)
    
    logging.info('- [EJ] All exposure outputs have been saved.')

    return

def region_pwm_helper(name, group, full_dataset):
    ''' 
    Estimates population-weighted mean for a subset of the full_dataset
    
    INPUTS:
        - name: the specific name of the region type (e.g., SF BAY AREA)
        - group: the racial/ethnic group of interest
        - full_dataset: a dataframe containing all of the concentraion and population
          intersection objects with regions assigned
        
    OUTPUTS:
        - pwm: the population-weighted mean concentration of PM2.5
    
    '''
    # Slice relevant parts of the dataframe
    tmp = full_dataset[full_dataset['NAME']==name][['TOTAL_CONC_UG/M3',group]].copy()

    # Estimate the PWM
    pwm = (tmp[group]*tmp['TOTAL_CONC_UG/M3']).sum()/(tmp[group].sum())

    return pwm

def export_pwm_map(pop_exp, conc, output_dir, output_region, f_out, ca_shp_path, shape_out):
    ''' 
    Creates the exports for the population-weighted products requested when the 
    user inputs an output resolution larger than the ISRM grid
    
    INPUTS:
        - pop_exp: a dataframe containing the population information without age-resolution
        - conc: a concentration object
        - output_dir: a filepath string of the location of the output directory
        - output_region: the geometry of the desired output region
        - f_out: the name of the file output category (will append additional information)
        - ca_shp_path: a filepath string of the location of the California boundary shapefile
        - shape_out: a filepath string of the location of the shapefile output directory
        
    OUTPUTS:
        - None
    
    '''
    # Log statement
    logging.info('- [EJ] Creating population-weighted mean summaries at the output resolution requested.')
    
    ## We will need to combine three geometries in order to do this properly
    # Collect the necessary objects
    crosswalk = conc.crosswalk[['NAME','ISRM_ID', 'TOTAL_CONC_UG/M3', 'geometry']].copy() #<-- output geometry to ISRM crosswalk
    pop_exp = pop_exp[['POP_ID','TOTAL','ASIAN','BLACK','HISLA',
                        'INDIG','PACIS','WHITE','OTHER','geometry']].copy() #<-- population data
    
    ## Intersect these two dataframes and apportion population onto the crosswalk
    ## by area
    # Project to the same coordinates
    pop_exp = pop_exp.to_crs(crosswalk.crs)
    
    # Create an intersection object and collect total area
    intersect = gpd.overlay(pop_exp, crosswalk, how='union', keep_geom_type='False')
    
    # Remove null matches
    intersect = intersect[(~intersect['POP_ID'].isna())&(~intersect['ISRM_ID'].isna())]
    
    # Estimate areas and collect total area
    intersect['AREA_M2'] = intersect.geometry.area/(1000.*1000.)
    pop_totalarea = intersect.groupby('POP_ID').sum(['AREA_M2'])[['AREA_M2']].to_dict()['AREA_M2']
    
    # Add the fractional areas as temporary columns
    intersect['AREA_POP_TOTAL'] = intersect['POP_ID'].map(pop_totalarea)
    intersect['AREA_FRAC'] = intersect['AREA_M2']/intersect['AREA_POP_TOTAL']
    
    # Apportion population from each group into the intersect
    # Concentration should be preserved
    total_pop = pop_exp['TOTAL'].sum()
    for group in ['TOTAL', 'ASIAN', 'BLACK', 'HISLA', 'INDIG', 'PACIS', 'WHITE', 'OTHER']:
        intersect[group] = intersect[group]*intersect['AREA_FRAC']
    assert np.isclose(total_pop, intersect['TOTAL'].sum()) # check
    
    ## Get the output resolution names and geometries
    # Dissolve the crosswalk dataframe
    output_res_geo = crosswalk[['NAME','geometry']].dissolve(by='NAME').reset_index()
    
    # Estimate the PWM per group
    for group in ['TOTAL', 'ASIAN', 'BLACK', 'HISLA', 'INDIG', 'PACIS', 'WHITE', 'OTHER']:
        output_res_geo[group+'_PWM'] = output_res_geo.apply(lambda x: region_pwm_helper(x['NAME'], group, intersect), axis=1)
        
    ## Plot this as a map
    logging.info('- [EJ] Exporting map of population-weighted mean summaries at the output resolution requested.')
    visualize_pwm_conc(output_res_geo, output_region, output_dir, f_out, ca_shp_path)
    
    ## Create a shapefile to output, too
    to_shp = output_res_geo[['NAME','TOTAL_PWM','geometry']].copy()
    to_shp.columns = ['NAME', 'PWM_UG_M3', 'geometry']
    to_shp.to_file(path.join(output_dir, 'shapes', f_out + '_pwm_concentration.shp'))

    return output_res_geo

def visualize_pwm_conc(output_res_geo, output_region, output_dir, f_out, ca_shp_path):
    ''' 
    Creates map of PWM concentrations using simple chloropleth 
    
    INPUTS:
        - output_res_geo: a dataframe containing the population-weighted mean
          concentrations for each output resolution
        - output_region: the geometry of the desired output region
        - output_dir: a filepath string of the location of the output directory
        - f_out: the name of the file output category (will append additional information)
        - ca_shp_path: a filepath string of the location of the California boundary shapefile
        
    OUTPUTS:
        - None
    
    '''
    # Read in CA boundary
    ca_shp = gpd.read_feather(ca_shp_path)
    ca_prj = ca_shp.to_crs(output_res_geo.crs)
    
    # Reproject output_region
    output_region = output_region.to_crs(output_res_geo.crs)
    
    # Create necessary labels and strings
    pol = 'All Emissions'
    st_str = '* Population-Weighted Average'
    fname = f_out + '_' + 'pop_wtd_concentrations.png'
    t_str = r'PM$_{2.5}$ Concentrations* '+'from {}'.format(pol)
        
    # Tie things together
    fname = str.lower(fname)
    fpath = os.path.join(output_dir, fname)
    
    # Grab relevant info
    c_to_plot = output_res_geo[['NAME', 'TOTAL_PWM', 'geometry']].copy()
    
    # Clip to output region
    c_to_plot = gpd.clip(c_to_plot, output_region)
    
    sns.set_theme(context="notebook", style="whitegrid", font_scale=1.25)
    
    fig, ax = plt.subplots(1,1)
    c_to_plot.plot(column='TOTAL_PWM',
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
    
    ax.set_title(t_str)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.text(minx-(maxx-minx)*0.1, miny-(maxy-miny)*0.1, st_str, fontsize=12)
    
    fig.tight_layout()
    fig.savefig(fpath, dpi=200)
        
    return 

def create_rename_dict():
    ''' 
    Makes a global rename code dictionary for easier updating
    
    INPUTS: None
     
    OUTPUTS: 
         - rename_dict: a dictionary that maps demographic group names to codes
         
    '''
     
    # Set rename dictionary one time
    rename_dict = {'TOTAL':'Total', 'ASIAN':'Asian','BLACK':'Black',
                   'HISLA':'Hispanic/Latino', 'INDIG':'Native American', 
                   'PACIS':'Pacific Islander', 'WHITE':'White', 'OTHER':'Other'}
    
    return rename_dict