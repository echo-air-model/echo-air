#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Health Impact Functions

@author: libbykoolik
last modified: 2023-09-12
"""

# Import Libraries
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow
from scipy.io import netcdf_file as nf
import logging
import os
from os import path
import sys
from inspect import currentframe, getframeinfo
sys.path.append('./scripts')
from tool_utils import *
sys.path.append('./supporting')
from health_data import health_data
from matplotlib_scalebar.scalebar import ScaleBar

#%% Health Calculation Helper Functions
def create_hia_inputs(pop, load_file: bool, verbose: bool, geodata:pd.DataFrame,
                      incidence_fp: str, debug_mode:bool):
    """ Creates the hia_inputs object.
    
        Moving this into a separate function allows us to run this in parallel while
        other functions are running, speeding up the overall execution of the
        application.
    
    INPUTS:
        - pop: the population object input
        - load_file: a boolean telling program to load or not
        - verbose: a boolean telling program to return additional log statements or not
        - geodata: the geographic data from the ISRM 
        - incidence_fp: the filepath where the incidence data is stored
        
    OUTPUTS: 
        - a health data object ready for health calculations
    
    """
    hia_pop_alloc = pop.allocate_population(pop.pop_all, geodata, 'ISRM_ID', True)
    return health_data(hia_pop_alloc, incidence_fp, verbose=verbose, race_stratified=False, debug_mode=debug_mode)

def krewski(conc, inc, pop, endpoint):
    ''' 
    Estimates excess mortality from all causes using the Krewski (2009) function 
    
    INPUTS:
        - verbose: a Boolean indicating whether or not detailed logging statements should 
          be printed
        - conc: a float with the exposure concentration for a given geography
        - inc: a float with the background incidence for a given group in a given geography
        - pop: a float with the population estimate for a given group in a given geography
        - endpoint: a string containing either 'ALL CAUSE', 'ISCHEMIC HEART DISEASE', 
          or 'LUNG CANCER'
        
    OUTPUTS: 
        - a float estimating the number of excess mortalities for the `endpoint` across 
          the group in a given geography
    
    '''
    # Beta from Krewski et al (2009) and BenMAP
    beta_dict = {'ALL CAUSE':0.005826891,
                 'ISCHEMIC HEART DISEASE': 0.021511138,
                 'LUNG CANCER': 0.013102826}
    beta = beta_dict[endpoint]
    
    return (1 - (1/np.exp(beta*conc)))*inc*pop

def create_logging_code():
    ''' Makes a global logging code for easier updating 
    
    INPUTS: None
    
    OUTPUTS: 
        - logging_code: a dictionary that maps endpoint names to log codes
    '''
    logging_code = {'ALL CAUSE':'[ACM]', 
                    'ISCHEMIC HEART DISEASE':'[IHD]', 
                    'LUNG CANCER':'[LCM]'}
    return logging_code

#%% Main Calculation Functions
def calculate_excess_mortality(conc, health_data_pop_inc, pop, endpoint, function, verbose, debug_mode):
    ''' 
    Calculate Excess Mortality 
    
    INPUTS:
        - conc: a float with the exposure concentration for a given geography
        - health_data_pop_inc: a `health_data` object's pop_inc member as defined in the `health_data.py` 
          supporting script
        - pop: a population count
        - endpoint: a string containing either 'ALL CAUSE', 'ISCHEMIC HEART DISEASE', 
          or 'LUNG CANCER'
        - function: the health impact function of choice (currently only `krewski` is 
          built out)
        - verbose: a Boolean indicating whether or not detailed logging statements 
          should be printed 
        - debug_mode: a Boolean indicating whether or not to output debug statements
        
    OUTPUTS:
        - pop_inc_conc: a dataframe containing excess mortality for the `endpoint` using
          the `function` provided
    
    '''
    
    # Set up some logging things and print statements
    logging_code = create_logging_code()[endpoint]
    logging.info('- {} Estimating excess {} mortality from PM2.5. This step may take time.'.format(logging_code, endpoint.lower()))
    
    # Get the population-incidence  and total concentration
    verboseprint(verbose, '- {} Creating dataframe to combine concentration data with {} mortality BenMAP inputs.'.format(logging_code, endpoint.lower()), debug_mode, frameinfo=getframeinfo(currentframe()))

     # Ensure conc is a GeoDataFrame by converting it explicitly.
    # Here, 'conc' is assumed to have a 'geometry' column and a CRS stored in conc.crs
    conc_hia = gpd.GeoDataFrame(conc.copy(), geometry='geometry', crs=conc.crs)

    if not isinstance(health_data_pop_inc, gpd.GeoDataFrame):
        if "geometry" in health_data_pop_inc.columns:
            health_data_pop_inc = gpd.GeoDataFrame(health_data_pop_inc, geometry="geometry")
        else:
            raise ValueError("health_data_pop_inc is missing a 'geometry' column, cannot convert to GeoDataFrame.")
    elif "geometry" not in health_data_pop_inc.columns:
        raise ValueError("health_data_pop_inc is missing a 'geometry' column, cannot convert to GeoDataFrame.")

    # Now, convert the population-incidence data to the same CRS
    if not isinstance(health_data_pop_inc, gpd.GeoDataFrame):
        if "geometry" in health_data_pop_inc.columns:
            pop_inc_conc = gpd.GeoDataFrame(health_data_pop_inc, geometry="geometry")
    
    # Merge these on ISRM_ID
    pop_inc_conc = health_data_pop_inc.merge(conc_hia[['ISRM_ID', 'TOTAL_CONC_UG/M3']], on='ISRM_ID')
    verboseprint(verbose, '- {} Successfully merged concentrations and {} input data.'.format(logging_code, endpoint.title()), debug_mode, frameinfo=getframeinfo(currentframe()))
    verboseprint(verbose, '- {} Estimating {} mortality for each ISRM grid cell.'.format(logging_code, endpoint.title()), debug_mode, frameinfo=getframeinfo(currentframe()))
        
    # Estimate excess mortality
    pop_inc_conc[endpoint] = pop_inc_conc.apply(lambda x: function(x['TOTAL_CONC_UG/M3'],
                                                                   x[endpoint+' INC'],
                                                                   x['POPULATION'],
                                                                   endpoint), axis=1)
    verboseprint(verbose, '- {} Successfully estimated {} mortality for each ISRM grid cell.'.format(logging_code, endpoint.title()), debug_mode, frameinfo=getframeinfo(currentframe()))
    verboseprint(verbose, '- {} Determining excess {} mortality by racial/ethnic group.'.format(logging_code, endpoint.title()), debug_mode, frameinfo=getframeinfo(currentframe()))
    
    # Pivot the dataframe to get races as columns
    pop_inc_conc = pop_inc_conc.pivot_table(index='ISRM_ID',columns='RACE', 
                                            values=endpoint, aggfunc='sum', 
                                            fill_value=0)
    verboseprint(verbose, '- {} Performing initial clean up of excess {} mortality data.'.format(logging_code, endpoint.title()), debug_mode, frameinfo=getframeinfo(currentframe()))

    # Add geometry back in
    pop_inc_conc = pd.merge(conc_hia, pop_inc_conc, on='ISRM_ID', how='left')
    pop_inc_conc = pop_inc_conc.fillna(0)
    pop_inc_conc = gpd.GeoDataFrame(pop_inc_conc, geometry='geometry')
    
    # Update column names
    col_rename_dict = {'ASIAN':endpoint+'_ASIAN',
                        'BLACK':endpoint+'_BLACK',
                        'HISLA':endpoint+'_HISLA',
                        'INDIG':endpoint+'_INDIG',
                        'PACIS':endpoint+'_PACIS',
                        'TOTAL':endpoint+'_TOTAL',
                        'WHITE':endpoint+'_WHITE',
                        'OTHER':endpoint+'_OTHER'}
    pop_inc_conc.rename(columns=col_rename_dict, inplace=True)
    
    # Merge the population back in
    verboseprint(verbose, '- {} Adding population data back in for per capita calculations.'.format(logging_code), debug_mode, frameinfo=getframeinfo(currentframe()))
        
    pop_inc_conc = pd.merge(pop_inc_conc, pop, left_on='ISRM_ID', right_on='ISRM_ID', how='left')
    pop_inc_conc = pop_inc_conc.fillna(0)
    
    # Final Clean Up
    verboseprint(verbose, '- {} Performing final clean up.'.format(logging_code), debug_mode, frameinfo=getframeinfo(currentframe()))

    pop_inc_conc = pop_inc_conc[['ISRM_ID', 'TOTAL_CONC_UG/M3', 'ASIAN', 'BLACK', 'HISLA',
                                 'INDIG', 'PACIS', 'WHITE', 'TOTAL', 'OTHER', endpoint+'_ASIAN', endpoint+'_BLACK', 
                                 endpoint+'_HISLA', endpoint+'_INDIG',endpoint+'_TOTAL', 
                                 endpoint+'_WHITE', endpoint+'_PACIS', endpoint+'_OTHER', 'geometry']]
    
    # Print statement
    logging.info('- {} {} health impacts calculated.'.format(logging_code, endpoint.title()))
    
    return pop_inc_conc


#%% Formatting and Exporting Functions
def plot_total_mortality(hia_df, ca_shp_fp, group, endpoint, output_resolution, boundary, output_dir, f_out, verbose, debug_mode):
    ''' 
    Plots mortality maps and exports as a png. 
    
    INPUTS:
        - hia_df: a dataframe containing excess mortality for the `endpoint` using the `function`
          provided
        - ca_shp_fp: a filepath string of the California state boundary shapefile
        - group: the racial/ethnic group name
        - endpoint: a string containing either 'ALL CAUSE', 'ISCHEMIC HEART DISEASE', or 
          'LUNG CANCER'
        - output_resolution: a String that represents the output resolution 
        - boundary: a GeoDataFrame that represents the output resolution data
        - output_dir: a filepath string of the location of the output directory
        - f_out: the name of the file output category (will append additional information) 
        - verbose: a Boolean indicating whether or not detailed logging statements should 
          be printed
        - debug_mode: a Boolean indicating whether or not to output debug statements
        
    OUTPUTS:
        - fname: a string filename made by combining the `f_out` with the `group`
          and `endpoint`.
          
    '''
    logging_code = create_logging_code()[endpoint]
    verboseprint(verbose, '- {} Drawing plot of excess {} mortality from PM2.5 exposure.'.format(logging_code, endpoint.lower()), debug_mode, frameinfo=getframeinfo(currentframe()))
    
    sns.set_theme(context="notebook", style="whitegrid", font_scale=1.25)
    plt.rcParams['patch.linewidth'] = 0
    plt.rcParams['patch.edgecolor'] = 'none'
    plt.rcParams["patch.force_edgecolor"] = False
    
    # Create the output file directory and name string
    fname = f_out + '_' + group + '_' + endpoint + '_excess_mortality.png'
    fname = str.lower(fname)
    fpath = os.path.join(output_dir, fname)
    
    # Read in CA boundary and project hia_df to same coordinates (meters)
    ca_shp = gpd.read_feather(ca_shp_fp)
    hia_df = hia_df.to_crs(ca_shp.crs)
    
    # Clip dataframe to California
    hia_df = gpd.clip(hia_df, ca_shp)

    # Determine which column to use
    group = group.upper()
    mortality_col = endpoint + '_' + group
    group_label = group.title()
    
    # Set true zeros to 10^-9 to avoid divide by zero issues
    hia_df.loc[hia_df[group]==0,group] = 10.0**-9.0
    hia_df.loc[hia_df[mortality_col]==0, mortality_col] = 10.0**-9.0

    # Add new columns to hia_df for plotting
    hia_df['POP_AREA_NORM'] = hia_df[group] / hia_df.area * 1000.0 * 1000.0
    hia_df['MORT_AREA_NORM'] = hia_df[mortality_col] / hia_df.area * 1000.0 * 1000.0
    hia_df['MORT_OVER_POP'] = hia_df[mortality_col] / hia_df[group] * 100000.0    
    
    # Grab the minimums that do not include the surrogate zeros
    hia_pop_area_min = hia_df.loc[hia_df[group] > 10.0**-9.0, 'POP_AREA_NORM'].min()
    hia_mort_area_min = hia_df.loc[hia_df[mortality_col] > 10.0**-9.0, 'MORT_AREA_NORM'].min()
    
    # Update MORT_OVER_POP to avoid 100% mortality in areas where there is no population
    hia_df.loc[hia_df[group] == hia_df[mortality_col], 'MORT_OVER_POP'] = hia_df['MORT_OVER_POP'].min() * 0.0001

    # Initialize the figure as four panes
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(22, 6))

    ## Pane 0: Population Density
    hia_df.plot(column='POP_AREA_NORM', legend=True,
                legend_kwds={'label': r'Population Density (population/km$^2$)'},
                edgecolor='none', cmap='mako_r',
                norm=matplotlib.colors.LogNorm(vmin=hia_pop_area_min,
                                                vmax=hia_df['POP_AREA_NORM'].max()),
                antialiased=False,
                ax=ax0)
    
    ## Pane 1: PM2.5 Exposure Concentration
    hia_df.plot(column='TOTAL_CONC_UG/M3', legend=True,
                legend_kwds={'label': r'PM$_{2.5}$ Concentration ($\mu$g/m$^3$)'},
                edgecolor='none', cmap='mako_r',
                norm=matplotlib.colors.LogNorm(vmin=hia_df['TOTAL_CONC_UG/M3'].min(),
                                                vmax=hia_df['TOTAL_CONC_UG/M3'].max()),
                antialiased=False,
                ax=ax1)
    
    ## Pane 2: Excess Mortality per Area
    hia_df.plot(column='MORT_AREA_NORM', legend=True,
                legend_kwds={'label': r'Excess Mortality (mortality/km$^2$)'},
                edgecolor='none', cmap='mako_r',
                norm=matplotlib.colors.LogNorm(vmin=hia_mort_area_min,
                                                vmax=hia_df['MORT_AREA_NORM'].max()),
                antialiased=False,
                ax=ax2)
    
    ## Pane 3: Excess Mortality per Population
    hia_df.plot(column='MORT_OVER_POP', legend=True,
                legend_kwds={'label': r'Mortality per Population (mortality/100 K people)'},
                edgecolor='none', cmap='mako_r',
                norm=matplotlib.colors.LogNorm(vmin=hia_df['MORT_OVER_POP'].min(),
                                                vmax=hia_df['MORT_OVER_POP'].max()),
                antialiased=False,
                ax=ax3)

    # Figure Formatting
    minx, miny, maxx, maxy = hia_df.geometry.total_bounds
    minx = minx - (maxx - minx) * 0.025
    miny = miny - (maxy - miny) * 0.025
    maxx = maxx + (maxx - minx) * 0.025
    maxy = maxy + (maxy - miny) * 0.025

    # Calculates the longitude and latitude of the center
    center_lon, center_lat = (minx + maxx) / 2, (miny + maxy) / 2

    # Calculate the north arrow angle 
    angle_to_north = calculate_true_north_angle(center_lon, center_lat, hia_df.crs)
    
    # Calculates the longitude and latitude of the center
    center_lon, center_lat = (minx + maxx) / 2, (miny + maxy) / 2

    # Calculates the angle of the north arrow 
    angle_to_north = calculate_true_north_angle(center_lon, center_lat, ca_shp.crs)

    for ax in [ax0, ax1, ax2, ax3]:
        ca_shp.dissolve().plot(edgecolor='black', facecolor='none', linewidth=1, ax=ax)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_xlim([minx, maxx])
        ax.set_ylim([miny, maxy])

        # Add north arrow
        add_north_arrow(ax, float(angle_to_north))

        # Add scale bar
        scalebar = ScaleBar(1, location='lower left', border_pad=0.5)  # 1 pixel = 1 unit
        ax.add_artist(scalebar)

    # Set titles
    ax0.set_title((group_label + ' Population Density').title())
    ax1.set_title((group_label + ' Exposure').title())
    ax2.set_title((group_label + ' ' + endpoint + ' Excess Mortality').title())
    ax3.set_title((group_label + ' ' + endpoint + ' Mortality per 100K').title())

    # Final cleanup
    fig.tight_layout()
    
    # Export the original plot
    fig.savefig(fpath, dpi=200)
    logging.info('- {} Plot of excess {} mortality from PM2.5 exposure output as {}'.format(logging_code, endpoint.lower(), fname))

    # Check if the output resolution requires a second plot
    if output_resolution in ['AB', 'AD', 'C']:
    
      # Ensure CRS match
      if boundary.crs != hia_df.crs:
          boundary = boundary.to_crs(hia_df.crs)

      #Create a hia_df copy? 
      hia_df2 = hia_df.copy()

      # Perform intersection
      intersect = gpd.overlay(hia_df, boundary, keep_geom_type=False, how='intersection')

      # Calculate area and fractions
      intersect['area_km2'] = intersect.geometry.area / 1e6
      total_area = intersect.groupby('NAME').sum(numeric_only=True)['area_km2'].to_dict()
      intersect['area_total'] = intersect['NAME'].map(total_area)
      intersect['area_frac'] = intersect['area_km2'] / intersect['area_total']

      # Aggregate directly by region without normalizing by area fraction
      region_data = intersect.groupby(['NAME']).agg({
          'area_km2': 'sum',  # Total area of intersected regions
          group: 'sum',  # Total population
          mortality_col: 'sum',  # Total excess mortality
          'TOTAL_CONC_UG/M3': 'mean'  # Ensure this column is aggregated correctly
      }).reset_index()

      # Calculate population density and mortality 
      region_data['POP_AREA_NORM'] = region_data[group] / region_data['area_km2']
      region_data['MORT_AREA_NORM'] = region_data[mortality_col] / region_data['area_km2']
      region_data['MORT_OVER_POP'] = (region_data[mortality_col] / region_data[group]) * 1e5
      
      # Merge with boundary to get full geometry
      hia_df = pd.merge(boundary, region_data, on='NAME', how='left')

      # Set true zeros to avoid divide by zero issues
      if group in hia_df.columns:
          hia_df.loc[hia_df[group] == 0, group] = 1e-9
      if endpoint + '_' + group in hia_df.columns:
          hia_df.loc[hia_df[endpoint + '_' + group] == 0, endpoint + '_' + group] = 1e-9

      # Plotting
      fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(22, 6))

      # Clip the
      hia_df = gpd.clip(hia_df, hia_df2)

      ## Pane 0: Population Density
      hia_df.plot(column='POP_AREA_NORM', legend=True,
                  legend_kwds={'label': r'Population Density (population/km$^2$)'},
                  edgecolor='none', cmap='mako_r',
                  norm=matplotlib.colors.LogNorm(vmin=hia_df['POP_AREA_NORM'].min(), 
                                                  vmax=hia_df['POP_AREA_NORM'].max()),
                  antialiased=False,
                  ax=ax0)

      ## Pane 1: PM2.5 Exposure Concentration (Population-Weighted)
      hia_df.plot(column='TOTAL_CONC_UG/M3', legend=True,
                  legend_kwds={'label': r'Population-Weighted PM$_{2.5}$ Concentration ($\mu$g/m$^3$)'},
                  edgecolor='none', cmap='mako_r',
                  norm=matplotlib.colors.LogNorm(vmin=hia_df['TOTAL_CONC_UG/M3'].min(), 
                                                  vmax=hia_df['TOTAL_CONC_UG/M3'].max()),
                  antialiased=False,
                  ax=ax1)

      ## Pane 2: Excess Mortality per Area
      hia_df.plot(column='MORT_AREA_NORM', legend=True,
                  legend_kwds={'label': r'Excess Mortality (mortality/km$^2$)'},
                  edgecolor='none', cmap='mako_r',
                  norm=matplotlib.colors.LogNorm(vmin=hia_df['MORT_AREA_NORM'].min(),
                                                  vmax=hia_df['MORT_AREA_NORM'].max()),
                  antialiased=False,
                  ax=ax2)

      ## Pane 3: Excess Mortality per Population
      hia_df.plot(column='MORT_OVER_POP', legend=True,
                  legend_kwds={'label': r'Mortality per Population (mortality/100 K people)'},
                  edgecolor='none', cmap='mako_r',
                  norm=matplotlib.colors.LogNorm(vmin=hia_df['MORT_OVER_POP'].min(),
                                                  vmax=hia_df['MORT_OVER_POP'].max()),
                  antialiased=False,
                  ax=ax3)


      # Plotting each map
      for ax in [ax0, ax1, ax2, ax3]: 
          boundary.dissolve().plot(edgecolor='black', facecolor='none', linewidth=1, ax=ax)
          ax.xaxis.set_visible(False)
          ax.yaxis.set_visible(False)

          # Can use the same bounds stated previously
          ax.set_xlim([minx, maxx])
          ax.set_ylim([miny, maxy])

          # Add north arrow
          add_north_arrow(ax, float(angle_to_north))
            
          # Add scale bar
          scalebar = ScaleBar(1, location='lower left', border_pad=0.5)  # 1 pixel = 1 unit
          ax.add_artist(scalebar)

      # Set titles
      ax0.set_title((group + ' Population Density').title())
      ax1.set_title((group + ' Population-Weighted Exposure').title())
      ax2.set_title((group + ' ' + endpoint + ' Excess Mortality').title())
      ax3.set_title((group + ' ' + endpoint + ' Mortality per 100K').title())

      # Final cleanup
      fig.tight_layout()

      # Export the aggregated plot
      fname_aggregated = f_out + '_' + group + '_' + endpoint + '_excess_mortality_aggregated.png'
      fname_aggregated = str.lower(fname_aggregated)
      fpath_aggregated = os.path.join(output_dir, fname_aggregated)
      fig.savefig(fpath_aggregated, dpi=200)
      logging.info('- {} Plot of excess {} mortality from PM2.5 exposure at aggregated resolution output as {}'.format(logging_code, endpoint.lower(), fname_aggregated))
    
    return fname, fname_aggregated if output_resolution in ['AB', 'AD', 'C'] else fname

def export_health_impacts(hia_df, group, endpoint, output_dir, f_out, verbose, debug_mode):
    ''' 
    Plots mortality as a shapefile. 
    
    INPUTS:
        - hia_df: a dataframe containing excess mortality for the `endpoint` using the 
          `function` provided
        - group: the racial/ethnic group name
        - endpoint: a string containing either 'ALL CAUSE', 'ISCHEMIC HEART DISEASE', or 
          'LUNG CANCER'
        - output_dir: a filepath string of the location of the output directory
        - f_out: the name of the file output category (will append additional information) 
        - verbose: a Boolean indicating whether or not detailed logging statements should 
          be printed  
        - debug_mode: a Boolean indicating whether or not to output debug statements
        
    OUTPUTS:
        - fname: a string filename made by combining the `f_out` with the `group`
          and `endpoint`.
        
    '''
    logging_code = create_logging_code()[endpoint]
    verboseprint(verbose, '- {} Exporting excess {} mortality from PM2.5 exposure as a shapefile.'.format(logging_code, endpoint.lower()), debug_mode, frameinfo=getframeinfo(currentframe()))
        
    # Create the output file directory and name string
    fname = f_out + '_' + group + '_' + endpoint + '_excess_mortality.shp'
    fname = str.lower(fname)
    fpath = os.path.join(output_dir, fname)
    logging_code = create_logging_code()[endpoint]
    
    # Get endpoint shortlabel
    endpoint_labels = {'ALL CAUSE':'ACM_',
                       'ISCHEMIC HEART DISEASE':'IHD_',
                       'LUNG CANCER':'CAN_'}
    l = endpoint_labels[endpoint]
    
    # Update column names
    col_name_dict = {'TOTAL_CONC_UG/M3':'CONC_UG/M3', 
                     'ASIAN':'POP_ASIAN', 
                     'BLACK':'POP_BLACK',
                     'HISLA':'POP_HISLA',
                     'INDIG':'POP_INDIG',
                     'PACIS':'POP_PACIS',
                     'WHITE':'POP_WHITE',
                     'TOTAL':'POP_TOTAL',
                     'OTHER':'POP_OTHER',
                     endpoint+'_ASIAN':l+'ASIAN',
                     endpoint+'_BLACK':l+'BLACK',
                     endpoint+'_HISLA':l+'HISLA',
                     endpoint+'_INDIG':l+'INDIG',
                     endpoint+'_PACIS':l+'PACIS',
                     endpoint+'_TOTAL':l+'TOTAL',
                     endpoint+'_WHITE':l+'WHITE',
                     endpoint+'_OTHER':l+'OTHER'}
    hia_df.rename(columns=col_name_dict, inplace=True)
    
    # Export
    hia_df.to_file(fpath)
    logging.info('- {} Excess {} mortality from PM2.5 exposure output as a shapefile as {}'.format(logging_code, endpoint.lower(), fname))
    
    return fname

def export_health_impacts_csv(hia_df, endpoint, output_dir, f_out, verbose, debug_mode):
    ''' 
    Exports total mortality as a csv file. 
    
    INPUTS:
        - hia_df: a dataframe containing excess mortality for the `endpoint` using the 
          `function` provided
        - endpoint: a string containing either 'ALL CAUSE', 'ISCHEMIC HEART DISEASE', or 
          'LUNG CANCER'
        - output_dir: a filepath string of the location of the output directory
        - f_out: the name of the file output category (will append additional information) 
        - verbose: a Boolean indicating whether or not detailed logging statements should 
          be printed  
        - debug_mode: a Boolean indicating whether or not to output debug statements
        
    OUTPUTS:
        - fname: a string filename made by combining the `f_out` with the `group`
          and `endpoint`.
        
    '''
    logging_code = create_logging_code()[endpoint]
    verboseprint(verbose, '- {} Exporting excess {} mortality from PM2.5 exposure as a CSV file.'.format(logging_code, endpoint.lower()), debug_mode, frameinfo=getframeinfo(currentframe()))
        
    # Create the output file directory and name string
    fname = f_out + '_' + endpoint + '_excess_mortality.csv'
    fname = str.lower(fname)
    fpath = os.path.join(output_dir, fname)
    logging_code = create_logging_code()[endpoint]
    
    # Get endpoint shortlabel
    endpoint_nice = endpoint.title()
    endpoint_labels = {'ALL CAUSE':'ACM_',
                       'ISCHEMIC HEART DISEASE':'IHD_',
                       'LUNG CANCER':'CAN_'}
    l = endpoint_labels[endpoint]
    
    # Create the summary HIA
    hia_summary = create_summary_hia(hia_df, endpoint, verbose, l, endpoint_nice, debug_mode)
    
    ## Update column names
    # Import the rename dictionary and make a few edits
    rename_dict = create_rename_dict()
    pop_rename_dict = {'POP_'+k: v + ' (# People)' for k, v in rename_dict.items()} # Add units to population
    hia_rename_dict = {l+'_'+k:endpoint_nice+' - '+v+' (excess deaths)' for k,v in rename_dict.items()}

    # Rename the columns in series
    hia_df.rename(columns=pop_rename_dict, inplace=True)
    hia_df.rename(columns=hia_rename_dict, inplace=True)
    hia_df.rename(columns={'TOTAL_CONC_UG/M3':'PM2.5 Concentration (ug/m3)'}, inplace=True)
    
    # Get rid of geometry column
    hia_df.drop(['geometry'], axis=1)
    
    # Export
    hia_df.to_csv(fpath, index=False)
    logging.info('- {} Excess {} mortality from PM2.5 exposure output as a CSV as {}'.format(logging_code, endpoint.lower(), fname))
    
    return hia_summary

def create_summary_hia(hia_df, endpoint, verbose, l, endpoint_nice, debug_mode):
    ''' 
    Creates a summary table of health impacts by racial/ethnic group 
    
    INPUTS:
        - hia_df: a dataframe containing excess mortality for the `endpoint` using the 
          `function` provided
        - endpoint: a string containing either 'ALL CAUSE', 'ISCHEMIC HEART DISEASE', or 
          'LUNG CANCER'
        - verbose: a Boolean indicating whether or not detailed logging statements should 
          be printed   
        - l: an intermediate string that has the endpoint label string (e.g., ACM_)
        - endpoint_nice: an intermediate string that has a nicely formatted version
          of the endpoint (e.g., All Cause)
        - debug_mode: a Boolean indicating whether or not to output debug statements
    
    OUTPUTS:
        - hia_summary: a summary dataframe containing population, excess mortality,
          and excess mortality rate per demographic group.
        
    '''
    logging_code = create_logging_code()[endpoint]
    verboseprint(verbose, '- {} Creating a summary table of {} mortality from PM2.5 exposure.'.format(logging_code, endpoint.lower()), debug_mode, frameinfo=getframeinfo(currentframe()))

    # Set up a few useful variables
    groups = ['ASIAN', 'BLACK', 'HISLA', 'INDIG', 'PACIS', 'WHITE', 'TOTAL', 'OTHER']
    pop_cols = ['POP_'+grp for grp in groups]
    hia_cols = [l+grp for grp in groups]
        
    # Clean up the hia_df dataframe
    hia_df.drop(['geometry'], axis=1)
    
    # Get the summary results for population
    pop_df = hia_df[pop_cols].sum().reset_index()
    pop_df.rename(columns={'index':'Label',0:'Population (# People)'}, inplace=True)
    pop_df['Group'] = pop_df['Label'].str.split('_').str[1]
    pop_df = pop_df[['Group','Population (# People)']].copy()
    
    # Get the summary results for excess mortality
    exm_df = hia_df[hia_cols].sum().reset_index()
    exm_df.rename(columns={'index':'Label',0:endpoint_nice+' Mortality (# Excess Deaths)'}, inplace=True)
    exm_df['Group'] = exm_df['Label'].str.split('_').str[1]
    exm_df = exm_df[['Group', endpoint_nice+' Mortality (# Excess Deaths)']].copy()
    
    # Combine into a summary table (do not export as CSV yet)
    hia_summary = pd.merge(pop_df, exm_df, on='Group')
    hia_summary['Mortality Rate (per 100000)'] = hia_summary[endpoint_nice+' Mortality (# Excess Deaths)']/hia_summary['Population (# People)'] * 100000.0
    
    # Revise the Group column for clarity
    rename_dict = create_rename_dict()
    hia_summary['Group'] = hia_summary['Group'].map(rename_dict)
    
    return hia_summary

def visualize_and_export_hia(hia_df, ca_shp_fp, group, endpoint, output_dir, f_out, shape_out, output_resolution, boundary, verbose, debug_mode):
    ''' 
    Automates this process a bit.
    
    INPUTS:
        - hia_df: a dataframe containing excess mortality for the `endpoint` using the 
          `function` provided
        - ca_shp_fp: a filepath string of the California state boundary shapefile
        - group: the racial/ethnic group name
        - endpoint: a string containing either 'ALL CAUSE', 'ISCHEMIC HEART DISEASE', or 
          'LUNG CANCER'
        - output_dir: a filepath string of the location of the output directory
        - f_out: the name of the file output category (will append additional information) 
        - shape_out: a filepath string for shapefiles
        - output_resolution: a String that represents the output resoluotion 
        - boundary: a GeoDataFrame that represents the output resolution data
        - verbose: a Boolean indicating whether or not detailed logging statements should 
          be printed      
        - debug_mode: a Boolean indicating whether or not to output debug statements
        
    OUTPUTS:
        - hia_summary: a summary dataframe containing population, excess mortality,
          and excess mortality rate per demographic group.
    
    '''    
    logging_code = create_logging_code()[endpoint]
    logging.info('- {} Visualizing and exporting excess {} mortality.'.format(logging_code, endpoint.lower()))
    
    # Plot the map of mortality
    fname = plot_total_mortality(hia_df, ca_shp_fp, group, endpoint, output_resolution, boundary, output_dir, f_out, verbose, debug_mode)
    
    # Export the shapefile
    fname = export_health_impacts(hia_df, group, endpoint, shape_out, f_out, verbose, debug_mode)
    hia_summary = export_health_impacts_csv(hia_df, endpoint, output_dir, f_out, verbose, debug_mode)
        
    return hia_summary

def combine_hia_summaries(acm_summary, ihd_summary, lcm_summary, output_dir, f_out, verbose):
    '''
    Combines the three endpoint summary tables into one export file
    
    INPUTS:
        - acm_summary: a summary dataframe containing population, excess all-cause 
          mortality, and all-cause mortality rates
        - ihd_summary: a summary dataframe containing population, excess IHD 
          mortality, and IHD mortality rates 
        - lcm_summary: a summary dataframe containing population, excess lung cancer 
          mortality, and lung cancer mortality rates
        - output_dir: a filepath string of the location of the output directory
        - f_out: the name of the file output category (will append additional information) 
        - verbose: a Boolean indicating whether or not detailed logging statements should 
          be printed      
        
    OUTPUTS: None
        
    '''
    # Merge ACM and IHD first, then add LCM
    hia_summary = pd.merge(acm_summary, ihd_summary, on='Group')
    hia_summary = pd.merge(hia_summary, lcm_summary, on='Group')
    
    # Keep only necessary columns
    hia_summary = hia_summary.loc[:, ~hia_summary.columns.str.startswith('Mortality')]
    hia_summary = hia_summary.loc[:, ~hia_summary.columns.str.endswith('x')]
    hia_summary = hia_summary.loc[:, ~hia_summary.columns.str.endswith('y')]
    
    
    # Export results
    fname = f_out + '_excess_mortality_summary.csv'
    fname = str.lower(fname)
    fpath = os.path.join(output_dir, fname)
    hia_summary.to_csv(fpath, index=False)
    
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
