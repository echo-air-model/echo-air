#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Health Impact Functions

@author: libbykoolik

last modified: 2025-06-05

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
def create_hia_inputs(pop, population_columns, load_file: bool, verbose: bool, geodata:pd.DataFrame,
                          incidence_fp: str, debug_mode:bool):
        """ Creates the hia_inputs object.
        
            Moving this into a separate function allows us to run this in parallel while
            other functions are running, speeding up the overall execution of the
            application.
        
        INPUTS:
            - pop: the population object input
            - population_columns: a list of population columns to use from the population input file
            - load_file: a boolean telling program to load or not
            - verbose: a boolean telling program to return additional log statements or not
            - geodata: the geographic data from the ISRM 
            - incidence_fp: the filepath where the incidence data is stored
            
        OUTPUTS: 
            - a health data object ready for health calculations
        
        """
        hia_pop_alloc = pop.allocate_pop(pop.pop_all, geodata, True)
        return health_data(hia_pop_alloc, population_columns, incidence_fp, verbose=verbose, race_stratified=False, debug_mode=debug_mode)

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
                        'LUNG CANCER':'[LCM]',
                        'HAZARD QUOTIENT':'[HQ]',
                        'CANCER RISK':'[DPM]'}
        return logging_code

#%% Main Calculation Functions
def calculate_excess_mortality(population_columns, conc, health_data_pop_inc, pop, endpoint, function, verbose, debug_mode, dpm = False):
        ''' 
        Calculate Excess Mortality 
        
        INPUTS:
            - population_columns: a list of population columns to use from the population input file
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
        # Define dictionary for column of interest
        column = {"DPM":"RISK", "PM25":"TOTAL_CONC_UG/M3"}

        # Change logging statement depending on pollutant
        if dpm:
          logging_code = create_logging_code()[endpoint]
          endpoint_full = endpoint + " incidence"
          pollutant = "DPM"
          # CHECK
          logging.info('- {} Estimating excess {} from DPM. This step may take time.'.format(logging_code, endpoint_full.lower()))
        else:
        # Set up some logging things and print statements
          logging_code = create_logging_code()[endpoint]
          endpoint_full = endpoint + " mortality"
          pollutant = "PM25"
          logging.info('- {} Estimating excess {} from PM2.5. This step may take time.'.format(logging_code, endpoint_full.lower()))
        
        # Get the population-incidence  and total concentration
        verboseprint(verbose, '- {} Creating dataframe to combine concentration data with {} BenMAP inputs.'.format(logging_code, endpoint_full.lower()), debug_mode, frameinfo=getframeinfo(currentframe()))

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
        pop_inc_conc = health_data_pop_inc.merge(conc_hia[['ISRM_ID', column[pollutant]]], on='ISRM_ID')
        verboseprint(verbose, '- {} Successfully merged concentrations and {} input data.'.format(logging_code, endpoint.title()), debug_mode, frameinfo=getframeinfo(currentframe()))
        verboseprint(verbose, '- {} Estimating {} for each ISRM grid cell.'.format(logging_code, endpoint_full.title()), debug_mode, frameinfo=getframeinfo(currentframe()))
            
        # Estimate excess mortality
        if dpm:
             pop_inc_conc[endpoint] = pop_inc_conc.apply(lambda x: function(x['RISK'], x["POPULATION"]), axis=1)
        else:
             pop_inc_conc[endpoint] = pop_inc_conc.apply(lambda x: function(x['TOTAL_CONC_UG/M3'],
                                                                      x[endpoint+' INC'],
                                                                      x['POPULATION'],
                                                                      endpoint), axis=1)
        verboseprint(verbose, '- {} Successfully estimated {} for each ISRM grid cell.'.format(logging_code, endpoint_full.title()), debug_mode, frameinfo=getframeinfo(currentframe()))
        verboseprint(verbose, '- {} Determining excess {} by racial/ethnic group.'.format(logging_code, endpoint_full.title()), debug_mode, frameinfo=getframeinfo(currentframe()))
        
        # Pivot the dataframe to get races as columns
        pop_inc_conc = pop_inc_conc.pivot_table(index='ISRM_ID',columns='RACE', 
                                                values=endpoint, aggfunc='sum', 
                                                fill_value=0)
        verboseprint(verbose, '- {} Performing initial clean up of excess {} data.'.format(logging_code, endpoint_full.title()), debug_mode, frameinfo=getframeinfo(currentframe()))

        # Add geometry back in
        pop_inc_conc = pd.merge(conc_hia, pop_inc_conc, on='ISRM_ID', how='left')
        pop_inc_conc = pop_inc_conc.fillna(0)
        pop_inc_conc = gpd.GeoDataFrame(pop_inc_conc, geometry='geometry')
        
        # Update column names
        col_rename_dict = {col: f"{endpoint}_{col}" for col in population_columns}
              
        pop_inc_conc.rename(columns=col_rename_dict, inplace=True)
        
        # Merge the population back in
        verboseprint(verbose, '- {} Adding population data back in for per capita calculations.'.format(logging_code), debug_mode, frameinfo=getframeinfo(currentframe()))
            
        pop_inc_conc = pd.merge(pop_inc_conc, pop, left_on='ISRM_ID', right_on='ISRM_ID', how='left')
        pop_inc_conc = pop_inc_conc.fillna(0)
        
        # Final Clean Up
        verboseprint(verbose, '- {} Performing final clean up.'.format(logging_code), debug_mode, frameinfo=getframeinfo(currentframe()))

        col_endpoint = [f"{endpoint}_{col}" for col in population_columns]
        pop_inc_conc = pop_inc_conc[['ISRM_ID', column[pollutant]] + population_columns + col_endpoint +  ['geometry']]
        
        # Print statement
        logging.info('- {} {} health impacts calculated.'.format(logging_code, endpoint.title()))

        # Add if statement here and only get concentrations if DPM is flagged, then call DPM health function calculations (CHECK)
        
        return pop_inc_conc


#%% Formatting and Exporting Functions
def plot_total_mortality(hia_df, ca_shp_fp, group, endpoint, output_resolution, boundary, output_dir, f_out, verbose, debug_mode, pollutant):
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

        if pollutant == 'DPM':
            endpoint_full = endpoint + " incidence"
            #Begin creating output file directory and name string
            fname = fname = f_out + '_' + group + '_' + endpoint + '_excess_incidence.png'
        else:
            endpoint_full = endpoint + " mortality"
            #Begin creating output file directory and name string
            fname = f_out + '_' + group + '_' + endpoint + '_excess_mortality.png'
        
        verboseprint(verbose, '- {} Drawing plot of excess {} from {} exposure.'.format(logging_code, endpoint_full.lower(), pollutant), debug_mode, frameinfo=getframeinfo(currentframe()))
        
        sns.set_theme(context="notebook", style="whitegrid", font_scale=1.25)
        plt.rcParams['patch.linewidth'] = 0
        plt.rcParams['patch.edgecolor'] = 'none'
        plt.rcParams["patch.force_edgecolor"] = False
        
        # Contine creating the output file directory and name string
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

        # Define labels based on pollutant being plotted
        if pollutant == 'PM2.5':
          col = 'TOTAL_CONC_UG/M3'
          pretty_pol = 'PM$_{2.5}$'
          outcome_word = 'Mortality'
          outcome_word_lower = 'mortality'
          per_area_label = r'Excess Mortality (mortality/km$^2$)'
          per_pop_label = r'Mortality per Population (mortality/100 K people)'
          file_suffix = 'excess_mortality'
        else:  # DPM
          col = 'RISK'
          pretty_pol = 'DPM'
          outcome_word = 'Incidence'
          outcome_word_lower = 'incidence'
          per_area_label = r'Excess Incidence (incidence/km$^2$)'
          per_pop_label = r'Incidence per Population (incidence/100 K people)'
          file_suffix = 'excess_incidence'

        # Check for negative bounds that will break the LogNorm
        if hia_df[col].min() < 0:
            logging.info('* {} Negative concentrations and outcomes detected for {}. Health outcome plots may not represent true outcome distributions in space.'.format(logging_code, fname))

        ## Panel 0: Population Density
        hia_df.plot(column='POP_AREA_NORM', legend=True,
                    legend_kwds={'label': r'Population Density (population/km$^2$)'},
                    edgecolor='none', cmap='mako_r',
                    norm=matplotlib.colors.LogNorm(vmin=hia_pop_area_min,
                                                    vmax=hia_df['POP_AREA_NORM'].max()),
                    antialiased=False,
                    ax=ax0)
        
        ## Panel 1: PM2.5 Exposure Concentration
        hia_df.plot(column=col, legend=True,
                    legend_kwds={'label': fr'{pretty_pol} Concentration ($\mu$g/m$^3$)'},
                    edgecolor='none', cmap='mako_r',
                    norm=matplotlib.colors.LogNorm(vmin=max(hia_df[col].min(),1e-9),
                                                    vmax=hia_df[col].max()),
                    antialiased=False,
                    ax=ax1)
        
        ## Panel 2: Excess Mortality per Area
        hia_df.plot(column='MORT_AREA_NORM', legend=True,
                    legend_kwds={'label': per_area_label},
                    edgecolor='none', cmap='mako_r',
                    norm=matplotlib.colors.LogNorm(vmin=hia_mort_area_min,
                                                    vmax=hia_df['MORT_AREA_NORM'].max()),
                    antialiased=False,
                    ax=ax2)
        
        ## Panel 3: Excess Mortality per Population
        hia_df.plot(column='MORT_OVER_POP', legend=True,
                    legend_kwds={'label': per_pop_label},
                    edgecolor='none', cmap='mako_r',
                    norm=matplotlib.colors.LogNorm(vmin=max(hia_df['MORT_OVER_POP'].min(),1e-9),
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
        ax2.set_title(f"{group_label} {endpoint} Excess {outcome_word}")
        ax3.set_title(f"{group_label} {endpoint} {outcome_word} per 100K")

        # Final cleanup
        fig.tight_layout()
        
        # Export the original plot
        fig.savefig(fpath, dpi=200)
        logging.info('- {} Plot of excess {} {} from {} exposure output as {}'.format(logging_code, endpoint.lower(), outcome_word_lower, pollutant, fname))

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
              col : 'mean'  # Ensure this column is aggregated correctly
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
                      norm=matplotlib.colors.LogNorm(vmin=max(hia_df['POP_AREA_NORM'].min(),1e-9), 
                                                      vmax=hia_df['POP_AREA_NORM'].max()),
                      antialiased=False,
                      ax=ax0)

          ## Pane 1: PM2.5 Exposure Concentration (Population-Weighted)
          hia_df.plot(column=col, legend=True,
                      legend_kwds={'label': fr'Population-Weighted {pretty_pol} Concentration ($\mu$g/m$^3$)'},
                      edgecolor='none', cmap='mako_r',
                      norm=matplotlib.colors.LogNorm(vmin=max(hia_df[col].min(),1e-9), 
                                                      vmax=hia_df[col].max()),
                      antialiased=False,
                      ax=ax1)

          ## Pane 2: Excess Mortality per Area
          hia_df.plot(column='MORT_AREA_NORM', legend=True,
                      legend_kwds={'label': per_area_label},
                      edgecolor='none', cmap='mako_r',
                      norm=matplotlib.colors.LogNorm(vmin=max(hia_df['MORT_AREA_NORM'].min(),1e-9),
                                                      vmax=hia_df['MORT_AREA_NORM'].max()),
                      antialiased=False,
                      ax=ax2)

          ## Pane 3: Excess Mortality per Population
          hia_df.plot(column='MORT_OVER_POP', legend=True,
                      legend_kwds={'label': per_pop_label},
                      edgecolor='none', cmap='mako_r',
                      norm=matplotlib.colors.LogNorm(vmin=max(hia_df['MORT_OVER_POP'].min(),1e-9),
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
          ax2.set_title(f"{group_label} {endpoint} Excess {outcome_word}")
          ax3.set_title(f"{group_label} {endpoint} {outcome_word} per 100K")

          # Final cleanup
          fig.tight_layout()

          # Export the aggregated plot
          fname_aggregated = f_out + '_' + group + '_' + endpoint + f'_{file_suffix}_aggregated.png'
          fname_aggregated = str.lower(fname_aggregated)
          fpath_aggregated = os.path.join(output_dir, fname_aggregated)
          fig.savefig(fpath_aggregated, dpi=200)
          logging.info('- {} Plot of excess {} {} from {} exposure at aggregrated resolution output as {}'.format(logging_code, endpoint.lower(), outcome_word_lower, pollutant, fname))
        
        return fname, fname_aggregated if output_resolution in ['AB', 'AD', 'C'] else fname

def export_health_impacts(hia_df, population_columns, group, endpoint, output_dir, f_out, verbose, debug_mode, pollutant = 'PM2.5'):
        ''' 
        Plots mortality as a shapefile. 
        
        INPUTS:
            - hia_df: a dataframe containing excess mortality for the `endpoint` using the 
              `function` provided
            - population_columns: a list of population columns to use from the population input file
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
        
        # Create file name depending on pollutant
        if pollutant == 'DPM':
          endpoint_full = endpoint + " incidence"
          file_ending = '_excess_incidence.shp'
        else:
          endpoint_full = endpoint + " mortality"
          file_ending = '_excess_mortality.shp'
        verboseprint(verbose, '- {} Exporting excess {} from {} exposure as a shapefile.'.format(logging_code, endpoint_full.lower(), pollutant), debug_mode, frameinfo=getframeinfo(currentframe()))
            
        # Create the output file directory and name string
        fname = f_out + '_' + group + '_' + endpoint + file_ending
        fname = str.lower(fname)
        fpath = os.path.join(output_dir, fname)
        logging_code = create_logging_code()[endpoint]
        
        # Get endpoint shortlabel
        endpoint_labels = {'ALL CAUSE':'ACM_',
                          'ISCHEMIC HEART DISEASE':'IHD_',
                          'LUNG CANCER':'CAN_',
                          'HAZARD QUOTIENT':'HQ_',
                          "CANCER RISK": "CR_"
                          }
        l = endpoint_labels[endpoint]

        # Update column names
        col_name_dict = {'TOTAL_CONC_UG/M3':'CONC_UG/M3'}
        col_name_dict.update({col: f"POP_{col}" for col in population_columns})
        endpoint_cols = [f"{endpoint}_{col}" for col in population_columns]
        col_name_dict.update({col: l + col.replace(endpoint + "_", "") for col in endpoint_cols})
             
        hia_df.rename(columns=col_name_dict, inplace=True)
        hia_df = rename_for_shapefile(hia_df, endpoint)
          
        hia_df.to_file(fpath)
        logging.info('- {} Excess {} from {} exposure output as a shapefile as {}'.format(logging_code, endpoint_full.lower(), pollutant, fname))
        
        return fname

def export_health_impacts_csv(hia_df, population_columns, endpoint, output_dir, f_out, verbose, debug_mode, pollutant = 'PM2.5'):
        ''' 
        Exports total mortality as a csv file. 
        
        INPUTS:
            - hia_df: a dataframe containing excess mortality for the `endpoint` using the 
              `function` provided
            - population_columns: a list of population columns to use from the population input file
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

        # Create file name depening on pollutant
        if pollutant == 'DPM':
          endpoint_full = endpoint + " incidence"
          file_ending1 = '_excess_incidence.csv'
          file_ending2 = '_excess_incidence_summary.csv'
        else:
          endpoint_full = endpoint + " mortality"
          file_ending1 = '_excess_mortality.csv'
          file_ending2 = '_excess_mortality_summary.csv'

        verboseprint(verbose, '- {} Exporting excess {} from {} exposure as a CSV file.'.format(logging_code, endpoint_full.lower(), pollutant), debug_mode, frameinfo=getframeinfo(currentframe()))
            
        # Create the output file directory and name string
        fname = f_out + '_' + endpoint + file_ending1
        fname = str.lower(fname)
        fpath = os.path.join(output_dir, fname)
        logging_code = create_logging_code()[endpoint]
        
        # Create the output file directory and name string for the summary file
        summary_fname = f_out + '_' + endpoint + file_ending2
        summary_fname = str.lower(summary_fname)
        summary_fpath = os.path.join(output_dir, summary_fname)

        # Get endpoint shortlabel
        endpoint_nice = endpoint.title()
        endpoint_labels = {'ALL CAUSE':'ACM_',
                          'ISCHEMIC HEART DISEASE':'IHD_',
                          'LUNG CANCER':'CAN_',
                          'HAZARD QUOTIENT': 'HQ_',
                          "CANCER RISK":"CR_"}
        l = endpoint_labels[endpoint]
        
        # Create the summary HIA
        hia_summary = create_summary_hia(population_columns, hia_df, endpoint, verbose, l, endpoint_nice, debug_mode, pollutant)
        
        ## Update column names
        # Create the rename dictionary and make a few edits
        pop_rename_dict = {'POP_'+k: k + ' (# People)' for k in population_columns} # Add units to population
        hia_rename_dict = {l+'_'+k:endpoint_nice+' - '+k+' (excess deaths)' for k in population_columns}

        # Rename the columns in series
        hia_df.rename(columns=pop_rename_dict, inplace=True)
        hia_df.rename(columns=hia_rename_dict, inplace=True)
        hia_df.rename(columns={'TOTAL_CONC_UG/M3':'{} Concentration (ug/m3)'.format(pollutant), "RISK":"Cancer Risk"}, inplace=True)
        
        # Get rid of geometry column
        hia_df.drop(['geometry'], axis=1)
        
        # Export
        hia_df.to_csv(fpath, index=False)
        hia_summary.to_csv(summary_fpath, index=False)
        logging.info('- {} Excess {} from {} exposure output as a CSV as {}'.format(logging_code, endpoint_full.lower(), pollutant, fname))
        
        return hia_summary

def create_summary_hia(population_columns, hia_df, endpoint, verbose, l, endpoint_nice, debug_mode, pollutant = 'PM2.5'):
        ''' 
        Creates a summary table of health impacts by racial/ethnic group 
        
        INPUTS:
            - population_columns: a list of population columns to use from the population input file
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

        #Depending on pollutant, it is either excess incidence or excess mortality
        if pollutant == 'DPM':
          endpoint_full = endpoint + " incidence"
          rename1 = ' Incidence (# Excess Cases)'
          rename2 = 'Incidence'
        else:
          endpoint_full = endpoint + " mortality"
          rename1 = ' Mortality (# Excess Deaths)'
          rename2 = 'Mortality'
        verboseprint(verbose, '- {} Creating a summary table of {} from {} exposure.'.format(logging_code, endpoint_full.lower(), pollutant), debug_mode, frameinfo=getframeinfo(currentframe()))

        # Set up a few useful variables
        groups = population_columns
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
        exm_df.rename(columns={'index':'Label',0:endpoint_nice+rename1}, inplace=True)
        exm_df['Group'] = exm_df['Label'].str.split('_').str[1]
        exm_df = exm_df[['Group', endpoint_nice+rename1]].copy()
        
        # Combine into a summary table (do not export as CSV yet)
        hia_summary = pd.merge(pop_df, exm_df, on='Group')
        hia_summary[rename2 + ' Rate (per 100000)'] = hia_summary[endpoint_nice+rename1]/hia_summary['Population (# People)'] * 100000.0
        
        return hia_summary

def visualize_and_export_hia(hia_df, ca_shp_fp, population_columns, group, endpoint, output_dir, f_out, shape_out, output_resolution, output_png_flag, boundary, verbose, debug_mode, dpm = False):
        ''' 
        Automates this process a bit.
        
        INPUTS:
            - hia_df: a dataframe containing excess mortality for the `endpoint` using the 
              `function` provided
            - ca_shp_fp: a filepath string of the California state boundary shapefile
            - population_columns: a list of population columns to use from the population input file
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

        if dpm:
          pollutant = 'DPM'
          endpoint_full = endpoint + " incidence"
        else:
          endpoint_full = endpoint + " mortality"
          pollutant = 'PM2.5'
        
        logging.info('- {} Visualizing and exporting excess {}.'.format(logging_code, endpoint_full.lower()))

        # Plot the map of mortality
        if output_png_flag:
          fname = plot_total_mortality(hia_df, ca_shp_fp, group, endpoint, output_resolution, boundary, output_dir, f_out, verbose, debug_mode, pollutant)
        
        # Export the shapefile
        fname = export_health_impacts(hia_df, population_columns, group, endpoint, shape_out, f_out, verbose, debug_mode, pollutant)
        hia_summary = export_health_impacts_csv(hia_df, population_columns, endpoint, output_dir, f_out, verbose, debug_mode, pollutant)
            
        return hia_summary

def combine_hia_summaries(acm_summary, ihd_summary, lcm_summary, output_dir, f_out, verbose, dpm_summary=None):
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
        hia_summary = pd.merge(acm_summary, ihd_summary.drop(columns=['Population (# People)']), on='Group')


        hia_summary = pd.merge(hia_summary, lcm_summary.drop(columns=['Population (# People)']), on='Group')

        #Add DPM 
        if dpm_summary is not None:
            hia_summary = pd.merge(hia_summary, dpm_summary.drop(columns=['Population (# People)']), on='Group')
        
        # Keep only necessary columns
        hia_summary = hia_summary.loc[:, ~hia_summary.columns.str.startswith(('Mortality','Incidence'))]
        
        
        # Export results
        fname = f_out + '_excess_mortality_summary.csv'
        fname = str.lower(fname)
        fpath = os.path.join(output_dir, fname)
        hia_summary.to_csv(fpath, index=False)
        
        return

def rename_for_shapefile(df, endpoint, max_len=10):
    """
    Truncate column names to <= 10 chars.
    If duplicates occur, rename duplicates to POP_01, POP_02, ...
    """

    new_names = {}
    used = set()
    pop_counter = 1

    for col in df.columns:
        # Truncate to max_len
        if len(col) > max_len:    
          truncated = col[:max_len]
          # If this truncated name is unique, keep it
          if truncated not in used:
            new_names[col] = truncated
            used.add(truncated)
          else:
            # Duplicate → use POP_XX
            replacement = f"POP_{pop_counter:02d}"
            new_names[col] = replacement
            used.add(replacement)
            pop_counter += 1
    
    logging_code = create_logging_code()[endpoint]
    changes = ", ".join([f"{old}→{new}" for old, new in new_names.items()])
    logging.info("  - {} Columns too long for shapefile renamed: {}".format(logging_code, changes))

    # Apply renaming
    return df.rename(columns=new_names)
  
def hazard_quotient(conc, output_dir, f_out,):
    '''
    Calculates the Hazard Quotient (HQ) for each ISRM grid cell

    INPUTS:
      - conc: a vector with the dmp concentration for each grid cell
    
    OUTPUTS:
      - df_hq: a dataframe of HQs per grid cell
    '''

    # Divided DPM concentration by 5
    df_hq = conc.copy()
    df_hq["HQ"] = df_hq["DPM_CONC_UG/M3"]/5

    # Created file name for output
    fname = f_out + '_dpm_hazard_quotient.csv'
    fname = str.lower(fname)
    fpath = os.path.join(output_dir, fname)

    #Outputs file
    hq_output = df_hq[['ISRM_ID', 'DPM_CONC_UG/M3', 'HQ']]
    hq_output.to_csv(fpath, index=False)

    return df_hq

def dpm_risk(conc, output_dir, f_out, avg_time='30YR'): 
  ''' 
  Estimates excess cancer risk from DPM following OEHHA (2015) methodology 
    
  INPUTS:
      - conc: a vector with the concentration for each grid cell
      - avg_time: a string identifying if the user wants to run 30-year averaging time or 70-year
        
  OUTPUTS: 
      - dpm_risk: a vector with the excess cancer risk from DPM (units: cancer risk per million people)
    
    '''
  
  #Define output file path
  fname = f_out + '_dpm_cancer_risk.csv'
  fname = str.lower(fname)
  fpath = os.path.join(output_dir, fname)
  
  dpm_risk = conc.copy()
  # Estimate the unit dose for each age group
  #                Exposure Frequency | BR/BW |  CF (ug-->mg, L--m3)
  # Units              (days/days)    (L/kg-day)
  # OEHHA 2015 pp        122/231	     122-123 
  unit_dose = np.array([ 0.96    *    361.00 * 1e-6,  # 3rd trimester
                          0.96    *    1090.0 * 1e-6,  # 0-2 years
                          0.96    *    861.00 * 1e-6,  # 2-9 years -- this is never used
                          0.96    *    745.00 * 1e-6,  # 2-16 years
                          0.96    *    335.00 * 1e-6,  # 16-30 years
                          0.96    *    290.00 * 1e-6]) # 16-70 years

  # Define the age sensitivity factor (ASF, unitless)
  asf = np.array([10, 10, 3, 3, 1, 1]) # OEHHA 2015 Table 8.3

  # Define the fraction of time at home (FAH, unitless)
  fah = np.array([0.85, 0.85, 0.72, 0.72, 0.73, 0.73]) # OEHHA 2015 Table 8.4

  # Define the duration adjustment based on a 70 year life span regardless of averaging time
  dur_adj = np.array([0.25, 2, 7, 14, 14, 54]) / 70.

  # Define the cancer sensitivity factor
  csf = 1.1 # OEHHA 2015 Table 7.1

  # Choose the relevant age bins based on the avg_time parameter
  if avg_time == '30YR':
      unit_risk = unit_dose[[0,1,3,4]] * asf[[0,1,3,4]] * fah[[0,1,3,4]] * dur_adj[[0,1,3,4]] * csf
  if avg_time == '70YR':
      unit_risk = unit_dose[[0,1,3,5]] * asf[[0,1,3,5]] * fah[[0,1,3,5]] * dur_adj[[0,1,3,5]] * csf
    
  # Scale by concentrations and 10^6 to get risk per million people
  dpm_risk["RISK"] = unit_risk.sum() * conc["DPM_CONC_UG/M3"] * 1e6

  #Re-order column names
  dpm_risk_output = dpm_risk[["ISRM_ID","DPM_CONC_UG/M3","RISK"]]
  dpm_risk_output.to_csv(fpath, index=False)
    
  return dpm_risk

def dpm_excess_incidence(risk,population):
    '''
    For each given risk and population count, calculates the excess cancer incidence

    INPUTS:
        - risk: numerical risk value
        - population: population count

    OUTPUTS:
        - excess_incidence: excess number of cancer cases
    '''
    excess_incidence = risk*population/(10**(6))
    return excess_incidence


    
         


     
     

