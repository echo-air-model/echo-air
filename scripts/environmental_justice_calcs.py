  #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EJ Functions

@author: libbykoolik
last modified: 2025-06-05
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
from matplotlib_scalebar.scalebar import ScaleBar

#%%
def create_exposure_df(conc, isrm_pop_alloc, population_columns, verbose, debug_mode, dpm, nox_conc):
        ''' 
        Create an exposure geodataframe from concentration and population.
        
        INPUTS:
            - conc: concentration object
            - isrm_pop_alloc: population object re-allocated to the ISRM grid cell 
              geometry
            - population_columns: a list of population columns to use from the population input file
            - verbose: a Boolean indicating whether or not detailed logging statements 
              should be printed
            - debug_mode: a Boolean indicating whether or not to output debug statements
              
        OUTPUTS:
            - exposure_gdf: a geodataframe with the exposure concentrations and allocated 
              population by racial group
        
        '''
        # Pull the total concentration from the conc object
        population_columns = population_columns
        conc_gdf = conc.total_conc.copy()
        relevant_columns = ['ISRM_ID', 'geometry', 'PM25_UG_M3']

        if dpm:
             relevant_columns.append('DPM_UG_M3')
        if nox_conc:
             relevant_columns.append('NOX_CONC_PPB')

        conc_gdf.columns = relevant_columns

        if not isinstance(conc_gdf, gpd.GeoDataFrame):
          conc_gdf = gpd.GeoDataFrame(conc_gdf, geometry="geometry", crs=conc.crs)
        
        # Pull only relevant columns from isrm_pop_alloc
        isrm_pop_alloc = isrm_pop_alloc[['ISRM_ID']+ population_columns].copy()
        
        # Merge concentration and population data based on ISRM_ID
        exposure_gdf = pd.merge(conc_gdf, isrm_pop_alloc, left_on='ISRM_ID', right_on='ISRM_ID')
        
        # Get PWM columns per group
        verboseprint(verbose, '- [EJ] Estimating population weighted mean exposure for each demographic group.', 
                    debug_mode, frameinfo=getframeinfo(currentframe()))
        for group in population_columns:
            exposure_gdf = add_pwm_col(exposure_gdf, group, dpm, nox_conc)
            
        return exposure_gdf



def add_pwm_col(exposure_gdf, group, dpm, nox_conc):
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

        if dpm:
             dpm_pwm_col = group+'_DPM_PWM'
             exposure_gdf[dpm_pwm_col] = exposure_gdf[group]*exposure_gdf['DPM_UG_M3']
        if nox_conc:
              nox_conc_pwm_col = group+'_NOX_CONC_PWM'
              exposure_gdf[nox_conc_pwm_col] = exposure_gdf[group]*exposure_gdf['NOX_CONC_PPB']
        
        # Create a column for each ISRM cell that is the group total exposure
        exposure_gdf[pwm_col] = exposure_gdf[group]*exposure_gdf['PM25_UG_M3']
        
        return exposure_gdf

def get_pwm(exposure_gdf, group, dpm, nox_conc):
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

        if dpm:
            dpm_pwm_col = group+'_DPM_PWM'
            DPM_PWM_group = exposure_gdf[dpm_pwm_col].sum()/exposure_gdf[group].sum()
        else:
            DPM_PWM_group = 0

        if nox_conc:
            nox_conc_pwm_col = group+'_NOX_CONC_PWM'
            NOX_CONC_PWM_group = exposure_gdf[nox_conc_pwm_col].sum()/exposure_gdf[group].sum()
        else:
             NOX_CONC_PWM_group = 0
             
             
        
        return PWM_group, DPM_PWM_group, NOX_CONC_PWM_group

def get_overall_disparity(exposure_gdf, population_columns, dpm, nox_conc):
        ''' 
        Returns a table of overall disparity metrics 
        
        INPUTS:
            - exposure_gdf: a geodataframe with the exposure concentrations and allocated population 
              by racial group
            - population_columns: a list of population columns to use from the population input file
            
        OUTPUTS: 
            - pwm_df: a dataframe containing the PWM, absolute disparity, and relative disparity
              of each group
        
        '''


        # Create a dataframe to store information
        pwm_df = pd.DataFrame({'Group':population_columns}, columns=['Group'])
        
        # Use predefined function to get the group PWMs
        pwm_df['Group PWM (PM2.5)'] = pwm_df.apply(lambda x: get_pwm(exposure_gdf, x['Group'], dpm, nox_conc)[0], axis=1)
        
        # Calculate Absolute and Relative Disparities 
        pwm_df['Absolute Disparity (PM2.5)'] = pwm_df['Group PWM (PM2.5)'] - pwm_df.loc[0,'Group PWM (PM2.5)']
        pwm_df['Relative Disparity (PM2.5)'] = pwm_df['Absolute Disparity (PM2.5)']/pwm_df.loc[0,'Group PWM (PM2.5)']

        if dpm:
            pwm_df['Group PWM (DPM)'] = pwm_df.apply(lambda x: get_pwm(exposure_gdf, x['Group'], dpm, nox_conc)[1], axis=1)

            pwm_df['Absolute Disparity (DPM)'] = pwm_df['Group PWM (DPM)'] - pwm_df.loc[0,'Group PWM (DPM)']
            pwm_df['Relative Disparity (DPM)'] = pwm_df['Absolute Disparity (DPM)']/pwm_df.loc[0,'Group PWM (DPM)']

        if nox_conc:
            pwm_df['Group PWM (NOX_CONC)'] = pwm_df.apply(lambda x: get_pwm(exposure_gdf, x['Group'], dpm, nox_conc)[2], axis=1)

            pwm_df['Absolute Disparity (NOX_CONC)'] = pwm_df['Group PWM (NOX_CONC)'] - pwm_df.loc[0,'Group PWM (NOX_CONC)']
            pwm_df['Relative Disparity (NOX_CONC)'] = pwm_df['Absolute Disparity (NOX_CONC)']/pwm_df.loc[0,'Group PWM (NOX_CONC)']
             
        return pwm_df

def estimate_exposure_percentile(exposure_gdf, population_columns, verbose, dpm, nox_conc):
        ''' 
        Creates a dataframe of percentiles
        
        INPUTS:
            - exposure_gdf: a geodataframe with the exposure concentrations and allocated population 
              by racial group
            - population_columns: a list of population columns to use from the population input file
            - verbose: a Boolean indicating whether or not detailed logging statements should be printed
            
        OUTPUTS:
            - df_pctl: a dataframe of exposure concentrations by percentile of population exposed 
              by group
        
        '''
        if verbose:
            logging.info('- Estimating the exposure level for each percentile of each demographic group population.')
        pollutant_map = {'PM25_UG_M3': 'PM25_UG_M3'}
        if dpm:
          pollutant_map['DPM_UG_M3'] = 'DPM_UG_M3'
        if nox_conc:
          pollutant_map['NOX_CONC_PPB'] = 'NOX_CONC_PPB'
                  
        # Create a copy to avoid overwriting, then sort based on PM25 concentration
        df_pctl = exposure_gdf.copy()

        for pollutant_label, col_name in pollutant_map.items():
            df_pctl.sort_values(by=col_name, inplace=True)
            df_pctl.reset_index(drop=True, inplace=True)
            for group in population_columns:
              # Calculate cumulative sum for this specific pollutant sorting
              cum_sum_col = df_pctl[group].cumsum()
              total_pop = df_pctl[group].sum()
              # Create a unique column name, e.g., 'Percentile_White_NOX_PPB'
              new_col_name = f'Percentile_{group}_{pollutant_label}'
              df_pctl[new_col_name] = cum_sum_col / total_pop
              df_pctl.sort_values(by='PM25_UG_M3', inplace=True)
              df_pctl.reset_index(drop=True, inplace=True)
        
        # # Iterate through each group to estimate the percentile of exposure
        # for group in population_columns:
        #     # Create a slice of the percentile dataframe
        #     df_slice = df_pctl[['PM25_UG_M3',group]].copy()
            
        #     # Add the cumulative sum of the population
        #     df_slice.loc[:,'Cumulative_Sum_Pop'] = df_slice.loc[:, group].cumsum()
            
        #     # Estimate the total population in that group, then divide the cumulative sum
        #     # to get the percentile
        #     total_pop_group = df_slice[group].sum()
        #     df_slice.loc[:, 'Percentile_'+group] = df_slice['Cumulative_Sum_Pop']/total_pop_group
            
        #     # Add the Percentile column into the main percentile dataframe
        #     df_pctl.loc[:, group] = df_slice.loc[:, 'Percentile_'+group]
        
        return df_pctl

def run_exposure_calcs(conc, pop_alloc, population_columns, verbose, debug_mode, dpm = True, nox_conc = True):
        ''' 
        Run the exposure EJ calculations from one script 
        
        INPUTS:
            - conc: concentration object from `concentration.py`
            - isrm_pop_alloc: population object (from `population.py`) re-allocated to the 
              ISRM grid cell geometry
            - population_columns: a list of population columns to use from the population input file
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
        exposure_gdf = create_exposure_df(conc, pop_alloc, population_columns, verbose, debug_mode, dpm, nox_conc)
        exposure_disparity = get_overall_disparity(exposure_gdf, population_columns, dpm, nox_conc)
        exposure_pctl = estimate_exposure_percentile(exposure_gdf, population_columns, verbose, dpm, nox_conc)
        
        return exposure_gdf, exposure_pctl, exposure_disparity 

def export_exposure_gdf(population_columns, exposure_gdf, shape_out, f_out, dpm, nox_conc):
        ''' 
        Exports the exposure_gdf dataframe as a shapefile 
        
        INPUTS:
            - population_columns: a list of population columns to use from the population input file
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
        
        # Define relevant concentration columns
        conc_columns = ['ISRM_ID', 'PM25_UG_M3']
        if dpm:
          conc_columns.append('DPM_UG_M3')
        if nox_conc:
             conc_columns.append('NOX_CONC_PPB')

        # Update the columns slightly
        exposure_gdf = exposure_gdf[conc_columns + population_columns + ['geometry']].copy()
        exposure_gdf = rename_for_shapefile(exposure_gdf)

        # Export to file
        exposure_gdf.to_file(fpath)
        logging.info('   - [EJ] Exposure concentrations output as {}'.format(fname))

        return fname #placeholder for parallelization

def export_exposure_csv(population_columns, exposure_gdf, output_dir, f_out, dpm, nox_conc):
        ''' 
        Exports the exposure_gdf dataframe as a CSV file 
        
        INPUTS:
            - population_columns: a list of population columns to use from the population input file
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

        # Define relevant concentration columns
        conc_columns = ['ISRM_ID', 'PM25_UG_M3']
        if dpm:
          conc_columns.append('DPM_UG_M3')
        if nox_conc:
             conc_columns.append('NOX_CONC_PPB')

        
        # Update the columns slightly
        exposure_gdf = exposure_gdf[conc_columns + population_columns + ['geometry']].copy()

        # Change column names
        rename_dict = {k : k + ' (# People)' for k in population_columns}
        exposure_gdf.rename(columns=rename_dict, inplace=True)
        exposure_gdf.rename(columns={'PM25_UG_M3':'PM2.5 Concentration (ug/m3)'}, inplace=True)
        
        if dpm:
            exposure_gdf.rename(columns={'DPM_UG_M3':'DPM Concentration (ug/m3)'}, inplace=True)
        if nox_conc:
            exposure_gdf.rename(columns={'NOX_CONC_PPB':'NOx Concentration (ppb)'}, inplace=True)

        # Export to file
        exposure_gdf.to_csv(fpath, index=False)
        logging.info('   - [EJ] Exposure concentrations output as {}'.format(fname))

        return fname #placeholder for parallelization


def export_exposure_disparity(exposure_disparity, output_dir, f_out, dpm, nox_conc):
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
        
        pollutant_suffixes = ['PM2.5']
        if dpm:
          pollutant_suffixes.append('DPM')
        if nox_conc:
          pollutant_suffixes.append('NOX_CONC')

        for p in pollutant_suffixes:
          # Construct the expected input column names
          # Assuming the format is 'Column Name (Pollutant)'
          rel_col = f'Relative Disparity ({p})'
          pwm_col = f'Group PWM ({p})'
          abs_col = f'Absolute Disparity ({p})'

          # Update values slightly
          exposure_disparity[rel_col] = exposure_disparity[rel_col] * 100.0

          # Rename column names
          if p == 'NOX_CONC':
            exposure_disparity.rename(columns={
            pwm_col: 'NOx Group PWM (ppb)',
            abs_col: 'NOx Absolute Disparity (ppb)',
            rel_col: 'NOx Relative Disparity (%)'
            }, inplace=True)

          else:
            exposure_disparity.rename(columns={
              pwm_col: f'{p} Group PWM (ug/m3)',
              abs_col: f'{p} Absolute Disparity (ug/m3)',
              rel_col: f'{p} Relative Disparity (%)'
              }, inplace=True)

        # Export to file
        exposure_disparity.to_csv(fpath, index=False)
        logging.info('   - [EJ] Exposure concentrations output as {}'.format(fname))

        return fname

def plot_percentile_exposure(population_columns, output_dir, f_out, exposure_pctl, verbose, debug_mode, dpm, nox_conc):
        ''' 
        Creates a percentile plot by group 
        
        INPUTS:
            - population_columns: a list of population columns to use from the population input file
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
        
        # Melt the dataframe for easier use of seaborn
        pctl_melt = pd.melt(exposure_pctl, id_vars='PM25_UG_M3',
                            value_vars=population_columns,var_name='Racial/Ethnic Group', 
                            value_name='Percentile')
        pctl_melt['Racial/Ethnic Group'] = pctl_melt['Racial/Ethnic Group'].str.title()
        
        # Adjust formatting for a prettier plot
        pctl_melt['Percentile'] = pctl_melt['Percentile']*100
        sns.set_theme(context="notebook", style="whitegrid", font_scale=1.75)

        # Initialize the figure
        fig, ax = plt.subplots(figsize=(10,8))
        sns.lineplot(data=pctl_melt, x='Percentile', y='PM25_UG_M3', hue='Racial/Ethnic Group', ci=None, 
                    linewidth=3, palette='deep', ax=ax)
        ax.set(ylabel=r'PM$_{2.5}$ Exposure ($\mu$g/m$^3$)')
        ax.set_xticks(ticks=[5,25,50,75,95], 
                      labels=['5th','25th','50th','75th','95th'])
        
        # Save the file
        fname =f_out+'_PM25_Exposure_Percentiles.png' # File Name
        fpath = os.path.join(output_dir, fname)
        fig.savefig(fpath, dpi=200)
        logging.info('- [EJ] Exposure concentration by percentile figure output as {}'.format(fname))
        
        return fname

def export_exposure(population_columns, exposure_gdf, exposure_disparity, exposure_pctl, shape_out, output_dir, f_out, verbose, run_parallel, output_png_flag, dpm, nox_conc, debug_mode):
        ''' 
        Calls each of the exposure output functions in parallel
        
        INPUTS:
            - population_columns: a list of population columns to use from the population input file
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
                gdf_export_future = ej_executor.submit(export_exposure_gdf, population_columns, exposure_gdf, shape_out, f_out, dpm, nox_conc)
                csv_export_future = ej_executor.submit(export_exposure_csv, population_columns, exposure_gdf, output_dir, f_out, dpm, nox_conc)
                disp_export_future = ej_executor.submit(export_exposure_disparity, exposure_disparity, output_dir, f_out, dpm, nox_conc)
                if output_png_flag:
                  plot_export_future = ej_executor.submit(plot_percentile_exposure, population_columns, output_dir, f_out, exposure_pctl, verbose, 
                                                        debug_mode, dpm, nox_conc)
                
                # Wait for all to finish
                if output_png_flag: 
                  (tmp, tmp, tmp, tmp) = (gdf_export_future.result(), csv_export_future.result(),
                                          disp_export_future.result(), plot_export_future.result())
                else:
                  (tmp, tmp, tmp) = (gdf_export_future.result(), csv_export_future.result(),
                                        disp_export_future.result())
        else:
            # Call export functions linearly
            export_exposure_gdf(population_columns, exposure_gdf, shape_out, f_out, dpm, nox_conc)
            export_exposure_csv(population_columns, exposure_gdf, output_dir, f_out, dpm, nox_conc)
            export_exposure_disparity(exposure_disparity, output_dir, f_out, dpm, nox_conc)
            if output_png_flag:
              plot_percentile_exposure(population_columns, output_dir, f_out, exposure_pctl, verbose,
                                    debug_mode, dpm, nox_conc)
        
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
        den = tmp[group].sum()

        #If denominator is 0, assign NaN
        if den == 0 or pd.isna(den):
          pwm = np.nan
        else:
          pwm = (tmp[group] * tmp['TOTAL_CONC_UG/M3']).sum() / den
        return pwm

def export_pwm_map(population_columns, pop_exp, conc, output_dir, output_region, output_png_flag, f_out, ca_shp_path, shape_out):
        ''' 
        Creates the exports for the population-weighted products requested when the 
        user inputs an output resolution larger than the ISRM grid. In this step, 
        dropping geometry occurs before aggregation and then is reused as needed.
        
        INPUTS:
            - population_columns: a list of population columns to use from the population input file
            - pop_exp: a dataframe containing the population information without age-resolution
            - conc: a concentration object (which contains the crosswalk with geometry)
            - output_dir: a filepath string of the location of the output directory
            - output_region: the geometry of the desired output region
            - f_out: the name of the file output category (will append additional information)
            - ca_shp_path: a filepath string of the location of the California boundary shapefile
            - shape_out: a filepath string of the location of the shapefile output directory
            
        OUTPUTS:
            - output_res_geo: a GeoDataFrame with the aggregated population-weighted means.
        '''
        # Log statement
        logging.info('- [EJ] Creating population-weighted mean summaries at the output resolution requested.')
        
        # Collect the necessary objects:
        # crosswalk: from the concentration object (with geometry)
        crosswalk = conc.crosswalk[['NAME', 'ISRM_ID', 'TOTAL_CONC_UG/M3', 'geometry']].copy()
        # Population data from pop_exp (with geometry)
        pop_exp = pop_exp[['POP_ID'] + population_columns + ['geometry']].copy()
        
        # Project population data to the same CRS as crosswalk
        pop_exp = pop_exp.to_crs(crosswalk.crs)
        
        # Create an intersection object (union) between pop_exp and crosswalk
        intersect = gpd.overlay(pop_exp, crosswalk, how='union', keep_geom_type=False)
        
        # Remove null matches
        intersect = intersect[(~intersect['POP_ID'].isna()) & (~intersect['ISRM_ID'].isna())]
        
        # Estimate area (in km²)
        intersect['AREA_M2'] = intersect.geometry.area / (1000.0 * 1000.0)
        
        # --- Drop geometry before aggregation ---
        numeric_intersect = intersect.drop(columns='geometry')
        
        # Aggregate total area by POP_ID
        pop_totalarea = (numeric_intersect.groupby('POP_ID', as_index=False)['AREA_M2']
                        .sum()
                        .set_index('POP_ID')['AREA_M2']
                        .to_dict())
        
        # Map total area back onto intersect and calculate area fraction
        intersect['AREA_POP_TOTAL'] = intersect['POP_ID'].map(pop_totalarea)
        intersect['AREA_FRAC'] = intersect['AREA_M2'] / intersect['AREA_POP_TOTAL']
        
        # Apportion population for each group using the area fraction
        for group in population_columns:
            intersect[group] = intersect[group] * intersect['AREA_FRAC']
        
        # Get the output resolution names and geometries by dissolving the crosswalk (geometry is preserved here)
        output_res_geo = crosswalk[['NAME', 'geometry']].dissolve(by='NAME').reset_index()
        
        # Estimate the population-weighted mean (PWM) per group using your helper function.
        # (This function will internally slice intersect for the given NAME.)
        for group in population_columns:
            output_res_geo[group + '_PWM'] = output_res_geo.apply(
                lambda x: region_pwm_helper(x['NAME'], group, intersect), axis=1)
        
        # Export the map of population-weighted concentrations.
        logging.info('- [EJ] Exporting map of population-weighted mean summaries at the output resolution requested.')
        if output_png_flag:
          visualize_pwm_conc(output_res_geo, output_region, output_dir, f_out, ca_shp_path)
        
        # Create a shapefile to output, using only the relevant columns.
        to_shp = output_res_geo[['NAME', 'TOTAL_PWM', 'geometry']].copy()
        to_shp.columns = ['NAME', 'PWM_UG_M3', 'geometry']
        to_shp.to_file(os.path.join(output_dir, 'shapes', f_out + '_pwm_concentration.shp'))
        
        # --- Aggregate population by region ---
        # Drop geometry from intersect before grouping.
        numeric_intersect = intersect.drop(columns='geometry')
        pop_by_name = numeric_intersect.groupby('NAME', as_index=False)[population_columns].sum()
        
        # Merge the aggregated population data with the output resolution GeoDataFrame.
        pwm_cols = [f"{col}_PWM" for col in population_columns]
        to_csv = pd.merge(pop_by_name, 
                          output_res_geo[['NAME'] + pwm_cols], 
                          on='NAME')
        to_csv.to_csv(os.path.join(output_dir, f_out + '_aggregated_exposure_concentrations.csv'), index=False)
        
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

        # Calculates the longitude and latitude of the center
        center_lon, center_lat = (minx + maxx) / 2, (miny + maxy) / 2
        
        # Add north arrow
        angle_to_north = calculate_true_north_angle(center_lon, center_lat, output_res_geo.crs)
        add_north_arrow(ax,float(angle_to_north))
        
        # Add scale bar
        scalebar = ScaleBar(1, location='lower left', border_pad=0.5)  # 1 pixel = 1 unit
        ax.add_artist(scalebar)
        
        ax.set_title(t_str)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.text(minx-(maxx-minx)*0.1, miny-(maxy-miny)*0.1, st_str, fontsize=12)
        
        fig.tight_layout()
        fig.savefig(fpath, dpi=200)
            
        return 

def rename_for_shapefile(df, max_len=10):
    """
    Truncate column names to <= 10 chars.
    If duplicates occur, rename duplicates to POP_01, POP_02, ...
    """

    new_names = {}
    used = set()
    pop_counter = 1

    df.rename(columns={"NOX_CONC_PPB" : "NOXC_PPB"}, inplace = True)

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
    changes = ", ".join([f"{old}→{new}" for old, new in new_names.items()])
    logging.info("   - [EJ] Columns too long for shapefile renamed: {}".format(changes))

    # Apply renaming
    return df.rename(columns=new_names)
