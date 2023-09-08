#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Control File Generator

@author: libbykoolik
Last updated: 2023-09-08
"""

import pandas as pd
from os import path
import os
import argparse

#% Use argparse to parse command line arguments
# Initialize the parser object
parser = argparse.ArgumentParser(description="Creates control files for ECHO-AIR.")

# Add necessary arguments
parser.add_argument("-i", "--input", help="path to the control file creator CSV", type=str)
parser.add_argument("-o", "--output", help="filepath for saving outputs", type=str)

# Parse all arguments
args = parser.parse_args()
fp = args.input
output_dir = args.output

#%% Run Program
if __name__ == "__main__":        

    #%% First, get the template control file
    template_fp = path.join(os.getcwd(),'templates', 'control_file_template.txt')
    
    #%% Read and re-format the CSV file
    # Read in the data
    df = pd.read_csv(fp)
    
    # Replace the nans
    df = df.fillna('')
    
    # Re-index for lookup capabilities
    df = df.set_index('Column')
    
    # Store all the names
    new_file_paths = []
    
    #%% Iterate through columns to create control files
    for i in range(len(df.columns)):
        # Grab one column
        tmp = df.iloc[:,i]
        
        # Get the run name, batch name
        name = tmp['BATCH_NAME'] + '_' + tmp['RUN_NAME']
        
        # Create the destination file
        new_file_path = path.join(output_dir, name+'.txt')
        
        # Store for later
        new_file_paths.append(new_file_path)
        
        ## Go through and update values
        with open(new_file_path, 'w') as new_file:
            with open(template_fp, 'r') as source_file:
                
                # Go line-by-line
                for line in source_file:
                    
                    # Break the line into items
                    line_list = line.split(':')
                    
                    # Get just the column name
                    column = line_list[0].split('- ')[-1]
                    template_stuff = line_list[-1]
                    
                    # If the column name matches the CSV file, we will make an edit
                    if (len(line_list)>1) and (column in df.index):
                        new_file.write(line.replace(template_stuff, ' '+tmp.loc[column]+'\n'))
                        
                    else:
                        new_file.write(line)

    #%% Finally, spit out a new text file with all of the ECHO-AIR calls
    # Create and open a new text file
    with open(path.join(output_dir, 'ECHO_AIR_BATCH_TEXT.txt'),'w') as batch_file:
        
        # Iterate through names
        for nfp in new_file_paths:
            batch_file.write("python3 run_echo_air.py -i '{}'\n".format(nfp))
            