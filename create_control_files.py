#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Control File Generator

@author: libbykoolik
Last updated: 2023-09-11
"""

import pandas as pd
from os import path
import os
import argparse
import sys

#% Use argparse to parse command line arguments
# Initialize the parser object
parser = argparse.ArgumentParser(description="Creates control files for ECHO-AIR.")

# Add necessary arguments
parser.add_argument("-i", "--input", help="path to the control file creator CSV", type=str)
parser.add_argument("-o", "--output", help="filepath for saving outputs", type=str)
parser.add_argument("-p", "--parallel", help="adds the parallel tag to the batch file", action="store_true")

# Parse all arguments
args = parser.parse_args()
fp = args.input
output_dir = args.output
parallel = args.parallel

#%% Run Program
if __name__ == "__main__":        
    
    # Return a print statement to show it is working
    ## Logging statements aren't necessary for this module
    print('\n')
    print('╓────────────────────────────────╖')
    print('║ Creating Batch Control Files   ║')
    print('╙────────────────────────────────╜')
    print('\n')

    #%% First, get the template control file
    template_fp = path.join(os.getcwd(),'templates', 'control_file_template.txt')
    
    #%% Check that the file exists
    if not path.exists(fp):
        print('\n<< ERROR: The path to the control file input csv does not exist. Please correct and try again. >>')
        sys.exit()
    
    #%% Read and re-format the CSV file
    # Read in the data
    df = pd.read_csv(fp)
    
    # Replace the nans
    df = df.fillna('')
    
    # Re-index for lookup capabilities
    df = df.set_index('Column')
    
    # Store all the names
    new_file_paths = []
    
    # Get the number of control files to create
    n = len(df.columns)
    print('<< Generating {} control files. >>'.format(n))
    
    #%% Iterate through columns to create control files
    for i in range(n):
        # Grab one column
        tmp = df.iloc[:,i]
        
        # Get the run name, batch name
        name = tmp['BATCH_NAME'] + '_' + tmp['RUN_NAME']
        
        # Make a print statement
        print('{}. {}...'.format(i+1, name), end='')
        
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
                        
        # Return a message indicating this one is done
        print('done!', end='\n')

    #%% Finally, spit out a new text file with all of the ECHO-AIR calls
    print('\n<< Writing a batch file for running all >>')
    
    # Add the parallel tag, if appropriate
    if parallel:
        parallel_tag = '--parallel'
    else:
        parallel_tag = ''
    
    # Create and open a new text file
    with open(path.join(output_dir, 'ECHO_AIR_BATCH_TEXT.txt'),'w') as batch_file:
        
        # Iterate through names
        for nfp in new_file_paths:
            batch_file.write("python3 run_echo_air.py -i '{}' {}\n".format(nfp, parallel_tag))
    print('- Stored at: {}'.format(nfp))
    
    #%% One final print message
    print('\n')
    print('╓────────────────────────────────╖')
    print('║ Success! Script complete.      ║')
    print('╙────────────────────────────────╜\n')
    print('<< ECHO-AIR has created and exported all control files indicated. >>')