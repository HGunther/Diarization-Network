# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 19:52:06 2017

@author: sam
"""
from "tylersnn.py" import neuralnet as nn
import numpy as np
import csv
import sys
from os.path import basename
from utils import Smoother
from utils import format_csv_line_as_list
from utils import ms_to_seconds_as_string
from ProductionChunks import ProductionChunks
 # Get wave filename from the first command line argument
wave_filename = sys.argv[1]


 # Constants
CHUNK_LENGTH_IN_MS = 250
 


current_time_in_ms = 0

 # Set up post-net processing to write to csv files

result = nn.evaluate(sequential_chunks[0])

speaker1_is_talking_last_chunk = round(speaker1_smoother.get_smoothed_datapoint(result[0]))
speaker2_is_talking_last_chunk = round(speaker2_smoother.get_smoothed_datapoint(result[1]))

speaker1_smoother = Smoother()
speaker2_smoother = Smoother()

speaker1_segment_start_time = 0
speaker2_segment_start_time = 0

 # Open the two CSV files

speaker1_file = basename(wave_filename)+"_Spk1.csv"
speaker2_file = basename(wave_filename)+"_Spk2.csv"

with open (speaker1_file, 'w') as file1, open (speaker2_file, 'w') as file2:
    
    csv1 = csv.writer(file1)
    csv2 = csv.writer(file2)
    
    csv_header = ["tmi0","tmax","text","tier"]
    csv1.writerow(csv_header)
    csv2.writerow(csv_header)
    
    chunks = ProductionChunks(wave_filename, CHUNK_LENGTH_IN_MS)
    
     # For each sequential chunk
    for chunk_index in range(chunks.get_chunk_count()):
        
        chunk = chunks.get_chunk(chunk_index)
        # Preactions
        current_time_in_ms += CHUNK_LENGTH_IN_MS
        
        # Evaluate chunk on NN
        result = nn.evaluate(chunk)
        
        # Smooth chunks so timestamps are less fragmented
        
        speaker1_is_talking = round(speaker1_smoother.get_smoothed_datapoint(result[0]))
        speaker2_is_talking = round(speaker2_smoother.get_smoothed_datapoint(result[1]))
        
        
        # Actions: Write to CSV file and update segment time if needed
        if speaker1_is_talking != speaker1_is_talking_last_chunk:
            csv_line = format_csv_line_as_list(speaker1_segment_start_time, current_time_in_ms, speaker1_is_talking, 1)
            csv1.writerow(csv_line)
            speaker1_segment_start_time = current_time_in_ms
            
        if speaker2_is_talking != speaker2_is_talking_last_chunk:
            csv_line = format_csv_line_as_list(speaker2_segment_start_time, current_time_in_ms, speaker2_is_talking, 2)
            csv2.writerow(csv_line)
            speaker2_segment_start_time = current_time_in_ms
        
        # Postactions
        speaker1_is_talking_last_chunk = speaker1_is_talking
        speaker2_is_talking_last_chunk = speaker2_is_talking

# Files are now written




    
        
        