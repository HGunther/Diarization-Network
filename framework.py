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

 # Get wave filename from the first command line argument
wave_filename = sys.argv[1]


 # Constants
CHUNK_LENGTH_IN_MS = 250
 

 # Read in wav

 # Break wav into seq chunks


current_time_in_ms = 0

 # Set up post-net processing to write to csv files

result = nn.evaluate(sequential_chunks[0])

speaker1_is_talking_last_chunk = round(speaker1_smoother.get_smoothed_datapoint(result[0]))
speaker2_is_talking_last_chunk = round(speaker2_smoother.get_smoothed_datapoint(result[1]))

speaker1_smoother = smoother()
speaker2_smoother = smoother()

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
    
     # For each sequential chunk
    for chunk in sequential_chunks:
        
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



####
# HELPER FUNCTIONS AND CLASSES
####

def ms_to_seconds_as_string(time_in_ms_float):
    time_in_sec = time_in_ms_float/1000.0
    return "{:10.4f}".format(time_in_sec)

def format_csv_line_as_list(start_time_in_ms, end_time_in_ms, is_speaking, speaker_id):
    return [ms_to_seconds_str(start_time_in_ms), ms_to_seconds_str(end_time_in_ms), str(int(is_speaking)), str(int(speaker_id))]


class smoother:
    def __init__(self, smoothness=10):
        self.smoothness = smoothness
        self.data = []
        
    def __add_datapoint(self, next_raw_data_point):
        # Only keep the last 'smoothness' datapoints
        if len(self.data) > self.smoothness:
            self.data.pop(0)
        self.data.append(next_raw_data_point)
        
    def get_smoothed_datapoint(self, next_raw_datapoint):
        # Average of the last 'smothness' datapoints
        self.__add_datapoint(next_raw_datapoint)
        return sum(self.data)/float(len(self.data))
    
        
        