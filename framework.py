import numpy as np
import csv
import sys
from os.path import basename
from utils import Smoother
from utils import format_csv_line_as_list
from utils import ms_to_seconds_as_string
from ProductionChunks import ProductionChunks
from Constants import *

# Get wave filename from the first command line argument
wave_filename = sys.argv[1]

# Get the names of the two CSV files
speaker1_file = basename(wave_filename)+"_Spk1.csv"
speaker2_file = basename(wave_filename)+"_Spk2.csv"

with open (speaker1_file, 'w') as file1, open (speaker2_file, 'w') as file2:
    # Set up CSV files for writing
    csv1 = csv.writer(file1)
    csv2 = csv.writer(file2)
    
    csv_header = ["tmi0","tmax","text","tier"]
    csv1.writerow(csv_header)
    csv2.writerow(csv_header)

    # Values to keep track of
    current_time_in_ms = 0
    speaker1_segment_start_time = 0
    speaker2_segment_start_time = 0
    
    # Start up neural network
    print('Importing tensorflow and the neural network structure')
    from The_Ultimate_Net import evaluate
    # nn.initialize()

    # Create chunk generator (data source)
    chunks_gen = ProductionChunks(wave_filename, CHUNCK_SIZE_MS)
    
    # # Get data
    batch = []
    for i in range(chunks_gen.get_chunk_count()):
        chunk = chunks_gen.get_chunk(i)
        chunk = chunk.reshape([1, NUM_SAMPS_IN_CHUNCK, 2, 1])
        batch.append(chunk)
    batch = np.concatenate(batch, axis=0)

    model = "Model/ultimate_model_supertraining.ckpt"
    result1 = evaluate(batch, model)

    temp = np.array(batch[:,:,0,:])
    batch[:,:,0,:] = np.array(batch[:,:,1,:])
    batch[:,:,1,:] = temp
    
    result2 = evaluate(batch, model)



    # Set up post-net processing to write to csv files
    # speaker1_smoother = Smoother()
    # speaker2_smoother = Smoother()

    #  # Process data
    # for chunk_index in range(chunks_gen.get_chunk_count()):
    #     # Get chunk
    #     chunks = chunks_gen.get_chunk(chunk_index)
        
    #     # Evaluate chunk on NN
    #     result = nn.evaluate(chunks)
        
        # # Smooth chunks so segments are less fragmented
        # speaker1_is_talking = round(speaker1_smoother.get_smoothed_datapoint(result[0]))
        # speaker2_is_talking = round(speaker2_smoother.get_smoothed_datapoint(result[1]))

        # # Preactions
        # current_time_in_ms += CHUNCK_SIZE_MS
        
        # # Actions: Write to CSV file and update segment time if needed
        # if speaker1_is_talking != speaker1_is_talking_last_chunk:
        #     csv_line = format_csv_line_as_list(speaker1_segment_start_time, current_time_in_ms, speaker1_is_talking, 1)
        #     csv1.writerow(csv_line)
        #     speaker1_segment_start_time = current_time_in_ms
            
        # if speaker2_is_talking != speaker2_is_talking_last_chunk:
        #     csv_line = format_csv_line_as_list(speaker2_segment_start_time, current_time_in_ms, speaker2_is_talking, 2)
        #     csv2.writerow(csv_line)
        #     speaker2_segment_start_time = current_time_in_ms
        
        # # Postactions
        # speaker1_is_talking_last_chunk = speaker1_is_talking
        # speaker2_is_talking_last_chunk = speaker2_is_talking

# Files are now written
print('Done')




    
        
        