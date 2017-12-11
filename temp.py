from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import csv

spk1 = []
with open('Data/HS_D37_Spk1.csv', 'r') as f:
    reader = csv.reader(f)
    # Skip the header line
    next(f)
    for row in reader:
        spk1.append(np.array(row, dtype='float32'))

# Now spk1 is an N x 4 array of the form
# start time, end time, isSpeaking (encoded 0 or 1)
spk1 = np.array(spk1)

spk2 = []
with open('Data/HS_D37_Spk2.csv', 'r') as f:
    reader = csv.reader(f)
    # Skip the header line
    next(f)
    for row in reader:
        spk2.append(np.array(row, dtype='float32'))

# Now spk1 is an N x 4 array of the form
# start time, end time, isSpeaking (encoded 0 or 1)
spk2 = np.array(spk2)

file_rate, file_data = wavfile.read('Data/HS_D37.wav')
file_data = np.array(file_data)

def get_chunk(chunk_size_ms, chunk_num):
    num_samps_in_chunk = int(chunk_size_ms * (file_rate / 1000))
    start = chunk_num * num_samps_in_chunk
    end = start + num_samps_in_chunk
    chunk = file_data[start:end, :]
    midpoint_samp = (start+end)//2
    midpoint_sec = midpoint_samp / file_rate

    spk1_status = int(spk1[np.digitize(midpoint_sec, spk1[:,1]), 2])
    spk2_status = int(spk2[np.digitize(midpoint_sec, spk2[:,1]), 2])

    return chunk, spk1_status