from scipy.fftpack import rfft, fft
from math import ceil
from scipy.io import wavfile
from scipy.signal import decimate
import numpy as np


def downsample(infile, outfile, downrate):
    """Downsample an audio file"""
    rate, aud = wavfile.read(infile)
    daud = decimate(aud, downrate, axis=0)
    daudint = np.int16(daud/np.max(np.abs(daud)) * 32767)
    wavfile.write(outfile, rate // downrate, daudint)


def downsample_inline(infile, downrate_factor):
    """Downsample an audio file"""
    rate, aud = wavfile.read(infile)
    daud = decimate(aud, int(downrate_factor), axis=0)
    return np.int16(daud/np.max(np.abs(daud)) * 32767)
    
    
def get_freqs(batch, show=False):
    """FFT"""
    # Take FFT of each
    for i in range(batch.shape[0]):
        batch[i, :, 0, 0] = np.abs(fft(batch[i, :, 0, 0]))
        batch[i, :, 1, 0] = np.abs(fft(batch[i, :, 1, 0]))

    # Real number symmetry of Fourier Transform
    half_length = batch.shape[1] // 2
    batch = batch[:,:half_length,:,:]

    return batch
    
def assert_eq_shapes(shape1, shape2, indices):
    """Sanity check. Asserts that shape1 == shape2 at each index in the indicies"""
    for i in indices:
        errmsg = 'Index ' + str(i) + ': ' + str(shape1[i]) + ' vs ' + str(shape2[i])
        assert shape1[i] == shape2[i], errmsg

    
####
# HELPER FUNCTIONS AND CLASSES FOR FRAMEWORK.PY
####

def ms_to_seconds_as_string(time_in_ms_float):
    time_in_sec = time_in_ms_float/1000.0
    return "{:10.4f}".format(time_in_sec)

def format_csv_line_as_list(start_time_in_ms, end_time_in_ms, is_speaking, speaker_id):
    return [ms_to_seconds_as_string(start_time_in_ms), ms_to_seconds_as_string(end_time_in_ms), str(int(is_speaking)), str(int(speaker_id))]


class Smoother:
    def __init__(self, smoothness=5):
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