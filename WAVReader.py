from scipy.io import wavfile
from scipy import signal
import numpy as np
import scipy as sp


def downsample(data, sample_rate, new_sample_rate):
    """Attempts (and fails) to downsample audio

    Args:
        param1 (nx2 numpy array): A numpy matrix containing the two channel, audio data
        param2 (int): the curent sample rate of the audio
        param3 (int): the desired sample rate of the audio
    Returns:
        A nx2 numpy matrix containing the downsampled audio data
    """
    num_samples = data.shape[0]
    new_num_samples = int(num_samples/sample_rate * new_sample_rate)
    new_data = sp.signal.resample(data, new_num_samples)

    # REDUCTION_FACTOR = int(sample_rate / new_sample_rate)
    # new_data = data[::REDUCTION_FACTOR]
    return new_data

def downsample_by_factor(data, reduction_factor):
    """Actually downsamples audio (in the laziest way possible)

    Args:
        param1 (nx2 numpy array): A numpy matrix containing the two channel, audio data
        param2 (int): The factor by which the audio should be downsampled
    Returns:
        A nx2 numpy matrix containing the downsampled audio data
    """
    return data[::reduction_factor]
