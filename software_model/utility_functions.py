"""This module provides various utility functions."""
from scipy.fftpack import fft
from scipy.io import wavfile
from scipy.signal import decimate
import numpy as np


def downsample(infile, outfile, downrate_factor):
    """Downsample an audio file"""
    rate, aud = wavfile.read(infile)
    daud = decimate(aud, downrate_factor, axis=0)
    int16_max = np.iinfo(np.dtype('int16')).max
    daudint = np.int16(daud / np.max(np.abs(daud)) * int16_max)
    wavfile.write(outfile, rate // downrate_factor, daudint)


def downsample_inline(infile, downrate_factor):
    """Downsample an audio file"""
    _, aud = wavfile.read(infile)
    daud = decimate(aud, int(downrate_factor), axis=0)
    int16_max = np.iinfo(np.dtype('int16')).max
    return np.int16(daud / np.max(np.abs(daud)) * int16_max)


def apply_fft(array_of_chunk_audio):
    """FFT"""
    # Take FFT of each chunk in the array
    for i in range(array_of_chunk_audio.shape[0]):
        # Channel 1
        array_of_chunk_audio[i, :, 0, 0] = np.abs(fft(array_of_chunk_audio[i, :, 0, 0]))
        # Channel 2
        array_of_chunk_audio[i, :, 1, 0] = np.abs(fft(array_of_chunk_audio[i, :, 1, 0]))

    # Real number symmetry of Fourier Transform
    half_length = array_of_chunk_audio.shape[1] // 2
    array_of_chunk_audio = array_of_chunk_audio[:, :half_length, :, :]

    return array_of_chunk_audio
