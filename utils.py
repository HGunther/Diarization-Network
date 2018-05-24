"""This module provides various utility functions."""
from scipy.fftpack import fft
from scipy.io import wavfile
from scipy.signal import decimate
import numpy as np


def downsample(infile, outfile, downrate_factor):
    """Downsample an audio file"""
    rate, aud = wavfile.read(infile)
    daud = decimate(aud, downrate_factor, axis=0)
    daudint = np.int16(daud / np.max(np.abs(daud)) * 32767)
    wavfile.write(outfile, rate // downrate_factor, daudint)


def downsample_inline(infile, downrate_factor):
    """Downsample an audio file"""
    _, aud = wavfile.read(infile)
    daud = decimate(aud, int(downrate_factor), axis=0)
    return np.int16(daud / np.max(np.abs(daud)) * 32767)


def get_freqs(batch):
    """FFT"""
    # Take FFT of each
    for i in range(batch.shape[0]):
        batch[i, :, 0, 0] = np.abs(fft(batch[i, :, 0, 0]))
        batch[i, :, 1, 0] = np.abs(fft(batch[i, :, 1, 0]))

    # Real number symmetry of Fourier Transform
    half_length = batch.shape[1] // 2
    batch = batch[:, :half_length, :, :]

    return batch
