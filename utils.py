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
