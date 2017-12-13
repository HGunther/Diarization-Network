from scipy.fftpack import rfft, fft
from math import ceil
from scipy.io import wavfile
from scipy.signal import decimate
import numpy as np

def downsample(infile, outfile, downrate):
    rate, aud = wavfile.read(filename)
    daud = decimate(aud, downrate, axis=0)
    daudint = np.int16(daud/np.max(np.abs(daud)) * 32767)
    wavfile.write(outfile, rate // downrate, daudint)

# def get_fake_chunk(s):
#     # this chunk is mockup input
#     from math import sin, pi
#     if s==0:
#         freq1 = 50
#         freq2 = 100
#     else:
#         freq1 = 250
#         freq2 = 400
#     # A curve with frequency freq1 Hz
#     chunk1 = np.array([sin(2 * pi * freq1 * (x / samp_rate_s)) for x in range(num_samps_in_chunk)])
#     # A curve with frequency freq2 Hz
#     chunk2 = np.array([sin(2 * pi * freq2 * (x / samp_rate_s)) for x in range(num_samps_in_chunk)])
#     chunk = np.stack((chunk1, chunk2), axis=1).reshape([1, num_samps_in_chunk, num_channels, 1])

#     return chunk, np.array([s])
    
def get_freqs(batch, show=False):
    # Take FFT of each
    for i in range(batch.shape[0]):
        batch[i, :, 0, 0] = np.abs(fft(batch[i, :, 0, 0]))
        batch[i, :, 1, 0] = np.abs(fft(batch[i, :, 1, 0]))

    # Real number symmetry of Fourier Transform
    half_length = batch.shape[1] // 2
    batch = batch[:,:half_length,:,:]

    if(show):
        # Get appropriate time labels
        k = np.arange(half_length)
        T = samp_rate_s / (2 * len(k))
        freq_label = k * T

        for i in range(batch.shape[0]):
            # Look at FFT
            plt.plot(freq_label, batch[i, :, 0, 0])
            plt.plot(freq_label, batch[i, :, 1, 0])
            plt.show()

    return batch