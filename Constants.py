# *****************************************************************************
# CONSTANTS FILE is awesome
# *****************************************************************************

# About data
NUM_CHANNELS = 2
ORIGINAL_SAMP_RATE_S = 44100

# Downsampled data
DOWNSAMPLE_FACTOR = 4
SAMP_RATE_S = 44100//4 # Vals / s (Hz)
SAMP_RATE_MS = SAMP_RATE_S / 1000 # vals / ms (kHz)

# Chunks (post downsample)
CHUNCK_SIZE_MS = 100 # Milliseconds, not megaseconds
NUM_SAMPS_IN_CHUNCK = int(CHUNCK_SIZE_MS * SAMP_RATE_MS)