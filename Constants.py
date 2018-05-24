"""This module provides various constants used throughout the application"""

# About data
NUM_CHANNELS = 2
ORIGINAL_SAMP_RATE_S = 44100
DATA_FILES_LOCATION = 'Data/'

# Downsampled data
DOWNSAMPLE_FACTOR = 4
SAMP_RATE_S = 44100 // 4  # Vals / s (Hz)
SAMP_RATE_MS = SAMP_RATE_S / 1000  # vals / ms (kHz)

# Chunks (post downsample)
CHUNK_SIZE_MS = 100  # Milliseconds, not megaseconds
NUM_SAMPS_IN_CHUNK = int(CHUNK_SIZE_MS * SAMP_RATE_MS)
