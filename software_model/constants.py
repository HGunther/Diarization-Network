"""This module provides various constants used throughout the application"""

from datetime import datetime

# About data
NUM_CHANNELS = 2
ORIGINAL_SAMP_RATE_S = 44100
DATA_FILES_LOCATION = 'Data/'
PREDICTON_FILES_LOCATION = 'Data/Predicted/'

# Downsampled data
DOWNSAMPLE_FACTOR = 4  # 4
SAMP_RATE_S = ORIGINAL_SAMP_RATE_S // DOWNSAMPLE_FACTOR  # Vals / s (Hz)
SAMP_RATE_MS = SAMP_RATE_S / 1000  # vals / ms (kHz)
SAVE_DOWNSAMPLED_FILES = True

# Chunks (post downsample)
CHUNK_SIZE_MS = 3000  # 100 # Milliseconds, not megaseconds
CHUNK_SIZE_S = CHUNK_SIZE_MS/1000
NUM_SAMPS_IN_CHUNK = int(CHUNK_SIZE_MS * SAMP_RATE_MS)

# TensorFlow Log File
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
ROOT_LOGDIR = "tf_logs"
LOGDIR = "{}/run-{}/".format(ROOT_LOGDIR, now)
