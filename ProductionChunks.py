import numpy as np
from math import ceil
from utils import downsample_inline
from Constants import *

class ProductionChunks:
    def __init__(self, file_name, chunk_size_ms=CHUNK_SIZE_MS, downsample_factor=DOWNSAMPLE_FACTOR):
        self._chunk_size_ms = chunk_size_ms
        self._audio = downsample_inline(file_name, downsample_factor)
        # self._chunk_count = int(ceil(len(self._audio[0])/float(len(self.get_chunk(0)[0]))))
        self._chunk_count = int(ceil(self._audio.shape[0]/NUM_SAMPS_IN_CHUNK))

    def get_chunk_count(self):
        return self._chunk_count

    def get_chunk(self, chunk_index):
        audio_file = self._audio

        start = chunk_index * NUM_SAMPS_IN_CHUNK
        end = start + NUM_SAMPS_IN_CHUNK

        chunk = audio_file[start:end, :]

        if end >= audio_file.shape[0]:
            chunk = np.lib.pad(chunk, ((0, end-audio_file.shape[0]),(0, 0)), 'constant', constant_values=(0,0))

        return chunk
