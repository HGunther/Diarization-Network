import numpy as np
from math import ceil
from utils import downsample_inline

class ProductionChunks:
    def __init__(self, file_name, chunk_size_ms, downsample_rate=44100/4):
        self._chunk_size_ms = chunk_size_ms
        self._samp_rate = downsample_rate
        self._audio = downsample_inline(file_name, self._samp_rate)
        self._chunk_count = int(ceil(len(self._audio[0])/float(len(self.get_chunk(0)[0]))))

    def get_chunk_count(self):
        return self._chunk_count

    def get_chunk(self, chunk_index):
        audio_file = self._audio

        num_samps_in_chunk = int(self._chunk_size_ms * (self._samp_rate / 1000))

        start = chunk_index * num_samps_in_chunk
        end = start + num_samps_in_chunk

        chunk = audio_file[start:end, :]

        if end >= audio_file.shape[0]:
            chunk = np.lib.pad(chunk, ((0, end-audio_file.shape[0]),(0, 0)), 'constant', constant_values=(0,0))

        return chunk
