"""This module is an encapsulator for the AnnotatedChunks class."""
import csv
from math import ceil
import random as rand
import numpy as np
from scipy.io import wavfile
from software_model import utility_functions as utils
from software_model.constants import DATA_FILES_LOCATION, DOWNSAMPLE_FACTOR, NUM_SAMPS_IN_CHUNK


class Chunks:
    def __init__(self, file_list):
        self._file_list = file_list
        self._audio = self.read_wav_files()
        self.num_files = len(file_list)
    
    def read_wav_files(self):
        """This method reads in .wav sound files."""

        files_to_return = []

        try:
            files_to_return = list(
                np.array(wavfile.read(DATA_FILES_LOCATION + file_name + '_downsampled.wav')[1]) for file_name in self._file_list)
        except FileNotFoundError:
            # If the file isn't found, attempt to see if the file simply hasn't been downsampled yet and do so.
            for file_name in self._file_list:
                utils.downsample(DATA_FILES_LOCATION + file_name + '.wav', DATA_FILES_LOCATION + file_name + '_downsampled.wav', DOWNSAMPLE_FACTOR)
            # Retry Reading the files again
            files_to_return = list(np.array(wavfile.read(DATA_FILES_LOCATION + file_name + '_downsampled.wav')[1]) for file_name in self._file_list)

        return files_to_return
    def get_chunk(self, file_index, chunk_index):
        """Retrives a specified chunk given a file index and chunk index"""
        audio_file = self._audio[file_index]

        start = chunk_index * NUM_SAMPS_IN_CHUNK
        end = start + NUM_SAMPS_IN_CHUNK

        chunk = audio_file[start:end, :]

        if end >= audio_file.shape[0]:
            chunk = np.lib.pad(chunk, ((0, end - audio_file.shape[0]), (0, 0)), 'constant', constant_values=(0, 0))

        return chunk