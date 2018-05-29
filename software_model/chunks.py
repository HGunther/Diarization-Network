"""This module is an encapsulator for the Chunks class."""
import csv
from math import ceil
import random as rand
import numpy as np
from scipy.io import wavfile
from software_model import utility_functions as utils
from constants import ORIGINAL_SAMP_RATE_S, DATA_FILES_LOCATION, DOWNSAMPLE_FACTOR


class Chunks:
    """This class is representative of an element of a partioned sound file."""

    def __init__(self, file_list, chunk_size_ms, seed=42, samp_rate=ORIGINAL_SAMP_RATE_S):
        self._file_list = file_list
        self._chunk_size_ms = chunk_size_ms
        self._samp_rate = samp_rate
        self._audio = self.read_wav_files()
        self._spk1 = self.get_speaker('_Spk1')
        self._spk2 = self.get_speaker('_Spk2')
        self.num_files = len(file_list)
        rand.seed(seed)

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

    def get_speaker(self, ext):
        """This method reads in speaking/non-speaking data from a csv file for a given speaker."""
        return list(self.read_annotations_from_csv(f, ext) for f in self._file_list)

    def read_annotations_from_csv(self, file_name, ext):
        """This method reads and parses the speaking/non-speaking data from a csv file."""
        spk = []
        with open(DATA_FILES_LOCATION + file_name + ext + '.csv', 'r') as file_being_read:
            reader = csv.reader(file_being_read)
            # Skip the header line
            next(file_being_read)
            for row in reader:
                spk.append(np.array(row[1:3], dtype='float32'))
        return np.array(spk)

    def get_chunk(self, file_index, chunk_index):
        """Retrives a specified chunk given a file index and chunk index"""
        audio_file = self._audio[file_index]
        spk1 = self._spk1[file_index]
        spk2 = self._spk2[file_index]

        num_samps_in_chunk = int(self._chunk_size_ms * (self._samp_rate / 1000))

        start = chunk_index * num_samps_in_chunk
        end = start + num_samps_in_chunk

        chunk = audio_file[start:end, :]

        if end >= audio_file.shape[0]:
            chunk = np.lib.pad(chunk, ((0, end - audio_file.shape[0]), (0, 0)), 'constant', constant_values=(0, 0))

        midpoint_samp = (start + end) // 2
        midpoint_sec = midpoint_samp / self._samp_rate

        spk1_bin = np.digitize(midpoint_sec, spk1[:, 0])
        spk2_bin = np.digitize(midpoint_sec, spk2[:, 0])

        # If the midpoint of the chunk is part of the padded section,
        # There is no speaking here
        if spk1_bin >= spk1.shape[0]:
            spk1_status = 0
        else:
            spk1_status = int(spk1[spk1_bin, 1])

        if spk2_bin >= spk2.shape[0]:
            spk2_status = 0
        else:
            spk2_status = int(spk2[spk2_bin, 1])

        return chunk, [spk1_status, spk2_status]

    def get_rand_chunk(self):
        """Gets a random chunk from a random file provided in the class initialization."""

        file_index = rand.randint(0, len(self._audio) - 1)
        audio_file = self._audio[file_index]
        num_samps_in_chunk = int(self._chunk_size_ms * (self._samp_rate / 1000))
        num_chunks_in_file = ceil(audio_file.shape[0] / num_samps_in_chunk)
        chunk_index = rand.randint(0, num_chunks_in_file - 1)

        return self.get_chunk(file_index, chunk_index)

    def get_rand_batch(self, batch_size):
        """Returns a batch of the given size composed of random chunks."""

        num_samps_in_chunk = int(self._chunk_size_ms * (self._samp_rate / 1000))

        batch = []
        response_variables = []

        for _ in range(batch_size):
            chunk, status = self.get_rand_chunk()
            chunk = chunk.reshape([1, num_samps_in_chunk, 2, 1])
            batch.append(chunk)
            response_variables.append(status)

        batch = np.concatenate(batch, axis=0)
        response_variables = np.array(response_variables)

        return batch, response_variables
