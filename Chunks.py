from scipy.io import wavfile
import numpy as np
import random as rand
import utils as utils
from math import ceil
import csv
from Constants import *

class Chunks:
    def __init__(self, file_list, chunk_size_ms, seed=42, samp_rate=ORIGINAL_SAMP_RATE_S):
        self._chunk_size_ms = chunk_size_ms
        self._samp_rate = samp_rate
        self._audio = self.read_files(file_list)
        self._spk1 = self.get_speaker(file_list, '_Spk1')
        self._spk2 = self.get_speaker(file_list, '_Spk2')
        self.set_seed(seed)
        self.num_files = len(file_list)

    def set_seed(self, seed):
        rand.seed(seed)

    def read_files(self, file_list):

        files_to_return = []

        try:
            files_to_return = list(np.array(wavfile.read('Data/' + f + '_downsampled.wav')[1]) for f in file_list)
        except FileNotFoundError:
            #If the file isn't found, attempt to see if the file simply hasn't been downsampled yet and do so.
            for f in file_list:
                utils.downsample('Data/' + f + '.wav', 'Data/' + f + '_downsampled.wav', 4)
            #Retry Reading the files again
            files_to_return = list(np.array(wavfile.read('Data/' + f + '_downsampled.wav')[1]) for f in file_list)
        
        return files_to_return

    def read_annot(self, f, ext):
        spk = []
        with open('Data/' + f + ext + '.csv', 'r') as f:
            reader = csv.reader(f)
            # Skip the header line
            next(f)
            for row in reader:
                spk.append(np.array(row[1:3], dtype='float32'))
        return np.array(spk)

    def get_speaker(self, file_list, ext):
        return list(self.read_annot(f, ext) for f in file_list)

    def get_chunk(self, file_index, chunk_index):
        audio_file = self._audio[file_index]
        spk1 = self._spk1[file_index]
        spk2 = self._spk2[file_index]

        num_samps_in_chunk = int(self._chunk_size_ms * (self._samp_rate / 1000))
        num_chunks_in_file = ceil(audio_file.shape[0] / num_samps_in_chunk)

        start = chunk_index * num_samps_in_chunk
        end = start + num_samps_in_chunk

        chunk = audio_file[start:end, :]

        if end >= audio_file.shape[0]:
            chunk = np.lib.pad(chunk, ((0, end-audio_file.shape[0]),(0, 0)), 'constant', constant_values=(0,0))

        midpoint_samp = (start+end)//2
        midpoint_sec = midpoint_samp / self._samp_rate

        spk1_bin = np.digitize(midpoint_sec, spk1[:,0])
        spk2_bin = np.digitize(midpoint_sec, spk2[:,0])

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
        file_index = rand.randint(0, len(self._audio)-1)        
        audio_file = self._audio[file_index]
        num_samps_in_chunk = int(self._chunk_size_ms * (self._samp_rate / 1000))
        num_chunks_in_file = ceil(audio_file.shape[0] / num_samps_in_chunk)
        chunk_index = rand.randint(0, num_chunks_in_file-1)

        return self.get_chunk(file_index, chunk_index)

    def get_all_as_batch(self):
        num_samps_in_chunk = int(self._chunk_size_ms * (self._samp_rate / 1000))

        batch = []
        y = []
        
        for f in range(self.num_files):
            audio_file = self._audio[f]
            num_samps_in_chunk = int(self._chunk_size_ms * (self._samp_rate / 1000))
            num_chunks_in_file = ceil(audio_file.shape[0] / num_samps_in_chunk)
            for c in range(num_chunks_in_file):
                chunk, status = self.get_chunk(f, c)
                chunk = chunk.reshape([1, num_samps_in_chunk, 2, 1])
                batch.append(chunk)
                y.append(status)
        batch = np.concatenate(batch, axis=0)
        y = np.array(y)
        
        # for i in range(batch_size):
        #     chunk, status = self.get_rand_chunk()
        #     chunk = chunk.reshape([1, num_samps_in_chunk, 2, 1])
        #     batch.append(chunk)
        #     y.append(status)
        
        return batch, y

    def get_rand_batch(self, batch_size):
        num_samps_in_chunk = int(self._chunk_size_ms * (self._samp_rate / 1000))

        batch = []
        y = []
        for i in range(batch_size):
            chunk, status = self.get_rand_chunk()
            chunk = chunk.reshape([1, num_samps_in_chunk, 2, 1])
            batch.append(chunk)
            y.append(status)
        batch = np.concatenate(batch, axis=0)
        y = np.array(y)

        return batch, y

# # TEST CODE
# c = Chunks(('HS_D36', 'HS_D37'), 250)
# chunk, status = c.get_rand_chunk()
# print(chunk, chunk.shape)

# chunk, status = c.get_rand_chunk()
# print(chunk, chunk.shape)

# batch, statuses = c.get_rand_batch(5)
# print(batch.shape, statuses.shape)

# batch, statuses = c.get_seq_batch()
# print(batch.shape, statuses.shape)