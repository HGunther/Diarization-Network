import numpy as np
from scipy.fftpack import fft

class Chunk:

    def __init__(self, chunk_data):
        self.__raw0 = np.array(chunk_data[:,0])
        self.__raw1 = np.array(chunk_data[:,0])
        self.__fft0 = None # Use lazy loading
        self.__fft1 = None # Use lazy loading
   
    def get_raw0(self):
        return self.__raw0
    
    def get_raw1(self):
        return self.__raw1
    
    def get_fft0(self):
        if self.__fft0 == None:
            self.__fft0 = self.__get_fft(self.get_raw0())
            
        return self.__fft0
    def get_fft1(self):
        if self.__fft1 == None:
            self.__fft1 = self.__get_fft(self.get_raw1())

        return self.__fft1
 
    def __get_fft(self, raw):
            fft_data = np.abs(fft(raw))
            # Split in half due to the real number symmetry of the fft
            half_length = fft_data.shape[0]//2
            fft_data = np.array(fft_data[:half_length+1]) 
            return fft_data