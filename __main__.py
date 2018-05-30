'''
Created on May 29, 2018

@author: Sam
'''
import numpy as np
import os
import sys
from scipy.fftpack import fft
from software_model.neural_network import NeuralNetwork
from software_model.constants import *
from software_model.diarizer import Diarizer

if __name__ == '__main__':
    
    
    #sys.exit()
    cwd = os.getcwd()
    
    #print(int((11 * 60 * SAMP_RATE_S / NUM_SAMPS_IN_CHUNK) / 1))

    #print(cwd)
    #files = ['HS_D{0:0=2d}'.format(i) for i in range(1, 38)]
    #del files[files.index('HS_D11')]
    #del files[files.index('HS_D22')]
    
    #net = NeuralNetwork()
    #model_loc = "Model/may29.ckpt"
    model_in = "Model/ultimate_model_saved_weights.ckpt"
    #net.train_network(files, model_loc, in_model_location=model_in, num_epochs=5)
    
    dire = Diarizer(model_in)
    dire.annotate_wav_file("HS_D01")