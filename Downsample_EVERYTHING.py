import utils as utils

files = ['HS_D{0:0=2d}'.format(i) for i in range(1, 38)]

for f in files:
    utils.downsample('Data/' + f + '.wav', 'Data/' + f + '_downsampled.wav', 4)