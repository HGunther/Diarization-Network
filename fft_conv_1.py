from scipy.io import wavfile
from scipy.fftpack import rfft, fft
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Constants
chunk_size_ms = 1000
num_channels = 2
samp_rate_s = 44000 # Vals / s (Hz)
samp_rate_ms = samp_rate_s / 1000 # vals / ms (kHz)
num_samps_in_chunk = int(chunk_size_ms * samp_rate_ms)
num_inputs = int(num_samps_in_chunk / 2)
num_outputs = 2

# Input Layer
with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=[None, num_inputs, num_channels, 1], name="X")
    y = tf.placeholder(tf.int32, shape=[None], name="y")

# Convolutive Layers

# Create convolutive maps
# Number of convolutive maps in layer
conv1_fmaps = 32
# Size of each kernel
conv1_ksize = [int(samp_rate_ms * 20), num_channels]
# Move convolutive map 10 ms at a time
timeStride = int(samp_rate_ms * 10)
channelStride = 1
conv1_stride = [timeStride, channelStride]
conv1_pad = "SAME"

with tf.name_scope("conv1"):
    conv1 = tf.layers.conv2d(X, filters=conv1_fmaps, kernel_size=conv1_ksize,
                            strides=conv1_stride, padding=conv1_pad,
                            activation=tf.nn.relu, name="conv1")
    # conv1 shape is
    # [-1, num_inputs / timeStride, num_channels / channelStride, conv1_fmaps]
    conv1_flat = tf.reshape(conv1, shape=[-1, conv1_fmaps * (num_inputs // timeStride) * (num_channels // channelStride)])

# Number of nodes in fully connected layer
n_fc1 = 10
with tf.name_scope("fc1"):
    fc1 = tf.layers.dense(conv1_flat, n_fc1, activation=tf.nn.relu, name="fc1")

# Output Layer
with tf.name_scope("output"):
    logits = tf.layers.dense(fc1, num_outputs, name="output")
    Y_prob = tf.nn.softmax(logits, name="Y_prob")

# Training nodes
with tf.name_scope("train"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)

# Evaluate the network
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# Initialize the network
with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()



# TODO
# Get input somehow
# this chunk is mockup input
from math import sin
# A period 1/50 curve (frequency 50 Hz)
chunk1 = np.array([sin(314.159265 * (x / samp_rate_s)) for x in range(num_samps_in_chunk)])
# A period 1/100 curve (frequency 100 Hz)
chunk2 = np.array([sin(628.318520 * (x / samp_rate_s)) for x in range(num_samps_in_chunk)])
chunk = np.stack((chunk1, chunk2), axis=1).reshape([1, num_samps_in_chunk, num_channels, 1])

def get_freqs(chunk, show=False):
    # Take FFT of chunk
    channel_1 = fft(chunk[0, :, 0, 0])
    channel_2 = fft(chunk[0, :, 1, 0])
    chunk_freq = np.abs(np.stack((channel_1[:num_inputs], channel_2[:num_inputs]), axis=1))
    chunk_freq = chunk_freq.reshape([1, num_inputs, num_channels, 1])

    if(show):
        # Get appropriate time labels
        k = np.arange(num_inputs)
        T = samp_rate_s / len(k)
        freq_label = k * T

        # Look at FFT
        plt.plot(freq_label, chunk_freq[0, :, 0, 0])
        plt.plot(freq_label,chunk_freq[0, :, 1, 0])
        plt.show()

    return chunk_freq

n_epochs = 10

with tf.Session() as sess:
    init.run()

    X_chunk = get_freqs(chunk)
    y_chunk = np.array([0])
    acc = accuracy.eval(feed_dict={X: X_chunk, y: y_chunk})

    print(acc)

    for epoch in range(n_epochs):
        X_batch = X_chunk
        y_batch = y_chunk
        sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

        #save_path = saver.save(sess, "./my_mnist_model")
