# For debugging
import sys
old_tr = sys.gettrace()
sys.settrace(None)

from scipy.io import wavfile
from scipy.fftpack import rfft, fft
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from math import ceil

# For debugging
sys.settrace(old_tr)

# Info for TensorBoard
from datetime import datetime

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

# Constants
chunk_size_ms = 250
num_channels = 2
samp_rate_s = 44000 # Vals / s (Hz)
samp_rate_ms = samp_rate_s // 1000 # vals / ms (kHz)
num_samps_in_chunk = chunk_size_ms * samp_rate_ms
num_inputs = num_samps_in_chunk // 2 # Real symmetry in Fourier Transform
num_outputs = 2

# For sanity checks, assert that shape1==shape2 at each index in indices
def assert_eq_shapes(shape1, shape2, indices):
    for i in indices:
        errmsg = 'Index ' + str(i) + ': ' + str(shape1[i]) + ' vs ' + str(shape2[i])
        assert shape1[i] == shape2[i], errmsg

# Input Layer
with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=[None, num_inputs, num_channels, 1], name="X")
    y = tf.placeholder(tf.int32, shape=[None], name="y")

# Convolutive Layers

# Create convolutive maps
# Number of convolutive maps in layer
conv1_fmaps = 32
# Size of each kernel
conv1_ksize = [15, num_channels]
conv1_time_stride = 2
conv1_channel_stride = 1
conv1_stride = [conv1_time_stride, conv1_channel_stride]
conv1_pad = "SAME"

# Number of convolutive maps in layer
conv2_fmaps = 64
# Size of each kernel
conv2_ksize = [10, num_channels]
conv2_time_stride = 1
conv2_channel_stride = 1
conv2_stride = [conv2_time_stride, conv2_channel_stride]
conv2_pad = "SAME"

with tf.name_scope("convclust1"):
    conv1 = tf.layers.conv2d(X, filters=conv1_fmaps, kernel_size=conv1_ksize,
                            strides=conv1_stride, padding=conv1_pad,
                            activation=tf.nn.relu, name="conv1")

    conv1_output_shape = [-1, ceil(num_inputs / conv1_time_stride), ceil(num_channels / conv1_channel_stride), conv1_fmaps]
    assert_eq_shapes(conv1_output_shape, conv1.get_shape(), (1,2,3))

    conv2 = tf.layers.conv2d(conv1, filters=conv2_fmaps, kernel_size=conv2_ksize,
                            strides=conv2_stride, padding=conv2_pad,
                            activation=tf.nn.relu, name="conv2")

    conv2_output_shape = [-1, ceil(conv1_output_shape[1] / conv2_time_stride), ceil(conv1_output_shape[2] / conv2_channel_stride), conv2_fmaps]
    assert_eq_shapes(conv2_output_shape, conv2.get_shape(), (1,2,3))

with tf.name_scope("pool3"):
    pool3 = tf.nn.avg_pool(conv2, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding="VALID")

    pool3_output_shape = [-1, conv2_output_shape[1] // 2, conv2_output_shape[2], conv2_fmaps]
    assert_eq_shapes(pool3_output_shape, pool3.get_shape(), (1,2,3))

    pool3_flat = tf.reshape(pool3, shape=[-1, conv2_fmaps * pool3_output_shape[1] * pool3_output_shape[2]])

# Number of nodes in fully connected layer
n_fc1 = 10
with tf.name_scope("fc1"):
    fc1 = tf.layers.dense(pool3_flat, n_fc1, activation=tf.nn.relu, name="fc1")

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

def get_fake_chunk(s):
    # this chunk is mockup input
    from math import sin, pi
    if s==0:
        freq1 = 50
        freq2 = 100
    else:
        freq1 = 250
        freq2 = 400
    # A curve with frequency freq1 Hz
    chunk1 = np.array([sin(2 * pi * freq1 * (x / samp_rate_s)) for x in range(num_samps_in_chunk)])
    # A curve with frequency freq2 Hz
    chunk2 = np.array([sin(2 * pi * freq2 * (x / samp_rate_s)) for x in range(num_samps_in_chunk)])
    chunk = np.stack((chunk1, chunk2), axis=1).reshape([1, num_samps_in_chunk, num_channels, 1])

    return chunk, np.array([s])

def get_next_batch():
    # TODO
    # Get input somehow
    pass

def get_freqs(batch, show=False):
    # Take FFT of each
    for i in range(batch.shape[0]):
        batch[i, :, 0, 0] = np.abs(fft(batch[i, :, 0, 0]))
        batch[i, :, 1, 0] = np.abs(fft(batch[i, :, 1, 0]))

    # Real number symmetry of Fourier Transform
    batch = batch[:,:num_inputs,:,:]

    if(show):
        # Get appropriate time labels
        k = np.arange(num_inputs)
        T = samp_rate_s / len(k)
        freq_label = k * T

        for i in range(batch.shape[0]):
            # Look at FFT
            plt.plot(freq_label, batch[i, :, 0, 0])
            plt.plot(freq_label, batch[i, :, 1, 0])
            plt.show()

    return batch

def debug(X_chunk,y_chunk):
    print('X CHUNK SHAPE: ',X_chunk.shape)
    print('Y CHUNK SHAPE: ',y_chunk.shape)
    print('X: ',X)
    print('y: ',y)
    print('conv1: ',conv1)
    print('conv2: ',conv2)
    print('pool3: ',pool3)
    print('pool3flat: ',pool3_flat)
    print('fc1: ',fc1)
    print('logits: ',logits)
    print('Yprob: ',Y_prob)

num_epochs = 10
num_iterations = 1

with tf.Session() as sess:
    init.run()

    X_batch, y_batch = get_fake_chunk(0)
    X_chunk = get_freqs(X_batch, True)
    y_chunk = y_batch
    
    # Prints the structure of the network one layer at a time
    debug(X_chunk,y_chunk)

    # Initial check
    acc = accuracy.eval(feed_dict={X: X_chunk, y: y_chunk})

    print(acc)

    for epoch in range(num_epochs):
        for i in range(num_iterations):
            X_batch, y_batch = get_fake_chunk(i % 2)
            X_batch = get_freqs(X_batch)
        sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

        #save_path = saver.save(sess, "./my_mnist_model")
