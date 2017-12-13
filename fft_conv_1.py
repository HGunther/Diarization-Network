import numpy as np
from math import ceil
from Chunks import Chunks
from utils import *

# For debugging
import sys
old_tr = sys.gettrace()
sys.settrace(None)

# To disable warning that building TF from source will make it faster.
# For more information see:
# https://www.tensorflow.org/install/install_sources
# https://stackoverflow.com/questions/41293077/how-to-compile-tensorflow-with-sse4-2-and-avx-instructions
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

sys.settrace(old_tr)

# *****************************************************************************
# CONSTANTS
# *****************************************************************************
CHUNK_SIZE_MS = 250 # Milliseconds, not megaseconds
NUM_CHANNELS = 2
SAMP_RATE_S = 44100//4 # Vals / s (Hz)
SAMP_RATE_MS = SAMP_RATE_S / 1000 # vals / ms (kHz)
NUM_SAMPS_IN_CHUNK = int(CHUNK_SIZE_MS * SAMP_RATE_MS)
NUM_INPUTS = int(NUM_SAMPS_IN_CHUNK / 2)
NUM_OUTPUTS = 2

# Constants for running the training
NUM_EPOCHS = 2000
EPOCH_SIZE = 10
SAVE = False
RESTORE = False

# *****************************************************************************
# DEBUG
# *****************************************************************************

def debug():
    """Prints debug information"""
    print('X: ', X_freq)
    print('y: ', y)
    print('conv1: ', conv1)
    print('conv2: ', conv2)
    print('pool3: ', pool3)
    print('conv4: ', conv4)
    print('conv5: ', conv5)
    print('pool6: ', pool6)
    print('pool6flat: ', pool6_flat)
    print('fc1: ', fc1)
    print('fc2: ', fc2)
    print('logits: ', logits)
    print('Yprob: ', Y_prob)

# *****************************************************************************
# Data
# *****************************************************************************
import random as random
files = ['HS_D{0:0=2d}'.format(i) for i in range(1, 38)]
del files[files.index('HS_D11')]
del files[files.index('HS_D22')]
random.shuffle(files)

training_files = files[:int(0.8 * len(files))]
testing_files = files[int(0.8 * len(files)):]

print("Reading in training data")
train_data = Chunks(training_files, CHUNK_SIZE_MS, samp_rate=SAMP_RATE_S)
print("Reading in test data")
test_data = Chunks(testing_files, CHUNK_SIZE_MS, samp_rate=SAMP_RATE_S)

# *****************************************************************************
# Defining the net and layers
# *****************************************************************************
print("Defining layers in tensorflow")

# Info for TensorBoard
from datetime import datetime
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
ROOT_LOGDIR = "tf_logs"
LOGDIR = "{}/run-{}/".format(ROOT_LOGDIR, now)

# Input Layer
with tf.name_scope("inputs"):
    X_freq = tf.placeholder(tf.float32, shape=[None, NUM_INPUTS, NUM_CHANNELS, 1], name="X_freq")
    y = tf.placeholder(tf.int32, shape=[None, 2], name="y")

# Group of convolutional layers
with tf.name_scope("convclust1"):
    # Convolutive Layers

    # Create convolutive maps
    # Number of convolutive maps in layer
    conv1_fmaps = 32
    # Size of each kernel
    conv1_ksize = [15, NUM_CHANNELS]
    conv1_time_stride = 2
    conv1_channel_stride = 1
    conv1_stride = [conv1_time_stride, conv1_channel_stride]
    conv1_pad = "SAME"

    # Number of convolutive maps in layer
    conv2_fmaps = 64
    # Size of each kernel
    conv2_ksize = [10, NUM_CHANNELS]
    conv2_time_stride = 1
    conv2_channel_stride = 1
    conv2_stride = [conv2_time_stride, conv2_channel_stride]
    conv2_pad = "SAME"

    conv1 = tf.layers.conv2d(X_freq, filters=conv1_fmaps, kernel_size=conv1_ksize,
                            strides=conv1_stride, padding=conv1_pad,
                            activation=tf.nn.relu, name="conv1")

    conv1_output_shape = [-1, ceil(NUM_INPUTS / conv1_time_stride), ceil(NUM_CHANNELS / conv1_channel_stride), conv1_fmaps]
    assert_eq_shapes(conv1_output_shape, conv1.get_shape(), (1,2,3))

    conv2 = tf.layers.conv2d(conv1, filters=conv2_fmaps, kernel_size=conv2_ksize,
                            strides=conv2_stride, padding=conv2_pad,
                            activation=tf.nn.relu, name="conv2")

    conv2_output_shape = [-1, ceil(conv1_output_shape[1] / conv2_time_stride), ceil(conv1_output_shape[2] / conv2_channel_stride), conv2_fmaps]
    assert_eq_shapes(conv2_output_shape, conv2.get_shape(), (1,2,3))

# Avg Pooling layer
with tf.name_scope("pool3"):
    pool3 = tf.nn.avg_pool(conv2, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding="VALID")

    pool3_output_shape = [-1, conv2_output_shape[1] // 2, conv2_output_shape[2], conv2_fmaps]
    assert_eq_shapes(pool3_output_shape, pool3.get_shape(), (1,2,3))

# Second convolution cluster
with tf.name_scope("convclust2"):
    # Convolutive Layers

    # Create convolutive maps
    # Number of convolutive maps in layer
    conv4_fmaps = 15
    # Size of each kernel
    conv4_ksize = [10, NUM_CHANNELS]
    conv4_time_stride = 5
    conv4_channel_stride = 1
    conv4_stride = [conv4_time_stride, conv4_channel_stride]
    conv4_pad = "SAME"

    # Number of convolutive maps in layer
    conv5_fmaps = 20
    # Size of each kernel
    conv5_ksize = [5, NUM_CHANNELS]
    conv5_time_stride = 1
    conv5_channel_stride = 1
    conv5_stride = [conv5_time_stride, conv5_channel_stride]
    conv5_pad = "SAME"

    conv4 = tf.layers.conv2d(pool3, filters=conv4_fmaps, kernel_size=conv4_ksize,
                            strides=conv4_stride, padding=conv4_pad,
                            activation=tf.nn.relu, name="conv4")

    conv4_output_shape = [-1, ceil(pool3_output_shape[1] / conv4_time_stride), ceil(pool3_output_shape[2] / conv4_channel_stride), conv4_fmaps]
    assert_eq_shapes(conv4_output_shape, conv4.get_shape(), (1,2,3))

    conv5 = tf.layers.conv2d(conv4, filters=conv5_fmaps, kernel_size=conv5_ksize,
                            strides=conv5_stride, padding=conv5_pad,
                            activation=tf.nn.relu, name="conv5")

    conv5_output_shape = [-1, ceil(conv4_output_shape[1] / conv5_time_stride), ceil(conv4_output_shape[2] / conv5_channel_stride), conv5_fmaps]
    assert_eq_shapes(conv5_output_shape, conv5.get_shape(), (1,2,3))

# Max Pooling layer
with tf.name_scope("pool6"):
    pool6 = tf.nn.max_pool(conv5, ksize=[1, 3, 1, 1], strides=[1, 3, 1, 1], padding="VALID")

    pool6_output_shape = [-1, conv5_output_shape[1] // 3, conv5_output_shape[2], conv5_fmaps]
    assert_eq_shapes(pool6_output_shape, pool6.get_shape(), (1,2,3))

    pool6_flat = tf.reshape(pool6, shape=[-1, conv5_fmaps * pool6_output_shape[1] * pool6_output_shape[2]])

# Fully connected layers
with tf.name_scope("fc"):
    # Number of nodes in fully connected layer
    n_fc1 = 40
    n_fc2 = 60
    fc1 = tf.layers.dense(pool6_flat, n_fc1, activation=tf.nn.relu, name="fc1")
    fc2 = tf.layers.dense(fc1, n_fc2, activation=tf.nn.relu, name="fc2")

# Output Layer
with tf.name_scope("output"):
    logits = tf.layers.dense(fc2, NUM_OUTPUTS, name="output")
    Y_prob = tf.nn.sigmoid(logits, name="Y_prob")

# Training nodes
with tf.name_scope("train"):
    float_y = tf.cast(y, tf.float32)
    xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=float_y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)

# Evaluate the network
with tf.name_scope("eval"):
    error = Y_prob - float_y
    mse = tf.reduce_mean(tf.square(error), name='mse')

    errors = tf.abs(y - tf.cast(tf.round(Y_prob), tf.int32))
    misclassification_rate = tf.reduce_mean(tf.cast(errors, tf.float32), name='misclassification_rate')

# Initialize the network
with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

# Tensorboard stuff
with tf.name_scope("tensorboard"):
    mse_summary = tf.summary.scalar('MSE',mse)
    file_write = tf.summary.FileWriter(LOGDIR,tf.get_default_graph())


# *****************************************************************************
# Running and training the network
# *****************************************************************************
print("Preparing to run the network")
with tf.Session() as sess:
    init.run()

    # Restore variables from disk.
    if RESTORE:
        saver.restore(sess, "/tmp/model.ckpt")
        print("Model restored.")

    # Prints the structure of the network one layer at a time
    debug()

    print('\n****Pre-training accuracy*****')
    # Measure accuracy
    X_test, y_test = test_data.get_rand_batch(int(11 * 60 * 4))
    X_test_freq = get_freqs(X_test)
    
    acc_test = mse.eval(feed_dict={X_freq: X_test_freq, y: y_test})

    pmc = misclassification_rate.eval(feed_dict={X_freq: X_test_freq, y: y_test})

    print('Test MSE:', acc_test, 'pmc:', pmc)

    print('\n*****Training the net*****')
    for epoch in range(NUM_EPOCHS):
        for i in range(EPOCH_SIZE):
            # Get data
            X_batch, y_batch = train_data.get_rand_batch(EPOCH_SIZE)
            X_batch_freq = get_freqs(X_batch)

            # Log accuracy for Tensorboard reports
            if True:
                step = epoch * EPOCH_SIZE + i
                summary_str = mse_summary.eval(feed_dict={X_freq: X_batch_freq, y: y_batch})
                file_write.add_summary(summary_str, step)

            # Train
            sess.run(training_op, feed_dict={X_freq: X_batch_freq, y: y_batch})

        # Measure accuracy
        acc_train = mse.eval(feed_dict={X_freq: X_batch_freq, y: y_batch})
        acc_test = mse.eval(feed_dict={X_freq: X_test_freq, y: y_test})
        pmc = misclassification_rate.eval(feed_dict={X_freq: X_test_freq, y: y_test})
        print(epoch, "Train MSE:", acc_train, "Test MSE:", acc_test, "pmc:", pmc)

        # Save periodically
        if SAVE and epoch % 5 == 0:
           save_path = saver.save(sess, "/tmp/model.ckpt")
           print("Model saved in file: %s" % save_path)

    # Save the variables to disk
    if SAVE:
        save_path = saver.save(sess, "/tmp/model.ckpt")
        print("Model saved in file: %s" % save_path)