# For debugging
import sys
old_tr = sys.gettrace()
sys.settrace(None)
sys.settrace(old_tr)

# To disable warning that building TF from source will make it faster.
# For more information see:
# https://www.tensorflow.org/install/install_sources
# https://stackoverflow.com/questions/41293077/how-to-compile-tensorflow-with-sse4-2-and-avx-instructions
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from scipy.fftpack import rfft, fft
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from math import ceil
from Chunks import Chunks

# Info for TensorBoard
from datetime import datetime
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
ROOT_LOGDIR = "tf_logs"
LOGDIR = "{}/run-{}/".format(ROOT_LOGDIR, now)

# TODO Measuracy test classification accuracy
# TODO Set learning rate/step size
# TODO Change Kernal sizes
# TODO Dropout learning

# Constants
CHUNCK_SIZE_MS = 250 # Milliseconds, not megaseconds
NUM_CHANNELS = 2
SAMP_RATE_S = 44100 # Vals / s (Hz)
SAMP_RATE_MS = SAMP_RATE_S / 1000 # vals / ms (kHz)
NUM_SAMPS_IN_CHUNCK = int(CHUNCK_SIZE_MS * SAMP_RATE_MS)
NUM_INPUTS = NUM_SAMPS_IN_CHUNCK
NUM_OUTPUTS = 2

# For sanity checks, assert that shape1==shape2 at each index in indices
def assert_eq_shapes(shape1, shape2, indices):
    """Docstring"""
    for i in indices:
        errmsg = 'Index ' + str(i) + ': ' + str(shape1[i]) + ' vs ' + str(shape2[i])
        assert shape1[i] == shape2[i], errmsg

# Input Layer
with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=[None, NUM_INPUTS, NUM_CHANNELS, 1], name="X")
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

    conv1 = tf.layers.conv2d(X, filters=conv1_fmaps,
                                kernel_size=conv1_ksize,
                                strides=conv1_stride,
                                padding=conv1_pad,
                                activation=tf.nn.relu,
                                name="conv1")

    conv1_output_shape = [-1, ceil(NUM_INPUTS / conv1_time_stride), ceil(NUM_CHANNELS / conv1_channel_stride), conv1_fmaps]
    assert_eq_shapes(conv1_output_shape, conv1.get_shape(), (1,2,3))

    conv2 = tf.layers.conv2d(conv1, filters=conv2_fmaps,
                                    kernel_size=conv2_ksize,
                                    strides=conv2_stride,
                                    padding=conv2_pad,
                                    activation=tf.nn.relu,
                                    name="conv2")

    conv2_output_shape = [-1, ceil(conv1_output_shape[1] / conv2_time_stride), ceil(conv1_output_shape[2] / conv2_channel_stride), conv2_fmaps]
    assert_eq_shapes(conv2_output_shape, conv2.get_shape(), (1,2,3))

# Avg Pooling layer
with tf.name_scope("pool3"):
    pool3 = tf.nn.avg_pool(conv2, ksize=[1, 2, 1, 1],
                                strides=[1, 2, 1, 1],
                                padding="VALID")

    pool3_output_shape = [-1, conv2_output_shape[1] // 2, conv2_output_shape[2], conv2_fmaps]
    assert_eq_shapes(pool3_output_shape, pool3.get_shape(), (1,2,3))

    pool3_flat = tf.reshape(pool3, shape=[-1, conv2_fmaps * pool3_output_shape[1] * pool3_output_shape[2]])

# Fully connected layer
with tf.name_scope("fc1"):
    # Number of nodes in fully connected layer
    n_fc1 = 10
    fc1 = tf.layers.dense(pool3_flat, n_fc1, activation=tf.nn.relu, name="fc1")

# Output Layer
with tf.name_scope("output"):
    logits = tf.layers.dense(fc1, NUM_OUTPUTS, name="output")
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

with tf.name_scope("tensorboard"):
    mse_summary = tf.summary.scalar('MSE', mse)
    file_write = tf.summary.FileWriter(LOGDIR, tf.get_default_graph())

def debug():
    """Prints debug information"""
    print('X: ', X)
    print('y: ', y)
    print('conv1: ', conv1)
    print('conv2: ', conv2)
    print('pool3: ', pool3)
    print('pool3flat: ', pool3_flat)
    print('fc1: ', fc1)
    print('logits: ', logits)
    print('Yprob: ', Y_prob)

NUM_EPOCHS = 20
EPOCH_SIZE = 10
SAVE = False
RESTORE = False

with tf.Session() as sess:
    init.run()

    # Restore variables from disk.
    if RESTORE:
        saver.restore(sess, "/tmp/model.ckpt")
        print("Model restored.")

    # Prints the structure of the network one layer at a time
    debug()

    train_data = Chunks(['HS_D36', 'HS_D37'], CHUNCK_SIZE_MS)

    test_data = Chunks(['HS_D35'], CHUNCK_SIZE_MS)

    # print('\n*****Testing the net (Pre training)*****')
    # for i in range(5):
    #     X_batch, y_batch = train_data.get_rand_batch(EPOCH_SIZE)
    #     ev = Y_prob.eval(feed_dict={X: X_batch, y: y_batch})
    #     batch_mse = mse.eval(feed_dict={X: X_batch, y: y_batch})
    #     print(ev, batch_mse)

    print('\n*****Training the net*****')
    for epoch in range(NUM_EPOCHS):
        for i in range(EPOCH_SIZE):
            # Get data
            X_batch, y_batch = train_data.get_rand_batch(EPOCH_SIZE)

            # Log accuracy for Tensorboard reports
            if i % 10 == 0:
                step = epoch * EPOCH_SIZE + i
                summary_str = mse_summary.eval(feed_dict={X:X_batch,y:y_batch})
                file_write.add_summary(summary_str, step)
            
            # Train
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

        # Measure accuracy
        acc_train = mse.eval(feed_dict={X: X_batch, y: y_batch})
        X_test, y_test = test_data.get_rand_batch(EPOCH_SIZE)
        acc_test = mse.eval(feed_dict={X: X_test, y: y_test})
        # Percent Mis-classified
        pmc = misclassification_rate.eval(feed_dict={X: X_test, y: y_test})
        print(epoch, "Train MSE:", acc_train, "Test MSE:", acc_test, "pmc:", pmc)

        # Save periodically in case of crashes and @!$#% windows updates
        if SAVE and epoch % 5 == 0:
           save_path = saver.save(sess, "/tmp/model.ckpt")
           print("Model saved in file: %s" % save_path)

    # print('\n*****Testing the net (Post training)*****')
    # for i in range(2):
    #     X_batch, y_batch = train_data.get_rand_batch(EPOCH_SIZE)
    #     ev = Y_prob.eval(feed_dict={X: X_batch, y: y_batch})
    #     batch_mse = mse.eval(feed_dict={X: X_batch, y: y_batch})
    #     print(ev, batch_mse)
    

    # Save the variables to disk
    if SAVE:
        save_path = saver.save(sess, "/tmp/model.ckpt")
        print("Model saved in file: %s" % save_path)
