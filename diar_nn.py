from diar_nn_declare import *
import tensorflow as tf

def evaluate(chunk):
    with tf.session() as sess:
        # TODO restore network weights
        chunk_freq = get_freqs(chunk)
        chunk_val = Y_prob.eval(feed_dict={X: chunk, X_freq: chunk_freq})
        return chunk_val
