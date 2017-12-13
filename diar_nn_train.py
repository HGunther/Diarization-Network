from diar_nn_declare import *
import tensorflow as tf
import Chunks

# TODO find a way to store the current step number for iterative training

def fresh_train(train_list, test_list, chunk_size_ms, batch_size, num_epochs):
    train_data = Chunks(train_list, chunk_size_ms)
    num_batches = train_data.get_num_batches(batch_size)
    with tf.session() as sess:
        for epoch in range(num_epochs):
            for batch_index in range(num_batches):
                step = epoch * batch_size + batch_index
                X_batch, y_batch = train_data.get_rand_batch(batch_size)
                X_batch_freq = get_freqs(X_batch)
                if i%10 == 0:
                    #TODO summary info to print out to tensorboard
                    pass
                # TODO handle the test data to print summary to console
                sess.run(training_op, feed_dict={X: X_batch, X_freq: X_batch_freq, y: y_batch})
        # TODO save the weights periodically and at end of training


def cont_train(train_list, test_list, chunk_size_ms, batch_size, num_epochs):
    train_data = Chunks(train_list, chunk_size_ms)
    num_batches = train_data.get_num_batches(batch_size)
    with tf.session() as sess:
        # TODO RESTORE THE WEIGHTS
        for epoch in range(num_epochs):
            for batch_index in range(num_batches):
                step = epoch * batch_size + batch_index
                X_batch, y_batch = train_data.get_rand_batch(batch_size)
                X_batch_freq = get_freqs(X_batch)
                if i%10 == 0:
                    #TODO summary info to print out to tensorboard
                    pass
                # TODO handle the test data to print summary to console
                sess.run(training_op, feed_dict={X: X_batch, X_freq: X_batch_freq, y: y_batch})
        # TODO save the weights periodically and at end of training
