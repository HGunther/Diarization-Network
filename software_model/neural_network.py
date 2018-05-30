from math import ceil
import tensorflow as tf
from software_model.constants import NUM_SAMPS_IN_CHUNK, NUM_CHANNELS, CHUNK_SIZE_MS, LOGDIR
from software_model.network_data_preprocessor import NetworkDataPreprocessor
from software_model.network_data_preprocessor_for_training import NetworkDataPreprocessorForTraining

class NeuralNetwork:
    
    
    def __init__(self):
        self.NUM_INPUTS = NUM_SAMPS_IN_CHUNK
        self.NUM_INPUTS_FREQ = int(self.NUM_INPUTS // 2 + 1)
        self.NUM_OUTPUTS = 2
        self.__setup_network_in_tensorflow()
    
        
    def __setup_network_in_tensorflow(self):
        tf.reset_default_graph()
        
        with tf.name_scope("inputs"):
            self.X = tf.placeholder(tf.float32, shape=[None, self.NUM_INPUTS, NUM_CHANNELS, 1], name="X")
            self.X_fft = tf.placeholder(tf.float32, shape=[None, self.NUM_INPUTS_FREQ, NUM_CHANNELS, 1], name="X_fft")
            self.y = tf.placeholder(tf.int32, shape=[None, 2], name="y")

        # Group of convolutional layers
        with tf.name_scope("convclust1"):
            # ***************
            # Convolution layer for raw network
            # ***************
            # Convolutive Layers
        
            # Create convolutive maps
            # Number of convolutive maps in layer
            conv1_fmaps_raw = 32
            # Size of each kernel
            conv1_ksize_raw = [int((20 / CHUNK_SIZE_MS) * NUM_SAMPS_IN_CHUNK), NUM_CHANNELS]
            conv1_time_stride_raw = int((10 / CHUNK_SIZE_MS) * NUM_SAMPS_IN_CHUNK)
            conv1_channel_stride_raw = 1
            conv1_stride_raw = [conv1_time_stride_raw, conv1_channel_stride_raw]
            conv1_pad_raw = "SAME"
        
            # Number of convolutive maps in layer
            conv2_fmaps_raw = 64
            # Size of each kernel
            conv2_ksize_raw = [10, NUM_CHANNELS]
            conv2_time_stride_raw = 1
            conv2_channel_stride_raw = 1
            conv2_stride_raw = [conv2_time_stride_raw, conv2_channel_stride_raw]
            conv2_pad_raw = "SAME"
        
            conv1_raw = tf.layers.conv2d(self.X, filters=conv1_fmaps_raw, kernel_size=conv1_ksize_raw, strides=conv1_stride_raw, padding=conv1_pad_raw, activation=tf.nn.relu, name="conv1_raw")
        
            conv1_output_shape_raw = [-1, ceil(self.NUM_INPUTS / conv1_time_stride_raw), ceil(NUM_CHANNELS / conv1_channel_stride_raw), conv1_fmaps_raw]
        
            conv2_raw = tf.layers.conv2d(conv1_raw, filters=conv2_fmaps_raw, kernel_size=conv2_ksize_raw, strides=conv2_stride_raw, padding=conv2_pad_raw, activation=tf.nn.relu, name="conv2_raw")
        
            conv2_output_shape_raw = [-1, ceil(conv1_output_shape_raw[1] / conv2_time_stride_raw), ceil(conv1_output_shape_raw[2] / conv2_channel_stride_raw), conv2_fmaps_raw]
        
            # ***************
            # Convolution layer for freqency net
            # ***************
            # Create convolutive maps
            # Number of convolutive maps in layer
            conv1_fmaps_freq = 32
            # Size of each kernel
            conv1_ksize_freq = [15, NUM_CHANNELS]
            conv1_time_stride_freq = 2
            conv1_channel_stride_freq = 1
            conv1_stride_freq = [conv1_time_stride_freq, conv1_channel_stride_freq]
            conv1_pad_freq = "SAME"
        
            # Number of convolutive maps in layer
            conv2_freq_fmaps_freq = 64
            # Size of each kernel
            conv2_freq_ksize_freq = [10, NUM_CHANNELS]
            conv2_freq_time_stride_freq = 1
            conv2_freq_channel_stride_freq = 1
            conv2_freq_stride_freq = [conv2_freq_time_stride_freq, conv2_freq_channel_stride_freq]
            conv2_freq_pad_freq = "SAME"
        
            conv1_freq = tf.layers.conv2d(self.X_fft, filters=conv1_fmaps_freq, kernel_size=conv1_ksize_freq, strides=conv1_stride_freq, padding=conv1_pad_freq, activation=tf.nn.relu, name="conv1_freq")
        
            conv1_output_shape_freq = [-1, ceil(self.NUM_INPUTS_FREQ / conv1_time_stride_freq), ceil(NUM_CHANNELS / conv1_channel_stride_freq), conv1_fmaps_freq]
        
            conv2_freq = tf.layers.conv2d(conv1_freq, filters=conv2_freq_fmaps_freq, kernel_size=conv2_freq_ksize_freq, strides=conv2_freq_stride_freq, padding=conv2_freq_pad_freq, activation=tf.nn.relu, name="conv2_freq")
        
            conv2_output_shape_freq = [-1, ceil(conv1_output_shape_freq[1] / conv2_freq_time_stride_freq), ceil(conv1_output_shape_freq[2] / conv2_freq_channel_stride_freq), conv2_freq_fmaps_freq]
        
        # Avg Pooling layer
        with tf.name_scope("pool3"):
            # Pool3_raw for raw side of network
            pool3_raw = tf.nn.avg_pool(conv2_raw, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding="VALID")
        
            pool3_raw_output_shape = [-1, conv2_output_shape_raw[1] // 2, conv2_output_shape_raw[2], conv2_fmaps_raw]
        
            pool3_raw_flat = tf.reshape(pool3_raw, shape=[-1, conv2_fmaps_raw * pool3_raw_output_shape[1] * pool3_raw_output_shape[2]])
        
            # Pool3_freq for frequency side of network
            pool3_freq = tf.nn.avg_pool(conv2_freq, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding="VALID")
        
            pool3_freq_output_shape = [-1, conv2_output_shape_freq[1] // 2, conv2_output_shape_freq[2], conv2_freq_fmaps_freq]
        
            pool3_freq_flat = tf.reshape(pool3_freq, shape=[-1, conv2_freq_fmaps_freq * pool3_freq_output_shape[1] * pool3_freq_output_shape[2]])
        
        # Fully connected layer
        with tf.name_scope("fc"):
            # Fully connected layer for raw side of network
            fc_raw_num_nodes = 30
            fc_raw = tf.layers.dense(pool3_raw_flat, fc_raw_num_nodes, activation=tf.nn.relu, name="fc_raw")
        
            # Fully connected layer for frequency side of network
            fc_freq_num_nodes = 20
            fc_freq = tf.layers.dense(pool3_freq_flat, fc_freq_num_nodes, activation=tf.nn.relu, name="fc_freq")
        
            # Fully connected layer which takes both other fully connected layers as inputs
            fc_combine_num_nodes = 10
            fc_combine = tf.layers.dense(tf.concat([fc_raw, fc_freq], axis=1), fc_combine_num_nodes, activation=tf.nn.relu, name="fc_combine")
        
        
        # Output Layer
        with tf.name_scope("output"):
            logits = tf.layers.dense(fc_combine, self.NUM_OUTPUTS, name="output")
            self.Y_prob = tf.nn.sigmoid(logits, name="Y_prob")
        
        # Training nodes
        with tf.name_scope("train"):
            float_y = tf.cast(self.y, tf.float32)
            xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=float_y)
            loss = tf.reduce_mean(xentropy)
            optimizer = tf.train.AdamOptimizer()
            self.training_op = optimizer.minimize(loss)
        
        # Evaluate the network
        with tf.name_scope("eval"):
            error_measure = self.Y_prob - float_y
            self.mse = tf.reduce_mean(tf.square(error_measure), name='mse')
        
            error_count = tf.abs(self.y - tf.cast(tf.round(self.Y_prob), tf.int32))
            self.misclassification_rate = tf.reduce_mean(tf.cast(error_count, tf.float32), name='misclassification_rate')
        
        # Initialize the network
        with tf.name_scope("init_and_save"):
            self.variable_initizlizer = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
            
        # Tensorboard stuff
        with tf.name_scope("tensorboard"):
            self.mse_summary = tf.summary.scalar('MSE', self.mse)
            self.tb_train_writer = tf.summary.FileWriter(LOGDIR + "_train", tf.get_default_graph())
            self.tb_test_writer = tf.summary.FileWriter(LOGDIR + "_test", tf.get_default_graph())

    def evaluate_chunks(self, chunks, network_location):
        print("Preparing to run the network")
        with tf.Session() as sess:
            # Initialize Network Structure
            sess.run(tf.global_variables_initializer())
    
            # Restore variables from disk.
            self.saver.restore(sess, network_location)
    
            # Get data
            raw_data, fft_data = NetworkDataPreprocessor.to_tensorflow_readable_evaluation_input(chunks)
            
            # Return Prediction
            return self.Y_prob.eval(feed_dict={self.X: raw_data, self.X_fft: fft_data})

    def train_network(self, wav_file_names, out_model_location, in_model_location=None, num_epochs = 2000, epoch_size = 1, batch_size = 2000):
        
        import random as random
        random.shuffle(wav_file_names)
    
        training_files = wav_file_names[:int(0.8 * len(wav_file_names))]
        testing_files = wav_file_names[int(0.8 * len(wav_file_names)):]
    
        print("Reading in training data")
        train_data = NetworkDataPreprocessorForTraining(training_files)
        print("Reading in test data")
        test_data = NetworkDataPreprocessorForTraining(testing_files)
        
        print("Preparing to run the network")

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
    
            # Restore variables from disk.
            if in_model_location != None:
                self.saver.restore(sess, in_model_location)
                print("Model restored.")
    
            print('\n*****Pre-training accuracy*****')
            # Measure accuracy
            X_test, y_test = test_data.get_batch_of_random_annotated_chunks(batch_size)
    
            X_test_raw, X_test_fft = NetworkDataPreprocessor.to_tensorflow_readable_evaluation_input(X_test)
    
            mse_test, percent_misclassified = sess.run([self.mse, self.misclassification_rate], feed_dict={self.X: X_test_raw, self.X_fft: X_test_fft, self.y: y_test})
            # Percent Mis-classified
            print('Test MSE:', mse_test, 'percent_misclassified:', percent_misclassified)
    
            best_test_mse = mse_test
            print('\n*****Training the net*****')
    
            for epoch in range(num_epochs):
                for i in range(epoch_size):
                    # Get data
                    X_batch, y_batch = train_data.get_batch_of_random_annotated_chunks(batch_size)
                    X_batch_raw, X_batch_fft = NetworkDataPreprocessor.to_tensorflow_readable_evaluation_input(X_batch)
    
                    # Train
                    sess.run(self.training_op, feed_dict={self.X: X_batch_raw, self.X_fft: X_batch_fft, self.y: y_batch})
    
                # Measure accuracy based on testing data
                mse_train, train_summary = sess.run([self.mse, self.mse_summary], feed_dict={self.X: X_batch_raw, self.X_fft: X_batch_fft, self.y: y_batch})
                # Measure accuracy based on testing data
                mse_test, percent_misclassified, test_summary = sess.run([self.mse, self.misclassification_rate, self.mse_summary], feed_dict={self.X: X_test_raw, self.X_fft: X_test_fft, self.y: y_test})
    
                # Print Percent Mis-classified
                print("{:03d}  Train MSE: {:1.8f}  Test MSE: {:1.8f}  Percent misclassified: {:1.6f}".format(epoch, mse_train, mse_test, percent_misclassified))
    
                # Log accuracy for Tensorboard reports
                if True:  # i % 10 == 0:
                    step = epoch  # * epoch_size + i
                    self.tb_test_writer.add_summary(test_summary, step)
                    self.tb_train_writer.add_summary(train_summary, step)
    
                # Save periodically in case of crashes and @!$#% windows updates
                if mse_test < best_test_mse:  # SAVE and epoch % 2 == 0:
                    best_test_mse = mse_test
                    best_test_percent_misclassified = percent_misclassified
                    save_path = self.saver.save(sess, out_model_location)
                    print("* New lowest model! Saved as: %s" % save_path)
    
            # Save the variables to disk
            save_path = self.saver.save(sess, out_model_location)
            print("Model saved in file: %s" % save_path)
    
            print("The best model had a percent misclassified of", best_test_percent_misclassified)
            
            