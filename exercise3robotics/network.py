import tensorflow as tf
import sys
import time
import random

from utils import Options

opt = Options()

class Network:
    def __init__(self, device, path):
        """
        Define network graph.
        """
        self.device = device
        self.path = path
        with tf.device(self.device):
            # input: arbitrary batch size, shape
            x = tf.placeholder(tf.float32, shape=[None, opt.state_siz * opt.hist_len])
            # output: arbitrary batch size, act_num classes
            y = tf.placeholder(tf.float32, shape=[None, opt.act_num])

            def weight_matrix(shape):
              initial = tf.truncated_normal(shape, stddev=0.1)
              return tf.Variable(initial)

            def bias_vector(shape):
              initial = tf.constant(0.1, shape=shape)
              return tf.Variable(initial)

            def conv2d(x, W):
              return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

            def max_pool_2x2(x):
              return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            x_length = opt.pob_siz * opt.cub_siz
            hist_len = opt.hist_len

            x_image = tf.reshape(x, [-1, x_length, x_length, hist_len])

            # first convolutional layer
            W_conv1 = weight_matrix([3, 3, hist_len, opt.num_filters])  # 3x3 filters of depth history
            b_conv1 = bias_vector([opt.num_filters])

            out_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

            # first max_pool
            out_pool1 = max_pool_2x2(out_conv1)

            # second convolutional layer
            W_conv2 = weight_matrix([3, 3, opt.num_filters, opt.num_filters])  # 3x3 filters of depth num_filters
            b_conv2 = bias_vector([opt.num_filters])

            out_conv2 = tf.nn.relu(conv2d(out_pool1, W_conv2) + b_conv2)

            # second max_pool
            out_pool2 = max_pool_2x2(out_conv2)
            flat_shape = int(out_pool2.shape[1]*out_pool2.shape[2]*out_pool2.shape[3])
            out_pool2_flat = tf.reshape(out_pool2, [-1, flat_shape])  # reshape to flat vector

            # first fully conected layer
            W_fcon1 = weight_matrix([flat_shape, opt.num_units_linear_layer])
            b_fcon1 = bias_vector([opt.num_units_linear_layer])

            out_fcon1 = tf.nn.relu(tf.matmul(out_pool2_flat, W_fcon1) + b_fcon1)

            # second fully conected layer
            W_fcon2 = weight_matrix([opt.num_units_linear_layer, opt.act_num])  # act_num units
            b_fcon2 = bias_vector([opt.act_num])

            y_pred = tf.nn.relu(tf.matmul(out_fcon1, W_fcon2) + b_fcon2)

            self.y_pred = y_pred
            self.x = x
            self.y = y

            # Add operations to later save the parameters
            self.saver = tf.train.Saver()

    def train(self, Dtrain, Dval, num_epochs, learning_rate, batch_size):
        """
        Train the network and safe parameters in file.
        TODO: param description
        Dtrain[0] is x and Dtrain[1] is y
        """
        print("... training")
        with tf.device(self.device):
            # Define training method (cross_entropy, gradient descent)
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.y_pred))
            train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

            # Define performance validation
            correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_pred, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            # Actual training
            num_samples = len(Dtrain[0])
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                for epoch in range(num_epochs):
                    print("Epoch %i" % epoch)
                    # Shuffle
                    combined = list(zip(Dtrain[0], Dtrain[1]))
                    random.shuffle(combined)
                    Dtrain[0][:], Dtrain[1][:] = zip(*combined)

                    # Validate accuracy using Dval
                    acc = sess.run(accuracy, feed_dict={self.x: Dval[0], self.y: Dval[1]})
                    print("Validation accuracy: %f" % acc)

                    # Perform training step for all batches
                    for i in range(0, num_samples, batch_size):
                        batch = (Dtrain[0][i : batch_size + i], Dtrain[1][i : batch_size + i])
                        sess.run(train_step, feed_dict={self.x: batch[0], self.y: batch[1]})

                print("saving...")
                save_path = self.saver.save(sess, self.path)
                print("Model saved in file: %s" % save_path)

    def compute_accuracy(self, data):
        """
        Computes accuracy on given data and returns it.
        TODO: param/ return description
        """
        with tf.device(self.device):
            # Define accuracy
            correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_pred, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            acc = 0. # stores return value
            with tf.Session() as sess:
                self.saver.restore(sess, self.path)
                acc = sess.run(accuracy, feed_dict={self.x: data[0], self.y: data[1]})

            return acc

    def predict(self, x):
        """
        Predict single input x, does not return y_pred but argmax(y_pred), i.e. most likely class
        TODO: param/ return description
        """
        class_pred = -1 # stores the return value
        with tf.device(self.device):
            with tf.Session() as sess:
              self.saver.restore(sess, self.path)
              # expects array of input data, hence [x]
              class_pred = sess.run(tf.argmax(self.y_pred, 1), feed_dict={self.x: [x]})
        return class_pred[0]  # we only predict one input
