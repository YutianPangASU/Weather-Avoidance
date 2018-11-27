import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import sys


class test_model(object):

    def __init__(self, dir):

        self.dir = dir
        self.conv1_channel = 32

    def inverse_normalization(self, tensor):

        delta_lat = (-61.65) - (-134.35)
        delta_lon = 48.90 - 19.36

        tensor[:, ::2] = tensor[:, ::2] * delta_lat + (-134.35)
        tensor[:, 1::2] = tensor[:, 1::2] * delta_lon + 19.36

        return tensor

    def load_graph(self):

        self.x = tf.placeholder(tf.float32, [None, 100, 100, 1])
        self.y = tf.placeholder(tf.float32, [None, 6])

        self.keep_prob = tf.placeholder(tf.float32)

        conv1 = tf.layers.conv2d(self.x, self.conv1_channel, kernel_size=5, padding='same', activation=tf.nn.relu,
                                 name='conv1')
        pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=2, name='pool1')
        conv2 = tf.layers.conv2d(pool1, self.conv1_channel * 2, kernel_size=5, padding='same',
                                 activation=tf.nn.relu, name='conv2')
        pool2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=2, name='pool2')
        conv3 = tf.layers.conv2d(pool2, self.conv1_channel * 4, kernel_size=5, padding='same',
                                 activation=tf.nn.relu, name='conv3')
        pool3 = tf.layers.max_pooling2d(conv3, pool_size=[2, 2], strides=2, padding='same', name='pool3')

        pool3_flat = tf.reshape(pool3, [-1, 13 * 13 * 128])

        fc1 = tf.layers.dense(pool3_flat, 2048, activation=tf.nn.relu, name='fc1')
        fc2 = tf.layers.dense(fc1, 512, activation=tf.nn.relu, name='fc2')
        self.y_out = tf.layers.dense(fc2, 6, activation=tf.nn.sigmoid, name='output')

        # cross entropy mae loss
        # self.loss = tf.losses.absolute_difference(labels=self.y, predictions=self.y_out)
        self.l1 = tf.reduce_mean(
            tf.sqrt(tf.square(self.y[:, 0] - self.y_out[:, 0]) + tf.square(self.y[:, 1] - self.y_out[:, 1])))
        self.l2 = tf.reduce_mean(
            tf.sqrt(tf.square(self.y[:, 2] - self.y_out[:, 2]) + tf.square(self.y[:, 3] - self.y_out[:, 3])))
        self.l3 = tf.reduce_mean(
            tf.sqrt(tf.square(self.y[:, 4] - self.y_out[:, 4]) + tf.square(self.y[:, 5] - self.y_out[:, 5])))
        self.loss = tf.reduce_mean((self.l1 + self.l2 + self.l3) / 3)

    def load_model(self):

        self.test_x = np.load('x_test.npy')
        self.test_y = np.load('y_test.npy')

        saver = tf.train.Saver()

        with tf.Session() as sess:

            # Initalize the variables
            sess.run(tf.global_variables_initializer())

            # Restore latest checkpoint
            saver.restore(sess, tf.train.latest_checkpoint('./' + self.dir + '/'))

            # To get the value (numpy array)
            self.output = sess.run(self.y_out, feed_dict={self.x: self.test_x, self.y: self.test_y})

        self.y_pred = self.inverse_normalization(self.output)
        self.y_test = self.inverse_normalization(self.test_y)

    def plot_results(self):

        folder = './plots'
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                # elif os.path.isdir(file_path): shutil.rmtree(file_path)  # this line will clear the directory as well
            except Exception as e:
                print(e)

        plot_range_idx = np.load('test_range_idx.npy')
        lat = np.load('lat.npy')
        lon = np.load('lon.npy')
        plot_range = np.load('plot_range.npy')
        sys.setrecursionlimit(int(1e5))

        for i in range(self.y_pred.shape[0]):
            plt.figure(i)

            # limit 1
            # plt.xlim(lon[plot_range_idx[i, 4]], lon[plot_range_idx[i, 6]])
            # plt.ylim(lat[plot_range_idx[i, 5]], lat[plot_range_idx[i, 7]])

            # limit 2
            # plt.xlim(plot_range[i, 0], plot_range[i, 2])
            # plt.ylim(plot_range[i, 1], plot_range[i, 3])

            x = np.linspace(plot_range[i, 0], plot_range[i, 2], 100)
            y = np.linspace(plot_range[i, 1], plot_range[i, 3], 100)
            plt.contourf(x, y, self.test_x[i, :, :, 0])

            plt.plot(np.insert(self.y_pred[i, ::2], [0, 3], [lon[plot_range_idx[i, 0]], lon[plot_range_idx[i, 2]]]),
                     np.insert(self.y_pred[i, 1::2], [0, 3], [lat[plot_range_idx[i, 1]], lat[plot_range_idx[i, 3]]]), 'ro--')

            plt.plot(np.insert(self.y_test[i, ::2], [0, 3], [lon[plot_range_idx[i, 0]], lon[plot_range_idx[i, 2]]]),
                     np.insert(self.y_test[i, 1::2], [0, 3], [lat[plot_range_idx[i, 1]], lat[plot_range_idx[i, 3]]]), 'bo--')

            plt.plot([lon[plot_range_idx[i, 0]], lon[plot_range_idx[i, 2]]],
                     [lat[plot_range_idx[i, 1]], lat[plot_range_idx[i, 3]]], 'ko-')

            plt.legend(['model prediction', 'original waypoints', 'original flight plan'])
            plt.title('Model Performance')
            #plt.show()
            plt.savefig('./plots/{}'.format(i))
            plt.clf()


if __name__ == '__main__':

    path = 'Batch_64_epoch_400_lr_0.15'

    fun = test_model(path)
    fun.load_graph()
    fun.load_model()
    fun.plot_results()

