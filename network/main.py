import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from train_test_separate import prepare_data


class cnn_model(object):

    def __init__(self, cfg):

        self.batch_size = cfg['batch_size']
        self.epoch = cfg['epoch']
        self.lr = cfg['lr']
        self.conv1_channel = cfg['conv1_channel']

        self.x_train = np.load('x_train.npy')
        self.y_train = np.load('y_train.npy')
        self.x_test = np.load('x_test.npy')
        self.y_test = np.load('y_test.npy')

    def build_graph(self):

        self.x = tf.placeholder(tf.float32, [None, 100, 100, 1])
        self.y = tf.placeholder(tf.float32, [None, 6])

        self.keep_prob = tf.placeholder(tf.float32)

        conv1 = tf.layers.conv2d(self.x, self.conv1_channel, kernel_size=5, padding='same', activation=tf.nn.relu, name='conv1')
        pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=2, name='pool1')
        conv2 = tf.layers.conv2d(pool1, self.conv1_channel*2, kernel_size=5, padding='same', activation=tf.nn.relu, name='conv2')
        pool2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=2, name='pool2')
        conv3 = tf.layers.conv2d(pool2, self.conv1_channel*4, kernel_size=5, padding='same', activation=tf.nn.relu, name='conv3')
        pool3 = tf.layers.max_pooling2d(conv3, pool_size=[2, 2], strides=2, padding='same', name='pool3')

        pool3_flat = tf.reshape(pool3, [-1, 13*13*128])

        fc1 = tf.layers.dense(pool3_flat, 2048, activation=tf.nn.relu, name='fc1')
        fc2 = tf.layers.dense(fc1, 512, activation=tf.nn.relu, name='fc2')
        self.y_out = tf.layers.dense(fc2, 6, activation=tf.nn.sigmoid, name='output')

        # # conv1: 5x5 kernel, 32 channels, padding 2, stride 1.
        # w_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32*dd], stddev=0.1))
        # b_conv1 = tf.Variable(tf.constant(0.1, shape=[32*dd]))
        # h_conv1 = tf.nn.relu(tf.nn.conv2d(self.x, w_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
        #
        # # maxpool1 2x2, stride 2, pad 0.
        # h_pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        #
        # # conv2: 5x5 kernel, 64 channels, padding 2, stride 1.
        # w_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32*dd, 64*dd], stddev=0.1))
        # b_conv2 = tf.Variable(tf.constant(0.1, shape=[64*dd]))
        # h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, w_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
        #
        # # maxpool2 2x2, stride 2, pad 0.
        # h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        #
        # # conv3: 5x5 kernel, 128 channels, padding 2, stride 1.
        # w_conv3 = tf.Variable(tf.truncated_normal([5, 5, 64*dd, 128*dd], stddev=0.1))
        # b_conv3 = tf.Variable(tf.constant(0.1, shape=[128*dd]))
        # h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, w_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3)
        #
        # # maxpool3 3x3, stride 3, pad 1.
        # h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
        #
        # # conv4: 3x3 kernel, channel 64, stride 3, pad 0.
        # w_conv4 = tf.Variable(tf.truncated_normal([3, 3, 128*dd, 256*dd], stddev=0.1))
        # b_conv4 = tf.Variable(tf.constant(0.1, shape=[256*dd]))
        # h_conv4 = tf.nn.relu(tf.nn.conv2d(h_pool3, w_conv4, strides=[1, 3, 3, 1], padding='SAME') + b_conv4)
        #
        # # maxpool4 3x3, stride 3, pad 1.
        # h_pool4 = tf.nn.max_pool(h_conv4, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
        #
        # # fc1: 1024
        # w_fc1 = tf.Variable(tf.truncated_normal([9*9*256*dd, 1024], stddev=0.1))
        # b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
        # h_pool4_flat = tf.reshape(h_pool4, [-1, 9*9*256*dd])
        # h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, w_fc1) + b_fc1)
        #
        # # dropout
        # # h_fc1_dropout = tf.nn.dropout(h_fc1, self.keep_prob)
        #
        # # fc2: 6 outputs
        # w_fc2 = tf.Variable(tf.truncated_normal([1024, 6], stddev=0.1))
        # b_fc2 = tf.Variable(tf.constant(0.1, shape=[6]))
        # h_fc2 = tf.matmul(h_fc1, w_fc2) + b_fc2
        #
        # # sigmoid activation
        # self.y_out = tf.nn.sigmoid(h_fc2)

        # cross entropy mae loss
        # self.loss = tf.losses.absolute_difference(labels=self.y, predictions=self.y_out)
        self.loss = tf.reduce_mean((tf.sqrt(tf.square(self.y[:,0]-self.y_out[:,0]) + tf.square(self.y[:,1]-self.y_out[:,1])) +
                                    tf.sqrt(tf.square(self.y[:,2]-self.y_out[:,2]) + tf.square(self.y[:,3]-self.y_out[:,3])) +
                                    tf.sqrt(tf.square(self.y[:,4]-self.y_out[:,4]) + tf.square(self.y[:,5]-self.y_out[:,5])))/3)

    def run_model(self):

        self.loss_tol = []

        # SGD
        train_step = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)

        # batch number
        batch_num = int(self.x_train.shape[0]/self.batch_size)

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            for j in range(self.epoch):
                for i in range(batch_num):

                    x_train_batch = self.x_train[self.batch_size*i:self.batch_size*(i+1), :, :, :]
                    y_train_batch = self.y_train[self.batch_size*i:self.batch_size*(i+1), :]

                    #writer = tf.summary.FileWriter('./Training_' + str(cfg['epoch']) + '_' + str(cfg['lr']), sess.graph)
                    [_, loss] = sess.run([train_step, self.loss], feed_dict={self.x: x_train_batch, self.y: y_train_batch, self.keep_prob: 0.5})

                    print "Epoch: " + str(j+1) + " Batch: " + str(i+1) + " Loss: " + str(loss)

                self.loss_tol = np.append(self.loss_tol, loss)

            save_path = tf.train.Saver().save(sess, './Training_' + str(cfg['epoch']) + '_' + str(cfg['lr']) + '/epoch_' + str(self.epoch))
            print("Model saved in path: %s" % save_path)
        sess.close()

    def draw_loss(self):

        plt.plot(np.array(range(len(self.loss_tol)))+1, self.loss_tol, 'k-')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title('Loss vs. Epochs during training process')
        plt.savefig('./Training_' + str(cfg['epoch']) + '_' + str(cfg['lr']) + '/Training_' + str(self.epoch) + '_epochs')
        # plt.show()


if __name__ == '__main__':

    cfg = {'lr': 0.15,
           'epoch': 200,
           'batch_size': 64,
           'conv1_channel': 32}

    if not os.path.exists('./Training_' + str(cfg['epoch']) + '_' + str(cfg['lr'])):
        os.makedirs('./Training_' + str(cfg['epoch']) + '_' + str(cfg['lr']))

    # prepare_data(test_ratio=0.05)

    fun = cnn_model(cfg)
    fun.build_graph()
    fun.run_model()
    fun.draw_loss()
