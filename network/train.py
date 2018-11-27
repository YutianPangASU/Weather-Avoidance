import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from train_test_separate import prepare_data


class cnn_model(object):

    def __init__(self, cfg, save_dir):

        self.batch_size = cfg['batch_size']
        self.epoch = cfg['epoch']
        self.lr = cfg['lr']
        self.conv1_channel = cfg['conv1_channel']

        self.x_train = np.load('x_train.npy')
        self.y_train = np.load('y_train.npy')
        self.x_test = np.load('x_test.npy')
        self.y_test = np.load('y_test.npy')
        self.path = save_dir

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

        # cross entropy mae loss
        # self.loss = tf.losses.absolute_difference(labels=self.y, predictions=self.y_out)
        self.l1 = tf.reduce_mean(tf.sqrt(tf.square(self.y[:,0]-self.y_out[:,0]) + tf.square(self.y[:,1]-self.y_out[:,1])))
        self.l2 = tf.reduce_mean(tf.sqrt(tf.square(self.y[:,2]-self.y_out[:,2]) + tf.square(self.y[:,3]-self.y_out[:,3])))
        self.l3 = tf.reduce_mean(tf.sqrt(tf.square(self.y[:,4]-self.y_out[:,4]) + tf.square(self.y[:,5]-self.y_out[:,5])))
        self.loss = tf.reduce_mean((self.l1 + self.l2 + self.l3)/3)

    def train_model(self):

        self.train_loss_tol = []
        self.test_loss_tol = []

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

                    [_, loss_train] = sess.run([train_step, self.loss], feed_dict={self.x: x_train_batch, self.y: y_train_batch, self.keep_prob: 0.5})

                    test_l1, test_l2, test_l3 = sess.run([self.l1, self.l2, self.l3], feed_dict={self.x: self.x_test, self.y: self.y_test, self.keep_prob: 1.0})

                    print "Epoch: {} Batch: {} Train Loss: {} Test Loss: {} {} {}".format(j+1, i+1, loss_train, test_l1, test_l2, test_l3)

                self.train_loss_tol = np.append(self.train_loss_tol, loss_train)
                self.test_loss_tol = np.append(self.test_loss_tol, [test_l1, test_l2, test_l3])

            save_path = tf.train.Saver().save(sess, self.path + '/model.ckpt')
            print("Model saved in path: %s" % save_path)
        sess.close()

    def draw_loss(self):

        xx = np.array(range(len(self.train_loss_tol))) + 1
        loss_test = np.reshape(self.test_loss_tol, (-1, 3))

        plt.plot(xx, self.train_loss_tol, 'k-')
        plt.plot(xx, loss_test[:, 0])
        plt.plot(xx, loss_test[:, 1])
        plt.plot(xx, loss_test[:, 2])
        plt.legend(["Train", "Test Loss 1", "Test Loss 2", "Test Loss 3"])
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title('Loss vs. Epochs')
        #plt.show()

        plt.savefig(self.path + '/Training_' + str(self.epoch) + '_epochs')


if __name__ == '__main__':

    cfg = {'lr': 0.15,
           'epoch': 400,
           'batch_size': 64,
           'conv1_channel': 32
           }

    save_dir = './Batch_' + str(cfg['batch_size']) + '_epoch_' + str(cfg['epoch']) + '_lr_' + str(cfg['lr'])

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        for the_file in os.listdir(save_dir):
            file_path = os.path.join(save_dir, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)

    # prepare_data(test_ratio=0.05)

    fun = cnn_model(cfg, save_dir)
    fun.build_graph()
    fun.train_model()
    fun.draw_loss()
