import numpy as np
import os
import warnings


def prepare_data(test_ratio):

    # process x
    x_tol = np.empty((100, 100), dtype=float)
    for name in sorted(os.listdir('unnormalized data')):
        if name.endswith('.npy'):
            print "Reading x data file " + name
            x_tol = np.dstack((x_tol, np.load('unnormalized data/' + name)))
    x_tol = x_tol[:, :, 1:]

    # arrange dimensions for tensorflow
    x_tol = np.expand_dims(np.transpose(x_tol, (2, 0, 1)), axis=3)

    # normalization: the critical value used for x_tol is 59000
    x_tol = x_tol/np.amax(x_tol)

    x_idx = int(test_ratio * x_tol.shape[0])
    x_test = x_tol[:x_idx, :, :, :]
    x_train = x_tol[x_idx:, :, :, :]

    np.save('x_train', x_train)
    np.save('x_test', x_test)

    # process y
    y_tol = np.empty((1, 10))
    for name in sorted(os.listdir('unnormalized data')):
        if name.endswith('.csv'):
            y_tol = np.vstack((y_tol, np.genfromtxt('unnormalized data/' + name, delimiter=',', dtype=float)))
            print "Reading y data file " + name
    y_tol = y_tol[1:, :]
    y_idx = int(test_ratio * y_tol.shape[0])

    y_test_range = np.concatenate((y_tol[:y_idx, 0:2], y_tol[:y_idx, 8:10]), axis=1) # save range for model performance evaluation
    np.save('plot_range', y_test_range)

    y_tol[:, ::2] = (y_tol[:, ::2] - (-134.35))/(-61.65 - (-134.35))
    y_tol[:, 1::2] = (y_tol[:, 1::2] - 19.36)/(48.90 - 19.36)

    y_tol = np.clip(y_tol, 0, 1)

    y_idx = int(test_ratio * y_tol.shape[0])
    y_test = y_tol[:y_idx, 2:8]
    y_train = y_tol[y_idx:, 2:8]

    np.save('y_train', y_train)
    np.save('y_test', y_test)

    # process other information
    other_tol = np.empty((1, 12))
    for name in sorted(os.listdir('start_end')):
        if name.endswith('.csv'):
            other_tol = np.vstack((other_tol, np.genfromtxt('start_end/' + name, delimiter=',', dtype=float)))
            print "Reading data file " + name
    other_tol = other_tol[1:, :]

    other_idx = int(test_ratio * other_tol.shape[0])
    other_test = np.array(other_tol[:other_idx, 4:], dtype='int')
    other_train = np.array(other_tol[other_idx:, 4:], dtype='int')
    np.save('test_range_idx', other_test)
    np.save('train_range_idx', other_train)

    # check
    if x_idx != y_idx or x_idx != other_idx:
        warnings.warn("The size of dataset X, dataset Y and other information dataset are not equal. Please check for possible reasons.")
    else:
        print "Training set is " + str(x_train.shape[0]) + ". Testing set is " + str(x_test.shape[0]) + "."
        print 'Done.'


if __name__ == '__main__':
    prepare_data(test_ratio=0.05)
