import numpy as np
import os
import warnings


def prepare_data(test_ratio):

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

    y_tol = np.empty((1, 10))
    for name in sorted(os.listdir('unnormalized data')):
        if name.endswith('.csv'):
            y_tol = np.vstack((y_tol, np.genfromtxt('unnormalized data/' + name, delimiter=',', dtype=float)))
            print "Reading y data file " + name
    y_tol = y_tol[1:, :]

    y_tol[:, ::2] = (y_tol[:, ::2] - (-134.35))/(-61.65 - (-134.35))
    y_tol[:, 1::2] = (y_tol[:, 1::2] - 19.36)/(48.90 - 19.36)

    y_tol = np.clip(y_tol, 0, 1)

    y_idx = int(test_ratio * y_tol.shape[0])
    y_test = y_tol[:y_idx, 2:8]
    y_train = y_tol[y_idx:, 2:8]

    np.save('y_train', y_train)
    np.save('y_test', y_test)

    if x_idx != y_idx:
        warnings.warn("The size of dataset X and dataset Y are not equal. Please check for possible reasons.")
    else:
        print "Training set is " + str(x_train.shape[2]) + ". Testing set is " + str(x_test.shape[2]) + "."
        print 'Done.'


if __name__ == '__main__':
    prepare_data(test_ratio=0.1)
