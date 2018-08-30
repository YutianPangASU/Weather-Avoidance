#import keras as ks
#import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts
import time
import scipy.io as io

def x_data(plot_range, mean, cov):
    class Generate_Weather_Data(object):
        def __init__(self, cfg):
            """define parameters to generate the weather data"""
            self.size = cfg['size'] # plot size
            self.temperature_center = cfg['temperature_center']  # center for temperature layer
            self.temperature_cov = cfg['temperature_cov']  # covariance for temperature layer
            self.pressure_center = cfg['pressure_center']  # center for pressure layer
            self.pressure_cov = cfg['pressure_cov']  # covariance for pressure layer
            self.humidity_center = cfg['humidity_center']  # center for humidity layer
            self.humidity_cov = cfg['humidity_cov']  # covariance for humidity layer
            self.mesh = cfg['mesh']  # mesh size
            self.traj_grid = cfg['traj_grid'] # grid size of trajectory

        def temp(self):
            """return multivariate gaussian temperature distribution"""
            self.x, self.y = np.mgrid[0:self.size:self.mesh, 0:self.size:self.mesh]
            pos = np.empty(self.x.shape + (2,))
            pos[:, :, 0] = self.x
            pos[:, :, 1] = self.y
            z = sts.multivariate_normal(self.temperature_center, self.temperature_cov)
            self.T = 1e2*z.pdf(pos)
            self.T[np.abs(self.T) < 1e-3] = 0

        def pressure(self):
            """return multivariate gaussian pressure distribution"""
            self.x, self.y = np.mgrid[0:self.size:self.mesh, 0:self.size:self.mesh]
            pos = np.empty(self.x.shape + (2,))
            pos[:, :, 0] = self.x
            pos[:, :, 1] = self.y
            z = sts.multivariate_normal(self.pressure_center, self.pressure_cov)
            self.P = 1e2*z.pdf(pos)
            self.P[np.abs(self.P) < 1e-3] = 0

        def humidity(self):
            """return multivariate gaussian pressure distribution"""
            self.x, self.y = np.mgrid[0:self.size:self.mesh, 0:self.size:self.mesh]
            pos = np.empty(self.x.shape + (2,))
            pos[:, :, 0] = self.x
            pos[:, :, 1] = self.y
            z = sts.multivariate_normal(self.humidity_center, self.humidity_cov)
            self.H = 1e2*z.pdf(pos)
            self.H[np.abs(self.H) < 1e-3] = 0

        def stack(self):
            self.weather = np.stack([self.T, self.P, self.H], axis=2)

        def build_trajectory(self):
            self.traj = self.weather[:, :, 0] # only take the first weather parameter for now
            self.traj[self.traj > 0] = 1
            np.fill_diagonal(self.traj, 1)
            #np.fill_diagonal(np.fliplr(self.traj), 1)
            trajectory = plt.figure()
            plt.imshow(1 - self.traj, cmap='gray', origin='lower', interpolation='nearest')
            trajectory.savefig('trajectory' + str(time.time()) + '.png')
            #plt.show()
            self.traj = np.fliplr(self.traj)

        def trajectory(self):
            """find the coordinate of single auxiliary waypoint"""
            # aux = [self.temperature_center[0] - np.max(np.asarray([self.temperature_cov[0][0], self.temperature_cov[1][1]])),
            #        self.temperature_center[1] + np.max(np.asarray([self.temperature_cov[0][0], self.temperature_cov[1][1]]))]
            # idx = np.zeros([len(self.x), 2], dtype='int')
            # for i in range(int(aux[1]+0.5)):
            #     idx[i, :] = [i, int(round(i+aux[1]/aux[0]+0.5))]
            # for i in range(int(aux[1]+0.5), 100):
            #     idx[i, :] = [i, int(round(i - aux[1] / aux[0] + 0.5))]
            #
            # traj = np.zeros([len(self.x), len(self.y)])
            # for i in range(len(idx)):
            #     traj[idx[i, 0], idx[i, 1]] = 1
            # print(idx)

            # find the coordinate of auxiliary points
            aux = [self.temperature_center[0] - np.max(np.asarray([self.temperature_cov[0][0], self.temperature_cov[1][1]])),
                   self.temperature_center[1] + np.max(np.asarray([self.temperature_cov[0][0], self.temperature_cov[1][1]]))]
            # x coordinate of the trajectory
            traj_x = np.unique(np.asarray([np.linspace(0, aux[0], self.traj_grid), np.linspace(aux[0], 100, self.traj_grid)]))
            # y coordinate of the trajectory
            k1 = aux[1]/aux[0]
            k2 = (100 - aux[1])/(100 - aux[0])
            traj_y_1 = np.asarray([k1 * traj_x[0:self.traj_grid]])
            traj_y_2 = np.asarray([k2 * (traj_x[self.traj_grid:10] - aux[0]) + aux[1]])
            traj_y = np.unique(np.hstack((traj_y_1, traj_y_2)))
            # stack x and y together
            self.traj = np.vstack((traj_x, traj_y))

        def plot_trajectory(self):
            plt.figure()
            plt.plot(self.traj[0, :], self.traj[1, :], 'b*-')
            plt.axis([0, 100, 0, 100])
            plt.show()

        def savefig(self):
            temp = plt.figure(1)
            plt.contourf(self.x, self.y, self.weather[:, :, 0])
            temp.savefig('temperature' + str(time.time()) + '.png')

            pressure = plt.figure(2)
            plt.contourf(self.x, self.y, self.weather[:, :, 1])
            pressure.savefig('pressure' + str(time.time()) + '.png')

            humidity = plt.figure(3)
            plt.contourf(self.x, self.y, self.weather[:, :, 2])
            humidity.savefig('humidity' + str(time.time()) + '.png')

    if __name__ == '__main__':

        cfg = {'size': plot_range,
               'temperature_center': mean[0] * np.ones([2, ]) + np.asarray([np.random.random(), np.random.random()], dtype=np.float64),
               'temperature_cov': [[plot_range/10, cov[0]], [cov[0], plot_range/10]],
               'pressure_center': mean[1] * np.ones([2, ]) + np.asarray([np.random.random(), np.random.random()], dtype=np.float64),
               'pressure_cov': [[plot_range/10, cov[1]], [cov[1], plot_range/10]],
               'humidity_center': mean[2] * np.ones([2, ]) + np.asarray([np.random.random(), np.random.random()], dtype=np.float64),
               'humidity_cov': [[plot_range/10, cov[2]], [cov[2], plot_range/10]],
               'mesh': 1,  # [0:size:mesh]
               'traj_grid': 5  # grid distance of trajectory grid
               #'visualization layer number': 2,  # 0 is temp, 1 is pressure, 2 is humidity
               }

        get_data = Generate_Weather_Data(cfg)
        get_data.temp()
        get_data.pressure()
        get_data.humidity()
        get_data.stack()
        get_data.savefig()
        #get_data.build_trajectory()  # grid trajectory
        get_data.trajectory()  # coordinates trajectory
        #get_data.plot_trajectory()  # visualization of trajectory

    return get_data.weather, get_data.traj


"""define parameters"""
number = 1  # number of plots
plot_range = 100  # plot range
traj_grid = 5
Data = np.empty([plot_range, plot_range, 3, number])
Trajectory = np.empty([2, 2*traj_grid-1, number])

for i in range(number):
    """random generated mean and covariance matrix for weather distributions"""
    #mean = np.asarray([plot_range * np.random.random(), plot_range * np.random.random(), plot_range * np.random.random()])
    mean = plot_range * np.random.uniform(low=0.2, high=0.8, size=(3,))
    cov = np.asarray([plot_range * 0.1 * np.random.random(), plot_range * 0.1 * np.random.random(), plot_range * 0.1 * np.random.random()])
    Data[:, :, :, i], Trajectory[:, :, i] = x_data(plot_range, mean, cov)  # weather data as x_train restored in Data variable


# save data for matlab fast marching
io.savemat('Data.mat', {'Data': np.squeeze(Data)})

# print(np.min(Data[:, :, 1, 1]))


'''def split_data(X, Y, test_size):
    x_test = X[:, :, :, :test_size]
    x_train = X[:, :, :, test_size:]
    y_test = Y[:, :, :test_size]
    y_train = Y[:, :, test_size:]
    return x_train, x_test, y_train, y_test


# test train split
x_train, x_test, y_train, y_test = split_data(Data, Trajectory, int(number*0.3))

# save numpy array into .npz file
np.savez_compressed('02272018_1000_XY', x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)'''

