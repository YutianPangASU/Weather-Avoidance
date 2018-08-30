from netCDF4 import Dataset
import matplotlib.pyplot as plt
import os
import pyproj
from utils import *


class load_ET(object):

    def __init__(self, date):

        a = 2559500
        b = 1759500

        self.x = np.arange(-a, a+1000, 1000)  # longitude
        self.y = np.arange(-b, b+1000, 1000)  # latitude
        self.lon = np.zeros_like(self.x, dtype='float64')  # allocate space
        self.lat = np.zeros_like(self.y, dtype='float64')
        self.date = date

    def save_labels(self):

        # # handle = str('20170406EchoTop/ciws.EchoTop.20170406T000000Z.nc')
        # # data = Dataset(handle)
        #
        # # data = Dataset("20170406EchoTop/ciws.EchoTop.20170406T000000Z.nc")
        #
        # print data.file_format
        #
        # print data.dimensions.keys()
        #
        # print data.dimensions['time']
        #
        # print data.variables.keys()
        #
        # print data.dimensions['x0']  # get dimensions of a variable
        #
        # print data.variables['ECHO_TOP']  # get variable information
        #
        # print data.Conventions
        #
        # print data.variables['ECHO_TOP'].units   # check unit of your specified variable
        #
        #
        # #x = np.asarray(data.variables['x0'])  # projection x coordinate ??
        # #y = np.asarray(data.variables['y0'])

        # convert to WGS84
        p = pyproj.Proj("+proj=laea +lat_0=38 +lat_ts=60 +lon_0=-98 +k=90 +x_0=0 +y_0=0 +a=6370997 +b=6370997 +units=m +no_defs")

        # lon, lat = p(x, y, inverse=True)
        # p1 = pyproj.Proj(init='epsg:3857')
        # p2 = pyproj.Proj(init='epsg:4326')

        for i in range(len(self.x)):
            for j in range(len(self.y)):
                self.lon[i], self.lat[j] = p(self.x[i], self.y[j], inverse=True)
                #lon[i], lat[j] = pyproj.transform(p1, p2, x[i], y[j])

        # save lon and lat
        np.save('lon.npy', self.lon)
        np.save('lat.npy', self.lat)
        # io.savemat('longitude.mat', {'longitude': np.asarray(self.lon)})
        # io.savemat('latitude.mat', {'latitude': np.asarray(self.lat)})

    def load_labels(self):

        self.lon = np.load('lon.npy')
        self.lat = np.load('lat.npy')

    def save_pics(self):

        handle = sorted(os.listdir(self.date + "EchoTop"))

        print ('There is ' + repr(len(handle)) + ' data files')

        for i in range(len(handle)):
        #for i in range(10):
            data = Dataset(self.date + "EchoTop/" + handle[i])
            values = np.squeeze(data.variables['ECHO_TOP'])  # extract values

            # save EchoTop values and restore a 3d array
            #self.GY.append(values)

            plt.contourf(self.lon, self.lat, values)
            # plt.axes([self.lon.min(), self.lon.max(), self.lat.min(), self.lat.max()])
            # plt.show()

            plt.savefig('EchoTopPic/' + str('{date:%Y-%m-%d_%H:%M:%S}'.format(date=datetime.datetime.now())))
            print ('I\'m reading file ' + repr(i))

        #io.savemat('EchoTop_20170406_WholeDay.mat', {'EchoTop': np.asarray(self.GY)})  # save whole day values into mat file

    def plot_weather_contour(self, unix_time, call_sign):

        pin = datetime.datetime.utcfromtimestamp(int(float(unix_time))).strftime('%Y%m%d %H%M%S')  # time handle to check CIWS database
        array = np.asarray([0, 230, 500, 730,
                            1000, 1230, 1500, 1730,
                            2000, 2230, 2500, 2730,
                            3000, 3230, 3500, 3730,
                            4000, 4230, 4500, 4730,
                            5000, 5230, 5500, 5730])

        nearest_value = int(find_nearest_value(array, np.asarray([int(eliminate_zeros(pin[-4:]))])))  # find the closest time for downloading data from CIWS
        nearest_value = make_up_zeros(str(nearest_value))  # make up zeros for 0 230 500 730

        # find compared nc file
        data = Dataset(str(self.date) + "EchoTop/ciws.EchoTop." + pin[:8] + "T" + str(pin[-6:-4]) + nearest_value + "Z.nc")
        values = np.squeeze(data.variables['ECHO_TOP'])  # extract values
        plt.contourf(self.lon, self.lat, values)

        plt.savefig('EchoTopPic/' + str(call_sign) + ' ' + pin)

        # plt.show()


if __name__ == '__main__':

    a = 2559500
    b = 1759500
    date = 20170406
    #unix_time = 1491425432.000  # a wrong time
    unix_time = 1491450567.000  # a correct time
    call_sign = 'AAL717'

    fun = load_ET(date)
    # fun.save_labels()  # only need to run this function once
    fun.load_labels()
    # fun.save_pics()
    fun.plot_weather_contour(unix_time, call_sign)
