#! /anaconda/bin/python3 python
# -*- coding: utf-8 -*-

"""
@Author: Yutian Pang
@Date: 2019-01-23

This Python script is able to,
1. Draw out the 2D history track points between two airports on top of a US map.
2. Plot the weather conditions during the flight given a specific flight call sign.
3. Make a gif using the plots generated from 2.

@Last Modified by: Yutian Pang
@Last Modified date: 2019-01-31
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
os.environ['PROJ_LIB'] = '/home/ypang6/anaconda3/share/proj'
from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset as NetCDFFile
import utils as utl


class draw_figure(object):

    def __init__(self, cfg):
        self.obj_dir = cfg['object_directory']
        self.file_list = sorted([x.split('.')[0] for x in os.listdir("{}/".format(self.obj_dir))])
        self.weather_dir = cfg['weather_directory']
        self.date = cfg['date']
        self.call_sign_to_draw = cfg['call_sign_to_draw']

    def plot2D(self):

        # create new figure, axes instances.
        fig = plt.figure()
        plt.title('Track Points from JFK to LAX')

        #ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

        # setup mercator map projection
        m = Basemap(llcrnrlon=-134.5, llcrnrlat=19.36, urcrnrlon=-61.5, urcrnrlat=48.90,
                    rsphere=(6378137.00, 6356752.3142), resolution='h', projection='merc', area_thresh=10000.)

        m.drawcoastlines()
        m.drawstates(linewidth=.25)
        m.drawcountries(linewidth=1)
        # m.fillcontinents()

        # draw parallels
        m.drawparallels(np.arange(0, 90, 15), labels=[1, 1, 0, 1])
        # draw meridians
        m.drawmeridians(np.arange(-180, 180, 30), labels=[1, 1, 0, 1])

        for i in range(len(self.file_list)):
            print("Loading file {}.csv".format(self.file_list[i]))
            self.track_points = np.genfromtxt('{}/{}.csv'.format(self.obj_dir, self.file_list[i]), delimiter=',', skip_header=1)

            # Convert latitude and longitude to coordinates X and Y
            x, y = m(self.track_points[:, 2], self.track_points[:, 1])

            m.plot(x, y, marker=None, color='r')

        plt.savefig('{}.png'.format(self.obj_dir))
        #plt.show()

    def draw_weather_contour(self):
        # load track data
        track = np.genfromtxt('{}/{}_{}.csv'.format(self.obj_dir, self.call_sign_to_draw, self.date),
                              delimiter=',', skip_header=1)

        # get flight plan str
        csv_name = 'flight_data_{}_{}_to_{}.csv'.format(self.date, self.obj_dir[-7:-4], self.obj_dir[-3:])
        with open(csv_name) as f:
            reader = csv.reader(f)
            for row in reader:
                if row[2] == self.call_sign_to_draw:
                    fp_str = row[4]
                    break

        # parse fp_string into array
        flight_plan = utl.fetch_from_web(fp_str)

        # get flight time info
        flight_start_time = utl.unixtime_to_datetime([track[0, 0]])[0]
        print("Flight {} Departured at {}".format(self.call_sign_to_draw, flight_start_time))

        flight_time = track[-1, 0] - track[0, 0]
        print("The total flight time is {} seconds/{} hours".format(int(flight_time), round(flight_time/3600),2))

        flight_end_time = utl.unixtime_to_datetime([track[-1, 0]])[0]
        print("Flight {} Arrived    at {}".format(self.call_sign_to_draw, flight_end_time))

        unix_time_seq = np.arange(track[0, 0], track[-1, 0], 150)

        # clear folder before run functions
        folder = './Plots'
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)

        for i in range(len(unix_time_seq)):
            print("Generating plot {}/{}".format(i+1, len(unix_time_seq)))
            pin, nearest_value = utl.get_weather_file(unix_time_seq[i])

            # load weather file
            filename = "ciws.EchoTop." + pin[:8] + "T" + str(pin[-6:-4]) + nearest_value + "Z"

            nc = NetCDFFile('{}/{}.nc'.format(self.weather_dir, filename))
            data = nc.variables['ECHO_TOP'][:]
            loncorners = nc.variables['x0'][:]
            latcorners = nc.variables['y0'][:]

            # create new figure, axes instances.
            fig = plt.figure()
            plt.title('{}'.format(filename))

            # setup mercator map projection
            m = Basemap(width=2559500*2, height=1759500*2, resolution='l', projection='laea', lat_ts=50, lat_0=38, lon_0=-98)

            m.drawcoastlines()
            m.drawstates(linewidth=.25)
            m.drawcountries(linewidth=1)
            # m.fillcontinents()

            # draw parallels
            m.drawparallels(np.arange(0, 90, 15), labels=[1, 1, 0, 1])
            # draw meridians
            m.drawmeridians(np.arange(-180, 180, 30), labels=[1, 1, 0, 1])

            ny = data.shape[2]
            nx = data.shape[3]
            lons, lats = m.makegrid(nx, ny)

            # import scipy.io
            # scipy.io.savemat('lats.mat', mdict={'arr': lats})
            # scipy.io.savemat('lons.mat', mdict={'arr': lons})

            # get lat/lons of ny by nx evenly space grid
            x, y = m(lons, lats)  # compute map proj coordinates

            data = data[0, 0, :, :].clip(min=0)

            # draw filled contours
            cs = m.contour(x, y, data)

            # Plot track points
            lon, lat = m(track[:, 2], track[:, 1])
            plot_1 = m.plot(lon, lat, marker=None, color='r')

            # plot flight plan
            lon_fp, lat_fp = m(flight_plan[:, 1], flight_plan[:, 0])
            plot_2 = m.plot(lon_fp, lat_fp, marker=None, color='b')

            # add color bar
            cbar = m.colorbar(cs, location='bottom', pad="5%")
            cbar.set_label('ET Unit')

            # add legend
            red_patch = mpatches.Patch(color='red', label='Track Points')
            blue_patch = mpatches.Patch(color='blue', label='Flight Plan')
            plt.legend(handles=[red_patch, blue_patch])

            #plt.show()
            plt.savefig('Plots/{}_{}.png'.format(self.call_sign_to_draw, str(pin[-6:-4]) + nearest_value))
            plt.close(fig)

    def make_gif(self):
        # make a gif
        import imageio

        filenames = sorted([x.split('.')[0] for x in os.listdir("Plots/")])
        with imageio.get_writer('{}.gif'.format(self.call_sign_to_draw), mode='I') as writer:
            for filename in filenames:
                image = imageio.imread("Plots/"+filename+".png")
                writer.append_data(image)
                print("Appending Frame {}".format(filename))


if __name__ == '__main__':

    cfg = {'departure_airport': 'JFK',
           'arrival_airport': 'LAX',
           'date': '20170407',
           'call_sign_to_draw': 'AAL293'}

    cfg['object_directory'] = "track_point_{}_{}2{}".format(cfg['date'], cfg['departure_airport'], cfg['arrival_airport'])
    cfg['weather_directory'] = '/mnt/data/Research/data/{}ET'.format(cfg['date'])

    fun = draw_figure(cfg)

    fun.plot2D()

    #fun.draw_weather_contour() # plots saved in '/Plots'

    #fun.make_gif() # make gif use the plots in '/Plots

