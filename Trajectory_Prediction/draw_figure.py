#! /usr/bin/python2 python2
#-*- coding: utf-8 -*-

"""
@Author: Yutian Pang
@Date: 2019-01-17

This Python script is used to draw the csv file into one plot given directories.

@Last Modified by: Yutian Pang
@Last Modified date: 2019-01-22
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm
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

        # get flight time info
        flight_start_time = utl.unixtime_to_datetime([track[0, 0]])[0]
        print("Flight {} Departured at {}".format(self.call_sign_to_draw, flight_start_time))

        flight_end_time = utl.unixtime_to_datetime([track[-1, 0]])[0]
        print("Flight {} Arrived    at {}".format(self.call_sign_to_draw, flight_end_time))

        pin, nearest_value = utl.get_weather_file(track[0, 0])
        print(pin, nearest_value)
        pin, nearest_value = utl.get_weather_file(track[-1, 0])
        print(pin, nearest_value)

        # load weather file
        filename = "ciws.EchoTop." + pin[:8] + "T" + str(pin[-6:-4]) + nearest_value + "Z"

        nc = NetCDFFile('{}/{}.nc'.format(self.weather_dir, filename))
        data = nc.variables['ECHO_TOP'][:]
        loncorners = nc.variables['x0'][:]
        latcorners = nc.variables['y0'][:]

        # create new figure, axes instances.
        fig = plt.figure()
        plt.title('EchoTop')

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
        lons, lats = m.makegrid(nx, ny)  # get lat/lons of ny by nx evenly space grid
        x, y = m(lons, lats)  # compute map proj coordinates

        data = data[0, 0, :, :].clip(min=0)

        # draw filled contours
        #clevs = np.linspace(0, 60000, 13)
        #cs = m.contourf(x, y, data)
        cs = m.contour(x, y, data)

        # add color bar
        cbar = m.colorbar(cs, location='bottom', pad="5%")
        cbar.set_label('ET Unit')

        plt.savefig('Plots/{}.png'.format(self.obj_dir))


if __name__ == '__main__':

    cfg = {'object_directory': 'track_point_JFK2LAX',
           'date': '20170407',
           'call_sign_to_draw': 'AAL185'}
    cfg['weather_directory'] = '/mnt/data/sherlock/data/{}ET'.format(cfg['date'])

    fun = draw_figure(cfg)
    # fun.plot2D()
    fun.draw_weather_contour()
