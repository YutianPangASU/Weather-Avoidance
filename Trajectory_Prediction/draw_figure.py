#! /usr/bin/python2 python2
#-*- coding: utf-8 -*-

"""
@Author: Yutian Pang
@Date: 2019-01-17

This Python script is used to draw the csv file into one plot given directories.

@Last Modified by: Yutian Pang
@Last Modified date: 2019-01-17
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.mplot3d import Axes3D


class draw_figure(object):

    def __init__(self, cfg):
        self.obj_dir = cfg['object_directory']
        self.file_list = sorted([x.split('.')[0] for x in os.listdir("{}/".format(self.obj_dir))])

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
        m.fillcontinents()
        # draw parallels
        m.drawparallels(np.arange(0, 90, 15), labels=[1, 1, 0, 1])
        # draw meridians
        m.drawmeridians(np.arange(-180, 180, 30), labels=[1, 1, 0, 1])


        for i in range(len(self.file_list)):
            print("Loading file {}.csv".format(self.file_list[i]))
            self.track_points = np.genfromtxt('{}/{}.csv'.format(self.obj_dir, self.file_list[i]), delimiter=',',
                                          skip_header=1, )

            # Convert latitude and longitude to coordinates X and Y
            x, y = m(self.track_points[:, 2], self.track_points[:, 1])

            m.plot(x, y, marker=None, color='r')

        plt.savefig('{}.png'.format(self.obj_dir))
        #plt.show()


if __name__ == '__main__':

    cfg = {'object_directory': 'track_point_20170405_JFK2LAX'}
    fun = draw_figure(cfg)
    fun.plot2D()
