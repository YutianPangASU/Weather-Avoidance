#! /home/anaconda3 python
#-*- coding: utf-8 -*-

"""
@Author: Yutian Pang
@Date: 2019-02-20

This Python script is used to parse the flight plan strings from FAA data and return array.
The source coordinates for flight plan is https://opennav.com

@Last Modified by: Yutian Pang
@Last Modified date: 2019-02-25
"""

import os
import numpy as np
import pandas as pd
from utils import fetch_from_web


class flight_data_generator(object):
    def __init__(self, cfg):
        self.date = cfg['date']
        self.call_sign = cfg['call_sign']
        self.departure_airport = cfg['departure_airport']
        self.arrival_airport = cfg['arrival_airport']
        self.dimension = cfg['output_dimension']
        self.altitude_threshold = cfg['altitude_buffer']

        try:
            os.makedirs('flight_plan_{}_{}2{}'.format(self.date, self.departure_airport, self.arrival_airport))
            os.makedirs('trajectory data')
            os.makedirs('flightplan data')
        except OSError:
            print("Path already exist.")
            pass

        self.traj = pd.read_csv('track_point_{}_{}2{}/{}_{}.csv'.
                                format(self.date, self.departure_airport, self.arrival_airport, self.call_sign, self.date))

        self.traj['UNIX TIME'].astype(int)  # convert time column to int as index of table

    def process_trajectory(self):

        self.traj = self.traj.set_index('UNIX TIME', drop=True)  # set unix time column  as table index

        # interpolate trajectory data to 1 second interval
        self.total_time = np.arange(self.traj.index[0], self.traj.index[-1])
        self.traj = self.traj.reindex((self.total_time), fill_value=np.nan)

        # fill nan values using linear interpolation
        self.traj = self.traj.interpolate(method='linear')  # interpolated trajectory with 1 second interval

        # buffer altitude over a given threshold
        self.traj_buffered = self.traj[pd.to_numeric(self.traj['ALTITUDE']) >= self.altitude_threshold]

        # get samples with sample interval
        self.sample_interval = round(len(self.traj_buffered) / self.dimension)
        self.traj_return = self.traj.iloc[int((len(self.traj)-self.sample_interval*self.dimension)/2):
                                          int((len(self.traj)+self.sample_interval*self.dimension)/2):
                                          self.sample_interval, :]
        # save fix size trajectory
        np.save('trajectory data/{}_{}.npy'.format(self.date, self.call_sign), self.traj_return)

    def process_flightplan(self):

        file = pd.read_csv(
            'flight_data_{}_{}_to_{}.csv'.format(self.date, self.departure_airport, self.arrival_airport),
            header=None)

        flight_plan_str = file[file[2] == self.call_sign].values[0][4]
        fp = fetch_from_web(flight_plan_str)

        # debug only
        # fp = np.asarray([[40.639751, -73.778925],
        #                       [40.970986, -74.959828],
        #                       [41.480492, -84.64675],
        #                       [41.515975, -85.613689],
        #                       [41.525155, -86.027953],
        #                       [41.518958, -91.613256],
        #                       [41.436999, -93.648707],
        #                       [40.92378, -96.742016],
        #                       [39.387861, -101.692306],
        #                       [38.294255, -104.429446],
        #                       [36.748393, -108.098899],
        #                       [35.147198, -111.674163],
        #                       [34.702556, -112.480349],
        #                       [34.562289, -113.639369],
        #                       [34.515769, -114.055722],
        #                       [33.942536, -118.408075]])

        # find the unix time correspond to waypoints and create a new dataframe
        lat = np.asarray(self.traj['LATITUDE'])
        lon = np.asarray(self.traj['LONGITUDE'])
        unix_time_fp = []
        altitude_fp = []
        for i in range(len(fp)):
            nn = (fp[i][0] - lat) ** 2 + (fp[i][-1] - lon) ** 2
            row_idx = np.unravel_index(nn.argmin(), nn.shape)
            altitude_fp += [self.traj.iloc[row_idx]['ALTITUDE']]
            unix_time_fp += [self.traj.index[row_idx]]
        self.fp = np.column_stack([np.asarray(unix_time_fp), fp, altitude_fp])  # flight plan with unix time and altitude

        # convert to a dataframe
        self.fp = pd.DataFrame(self.fp, columns=['UNIX TIME', 'LATITUDE', 'LONGITUDE', 'ALTITUDE'])

        self.fp = self.fp.set_index('UNIX TIME', drop=True)  # set unix time column as table index

        # interpolate to 1 second interval
        self.fp = self.fp.reindex((self.total_time), fill_value=np.nan)

        # fill nan values using linear interpolation
        self.fp = self.fp.interpolate(method='linear')  # interpolated trajectory with 1 second interval

        # buffer altitude over a given threshold
        self.fp_buffered = self.fp[pd.to_numeric(self.fp['ALTITUDE']) >= self.altitude_threshold]

        # get samples with sample interval
        self.fp_return = self.fp.iloc[int((len(self.fp)-self.sample_interval*self.dimension)/2):
                                      int((len(self.fp)+self.sample_interval*self.dimension)/2):
                                      self.sample_interval, :]
        # save fix size trajectory
        np.save('flightplan data/{}_{}.npy'.format(self.date, self.call_sign), self.fp_return)

        # save as csv to get weather cube
        self.fp_return.reset_index().to_csv('flight_plan_{}_{}2{}/{}_{}.csv'.format(
                        self.date, self.departure_airport, self.arrival_airport, self.call_sign, self.date),
                        sep=',', index=True,
                        header=['UNIX TIME', 'LATITUDE', 'LONGITUDE', 'ALTITUDE'])


if __name__ == '__main__':

    cfg = {'departure_airport': 'JFK',
           'arrival_airport': 'LAX',
           'date': 20170406,
           'call_sign': 'DAL424',
           'output_dimension': 1000,
           'altitude_buffer': 100,
           }

    fun = flight_data_generator(cfg)
    fun.process_trajectory()
    fun.process_flightplan()
