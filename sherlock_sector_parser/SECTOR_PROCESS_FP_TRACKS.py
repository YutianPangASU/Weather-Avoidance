#! /home/ypang6/anaconda3/bin/python
#-*- coding: utf-8 -*-

"""
@Author: Yutian Pang
@Date: 2019-09-10

This Python script is used to process the raw flight data from SECTOR_FLIGHT_PARSER_RAW.py (raw flight plan and raw trajectory).
The three input of this class are, date, sector name and number of points needed.
The output of the class are the processed, equal-length dictionaries of flight plan and trajectory.

@Last Modified by: Yutian Pang
@Last Modified date: 2019-09-10
"""

import pandas as pd
import numpy as np
import utils as ut
import pickle


class sector_processer(object):
    def __init__(self, cfg):

        self.length = cfg['number_of_points']
        self.date = cfg['date']
        self.sector_name = cfg['sector_name']

        self.fp_raw = np.load('{}/FP_{}_{}.npy'.format(self.sector_name, self.sector_name, self.date))
        self.traj_raw = np.load('{}/TRACKS_{}_{}.npy'.format(self.sector_name, self.sector_name, self.date), encoding='latin1')

    def process(self):
        dict_traj_return = {}
        dict_fp_return = {}
        for key in self.fp_raw.item():
            print("Processing Flight {}".format(key))
            try:
                # run the trajectory and flight plan parser
                self.process_traj_fp(key)
                # save into corresponidng dictionary
                dict_traj_return[key], dict_fp_return[key] = self.traj_return, self.fp_return
            except:
                print("Error in Flight {}".format(key))
                pass

        pickle.dump(dict_fp_return, open('FP_{}_{}.p'.format(self.sector_name, self.date), 'wb'))
        pickle.dump(dict_traj_return, open('TRACKS_{}_{}.p'.format(self.sector_name, self.date), 'wb'))

        #data = pickle.load(open('FP_{}_{}.p'.format(self.sector_name, self.date), 'rb'))

    def process_traj_fp(self, key):

        # get trajectory from dictionary
        self.traj = self.traj_raw.item().get(key).reset_index().loc[:, [1, 9, 10, 11]].astype(float)

        # convert the time column to integer then set as the index column
        self.traj[1] = self.traj[1].astype(int)
        self.traj = self.traj.set_index(1, drop=True)

        # interpolate trajectory data to 1 second interval
        self.total_time = np.arange(self.traj.index[0], self.traj.index[-1])
        self.traj = self.traj.reindex(self.total_time, fill_value=np.nan)

        # fill nan values using linear interpolation
        self.traj = self.traj.interpolate(method='linear')  # interpolated trajectory with 1 second interval

        # parse string format of flight plan
        self.fp = ut.fetch_from_web(self.fp_raw.item().get(key))
        lat, lon = np.asarray(self.traj[9]), np.asarray(self.traj[10])

        # find the time and altitude for flight plan points
        unix_time_fp = []
        altitude_fp = []
        out_idx = []
        for i in range(len(self.fp)):
            nn = (self.fp[i][0] - lat) ** 2 + (self.fp[i][-1] - lon) ** 2
            if nn.min() > 0.5:  # get the fp index that are out of the sector range
                out_idx += [i]
            else:
                row_idx = np.unravel_index(nn.argmin(), nn.shape)
                altitude_fp += [self.traj.iloc[row_idx][11]]
                unix_time_fp += [self.traj.index[row_idx]]

        # remove flight plan points out of the sector
        self.fp = np.delete(self.fp, out_idx, axis=0)

        # return flight plan with unix time and altitude
        self.fp = pd.DataFrame(np.column_stack([np.asarray(unix_time_fp), self.fp, altitude_fp]))

        # convert the time column to integer then set as the index column
        self.fp[0] = self.fp[0].astype(int)
        self.fp = self.fp.set_index(0, drop=True)

        # interpolate fp data to 1 second interval
        self.total_time_fp = np.arange(self.fp.index[0], self.fp.index[-1]+1)
        self.fp = self.fp.reindex(self.total_time_fp, fill_value=np.nan)

        # fill nan values using linear interpolation
        self.fp = self.fp.interpolate(method='linear')  # interpolated fp with 1 second interval

        # cut trajectory points between flight plan, use time match
        start_time = int(unix_time_fp[0])
        end_time = int(unix_time_fp[-1])
        self.traj = self.traj.loc[start_time:end_time, :]

        # get samples with sample interval
        self.sample_interval = int(len(self.traj) / self.length)

        # get the return trajectory and flight plan
        self.traj_return = self.traj.iloc[int((len(self.traj) - self.sample_interval * self.length) / 2):
                                          int((len(self.traj) + self.sample_interval * self.length) / 2):
                                          self.sample_interval, :]
        self.fp_return = self.fp.iloc[int((len(self.fp) - self.sample_interval * self.length) / 2):
                                          int((len(self.fp) + self.sample_interval * self.length) / 2):
                                          self.sample_interval, :]


if __name__ == '__main__':
    cfg = {}
    cfg['date'] = '20190805'
    cfg['sector_name'] = 'ZID'
    cfg['number_of_points'] = 50
    fun = sector_processer(cfg)
    fun.process()
