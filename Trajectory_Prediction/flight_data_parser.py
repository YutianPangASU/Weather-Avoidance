#! /home/anaconda3 python
#-*- coding: utf-8 -*-

"""
@Author: Yutian Pang
@Date: 2019-01-16

This Python script is used to parse the data downloaded from Sherlock FAA database with a given departure and arrival airport. 
All of the flights fly between these two airport will be parsed and flight information will be saved in a csv file.
The script will also create a folder to save the history track points of each flight callsign.

To run the script, change "path_to_data" and paste "python2 flight_data_parser.py" in terminal.

@Last Modified by: Yutian Pang
@Last Modified date: 2019-01-23
"""

import pandas as pd
import os


class FAA_Departure_Arrival_Parser(object):

    def __init__(self, cfg):

        self.departure = cfg['departure_airport']
        self.arrival = cfg['arrival_airport']
        self.chunk_size = cfg['chunk_size']
        self.date = cfg['file_date']
        self.downsample_rate = cfg['downsample_rate']

    def get_flight_data(self):

        df = pd.read_csv('{}/IFF_USA_{}.csv'.format(cfg['path_to_data'], str(self.date)), chunksize=self.chunk_size,
                         iterator=True, names=range(0, 18), low_memory=False)
        print ("Data File Loaded.")

        i = 0

        # create empty data frame to store useful information
        finfo = pd.DataFrame()

        # create empty data frame for flight plans
        fp = pd.DataFrame()

        for chunk in df:

            i += 1
            print ("Reading chunk number {}".format(str(i)))

            # return the rows numbers that departure and return airport matches
            self.rows = []
            self.rows.extend(chunk.index[chunk[13] == self.departure] & chunk.index[chunk[14] == self.arrival])

            num_of_flights = len(self.rows)

            print ("Found {} flight(s) within this chunk of data".format(num_of_flights))

            # store unix time, run way information, flight callsign and aircraft type
            finfo = chunk.loc[self.rows, [1, 4, 7, 9]]

            # store flight plan
            fp = chunk.loc[[x+1 for x in self.rows], [17]]

            # combine two data frames together
            finfo.reset_index(drop=True, inplace=True)
            fp.reset_index(drop=True, inplace=True)
            data = pd.concat([finfo, fp], axis=1)

            # write data to csv
            data.to_csv('flight_data_{}_{}_to_{}.csv'.format(self.date, self.departure, self.arrival),
                        sep=',',
                        mode='a',
                        index=False,
                        header=False)
                        #header=['UNIX TIME', 'RUNWAY INFO', 'CALL SIGN', 'AIRCRAFT TYPE', 'FLIGHT PLAN'])

            # store track point and write to csv
            for n in range(num_of_flights):
                track = chunk.loc[chunk.index[chunk[7] == finfo.iloc[n, 2]] & chunk.index[chunk[0] == 3], [1, 9, 10, 11]]
                track.to_csv('track_point_{}_{}2{}/{}_{}.csv'.format(
                             self.date, cfg['departure_airport'], cfg['arrival_airport'], finfo.iloc[n, 2], self.date),
                             sep=',',
                             index=False,
                             header=['UNIX TIME', 'LATITUDE', 'LONGITUDE', 'ALTITUDE'])

                downsampled_track = track.iloc[::self.downsample_rate, :]
                downsampled_track.to_csv('track_point_{}_{}2{}_downsampled/{}_{}.csv'.format(
                             self.date, cfg['departure_airport'], cfg['arrival_airport'], finfo.iloc[n, 2], self.date),
                             sep=',',
                             index=False,
                             header=['UNIX TIME', 'LATITUDE', 'LONGITUDE', 'ALTITUDE'])


if __name__ == '__main__':

    cfg = {'departure_airport': 'JFK',
           'arrival_airport': 'LAX',
           'chunk_size': 1e6,
           'file_date': 20170407,
           'downsample_rate': 5,  # take one row out of five rows
           'path_to_data': '/mnt/data/Research/data'}

    try:
        os.makedirs('track_point_{}_{}2{}'.format(cfg['file_date'], cfg['departure_airport'], cfg['arrival_airport']))
    except OSError:
        pass

    try:
        os.makedirs('track_point_{}_{}2{}_downsampled'.format(cfg['file_date'], cfg['departure_airport'], cfg['arrival_airport']))
    except OSError:
        pass

    fun = FAA_Departure_Arrival_Parser(cfg)
    fun.get_flight_data()
