#! /home/anaconda3 python
#-*- coding: utf-8 -*-

"""
@Author: Yutian Pang
@Date: 2019-02-11

This is the main function to loop through all the flight call sign files processed from flight_data_parser.py.

@Last Modified by: Yutian Pang
@Last Modified date: 2019-02-11
"""
from weather_cube_generator_ET import weather_cube_generator
import os


date_list = [20170405, 20170406, 20170407]  # folder name to loop through


cfg = {'cube_size': 20,  # the size of cube to generate
       'resize_ratio': 1,  # ratio of resize performs to the original weather source
       'downsample_ratio': 5,  # downsample ratio to trajectory files
       'departure_airport': 'JFK',
       'arrival_airport': 'LAX',
       'weather_path': '/mnt/data/Research/data/', # path to weather file
       'handle': 'downsampled'}  # altitude buffered, downsampled or None.

for date in date_list:
    call_sign_list = sorted([x.split('.')[0] for x in os.listdir("track_point_{}_{}2{}_{}/".
                     format(date, cfg['departure_airport'], cfg['arrival_airport'], cfg['handle']))])

    for call_sign in call_sign_list:

        cfg['date'] = date
        cfg['call_sign'] = call_sign.split('_')[0]
        # modify departure and arrival airport
        cfg['trajectory_path'] = 'track_point_{}_{}2{}_{}/{}_{}.csv'. \
            format(cfg['date'], cfg['departure_airport'], cfg['arrival_airport'], cfg['handle'], cfg['call_sign'], cfg['date'])
        print(cfg['trajectory_path'])

        try:
            fun = weather_cube_generator(cfg)
            fun.get_cube()

            print("Finish {}.".format(call_sign))

            del fun

        except:  # ignore file not found error
            print("Error in {}".format(call_sign))
            pass
