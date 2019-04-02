#! /home/anaconda3 python
#-*- coding: utf-8 -*-

"""
@Author: Yutian Pang
@Date: 2019-02-11

This is the main function to loop through all the flight call sign files processed from flight_data_parser.py.

@Last Modified by: Yutian Pang
@Last Modified date: 2019-03-26
"""
from weather_cube_generator_ET import weather_cube_generator
from process_flight_files import flight_data_generator
import os, utils

#date_list = [20170405, 20170406, 20170407]  # folder name to loop through
#date_list = [20170405]
#date_list = [20170905]  # folder name to loop through
#date_list = utils.get_date_list() # 100 days data
date_list = list(range(20181217, 20181229)) + list(range(20190101, 20190117))


cfg = {'cube_size': 20,  # the size of cube to generate
       'resize_ratio': 1,  # ratio of resize performs to the original weather source
       'downsample_ratio': 5,  # downsample ratio to trajectory files (deleted)
       'departure_airport': 'JFK',
       'arrival_airport': 'LAX',
       'output_dimension': 1000,  # output dimension for trajectory and flight plan
       'altitude_buffer': 0,  # altitude buffer unit: feet
       'weather_path': '/media/ypang6/paralab/Research/data/',  # path to weather file
       }


for date in date_list:
    call_sign_list = sorted([x.split('.')[0] for x in os.listdir("raw_track/track_point_{}_{}2{}/".
                     format(date, cfg['departure_airport'], cfg['arrival_airport']))])

    for call_sign in call_sign_list:

        cfg['date'] = date
        cfg['call_sign'] = call_sign.split('_')[0]

        # modify departure and arrival airport
        #cfg['trajectory_path'] = 'track_points_{}_{}2{}/{}_{}.csv'. \
        cfg['trajectory_path'] = 'flight_plan_{}_{}2{}/{}_{}.csv'. \
            format(cfg['date'], cfg['departure_airport'], cfg['arrival_airport'], cfg['call_sign'], cfg['date'])
        print(cfg['trajectory_path'])

        # run flight data generator
        try:
            fun = flight_data_generator(cfg)
            fun.process_trajectory()
            fun.process_flightplan()
            del fun
            print("Finish flight data for {}.".format(call_sign))

        except:  # ignore file not found error
            print("Error in flight data for {}".format(call_sign))
            pass

        try:
            fun = weather_cube_generator(cfg)
            fun.get_cube()
            del fun
            print("Finish weather data for {}.".format(call_sign))

        except:  # ignore file not found error
            print("Error in weather data for {}".format(call_sign))
            pass
