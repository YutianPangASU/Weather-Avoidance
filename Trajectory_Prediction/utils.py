#! /anaconda/bin/python3 python
# -*- coding: utf-8 -*-

"""
@Author: Yutian Pang
@Date: 2019-01-23

A list of useful python functions.

@Last Modified by: Yutian Pang
@Last Modified date: 2019-01-23
"""

import datetime
import numpy as np


def unixtime_to_datetime(unix_time):  # input can be an array
    time = []
    for i in range(len(unix_time)):
        time.append(datetime.datetime.utcfromtimestamp(int(float(unix_time[i]))).strftime('%Y-%m-%d %H:%M:%S'))
    return time


def find_nearest_value(array, num):
    nearest_val = array[abs(array - num) == abs(array - num).min()]
    return nearest_val


def find_nearest_index(array, num):
    nearest_idx = np.where(abs(array - num) == abs(array - num).min())[0]
    return nearest_idx


def eliminate_zeros(num):  # num should be a 4 digits number

    if num[0] == '0' and num[1] == '0' and num[2] == '0':
        return num[3]
    if num[0] == '0' and num[1] == '0' and num[2] != '0':
        return num[2:]
    if num[0] == '0' and num[1] != '0':
        return num[1:]
    if num[0] != '0':
        return num


def make_up_zeros(str):
    if len(str) == 4:
        return str
    if len(str) == 3:
        return "0" + str
    if len(str) == 2:
        return "00" + str
    if len(str) == 1:
        return "000" + str


def get_weather_file(unix_time):
    pin = datetime.datetime.utcfromtimestamp(int(float(unix_time))).strftime(
        '%Y%m%d %H%M%S')  # time handle to check CIWS database
    array = np.asarray([0, 230, 500, 730,
                        1000, 1230, 1500, 1730,
                        2000, 2230, 2500, 2730,
                        3000, 3230, 3500, 3730,
                        4000, 4230, 4500, 4730,
                        5000, 5230, 5500, 5730])

    # find the closest time for downloading data from CIWS
    nearest_value = int(find_nearest_value(array, np.asarray([int(eliminate_zeros(pin[-4:]))])))
    nearest_value = make_up_zeros(str(nearest_value))  # make up zeros for 0 230 500 730
    return pin, nearest_value


def flight_plan_parser(str):  # use local waypoint database

    str = str[:-5] # remove last 5 characters
    str_list = str.split('.') # break the string
    str_list = list(filter(None, str_list)) # remove empty strings
    print (str_list)

    # store coordinates
    coords = []

    import csv
    for i in range(len(str_list)):
        with open('myFPDB.csv', 'rt') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                if row[0] == str_list[i]:
                    coords += [[row[1], row[2]]]
    return coords


def fetch_from_web(str):  # use online waypoint database source

    str = str[:-5] # remove last 5 characters
    str_list = str.split('.') # break the string
    str_list = list(filter(None, str_list)) # remove empty strings
    print (str_list)

    # store coordinates
    coords = []

    import urllib.request
    # query departure airports
    websource = urllib.request.urlopen("https://opennav.com/airport/{}".format(str_list[0]))
    l = websource.readlines()[13].decode("utf-8")
    lon, lat = l[l.find("(") + 1:l.rfind(")")].split(',')
    coords += [[lon, lat]]

    # query waypoints
    for n in range(1, len(str_list)-1):
        websource = urllib.request.urlopen("https://opennav.com/waypoint/US/{}".format(str_list[n]))
        l = websource.readlines()[13].decode("utf-8")
        try:
            lon, lat = l[l.find("(") + 1:l.rfind(")")].split(',')
            coords += [[lon, lat]]
        except:
            print("Waypoint {} not found.".format(str_list[n]))
            pass

    # query arrival airports
    websource = urllib.request.urlopen("https://opennav.com/airport/{}".format(str_list[-1]))
    l = websource.readlines()[13].decode("utf-8")
    lon, lat = l[l.find("(") + 1:l.rfind(")")].split(',')
    coords += [[lon, lat]]

    return np.asarray(coords).astype(float)  # return flight plan as array


if __name__ == '__main__':
    fp = 'KJFK..COATE.Q436.RAAKK.Q438.RUBYY..DABJU..KG78M..DBQ.J100.JORDY..KP72G..OBH.J10.LBF..LEWOY..KD60U..JNC..HVE..PROMT.Q88.HAKMN.ANJLL1.KLAX/0539'
    #flight_plan_parser(fp)
    wps = fetch_from_web(fp)