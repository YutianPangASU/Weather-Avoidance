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
