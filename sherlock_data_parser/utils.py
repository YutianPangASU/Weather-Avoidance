import datetime
import numpy as np
import pandas as pd


def unixtime_to_datetime(unix_time):  # input can be an array
    time = []
    for i in range(len(unix_time)):
        time.append(datetime.datetime.utcfromtimestamp(int(float(unix_time[i]))).strftime('%Y-%m-%d %H:%M:%S'))
    return time


def save_csv(list, filename, time):

    my_df = pd.DataFrame(list)
    my_df.to_csv("/mnt/data/WeatherCNN/sherlock/traj_csv/" + time + "_" + filename + '.csv', index=False, header=False)


def save_trx(list, filename, time):

    f = open('/mnt/data/WeatherCNN/sherlock/cache/' + time + "_" + filename + '.trx', 'wb')
    f.write("TRACK_TIME 1121238067\n\n")

    fm = open('/mnt/data/WeatherCNN/sherlock/cache/' + time + "_" + filename + '_mfl.trx', 'wb')

    for i in range(len(list)):
        f.write("TRACK A" + str(i) + " ALOR1 370500N 1030900W 470 360 0 ZAB ZAB71\n")
        f.write("FP_ROUTE " + list[i] + "\n\n")
        fm.write("A" + str(i) + " 400\n")

    f.close()
    fm.close()


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
        return "00" + str
