import datetime
import numpy as np
import pandas as pd
from numpy.linalg import norm
from scipy import spatial


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


def calculate_max_distance(a, b, c):
    # this function is used to find the maximum deviation of the trajectory to the flight plan
    # a and b are start and end way points
    # c is a set of points in the trajectory

    m = 0  # m is the maximum area between three points
    idx = -1  # idx is the index of maximum points in c

    for i in range(len(c)):
        area = 0.5 * norm(np.cross(b - a, c[i, :] - a))
        if area > m:
            idx = idx + 1
            m = area

    length = norm(a - b)
    h = np.divide(2*m, length)  # h is calibrated maximum distance

    # return the maximum point in the trajectory and h
    if idx == -1:
        return None
    else:
        return c[idx, :], h


def ranges(nums):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))


if __name__ == '__main__':
    a = np.asarray([1, 0])
    b = np.asarray([5, 6])
    c = np.asarray([[2, 1], [3, 2], [4, 3], [5, 4]])
    point, distance = calculate_max_distance(a, b, c)
    print point, distance