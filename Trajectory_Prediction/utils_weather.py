import numpy as np


# input: latitude and longitude of a trajectory point
# output: corresponding index in the value matrix of weather data
def find_position(lat, lon, lats, lons):
    nn = (lats - lat) ** 2 + (lons - lon) ** 2
    x, y = np.unravel_index(nn.argmin(), nn.shape)
    return x, y


# input: current trajectory point and previous trajectory point
# output: the slope
def slope_cal(lat_start, lon_start, lat_end, lon_end):
    if lon_end == lon_start:
        return np.inf
    slope = (lat_end - lat_start) / (lon_end - lon_start)
    return slope


# input: original grid index
# output: rotated grid index(according to the slope)
def rorate(x_input, y_input, x_center, y_center, slope):
    x_trans = x_input - x_center
    y_trans = y_input - y_center
    if slope == 0:
        theta = 0
    else:
        theta = np.arctan(1 / slope)
    x_rotate = x_trans * np.cos(theta) - y_trans * np.sin(theta)
    y_rotate = x_trans * np.sin(theta) + y_trans * np.cos(theta)
    x_output = x_rotate + x_center
    y_output = y_rotate + y_center
    return x_output, y_output


# input: rotated index
# output: the value computed by linear interpolation
def interpolate(x_input, y_input, val):
    x_up = int(np.ceil(x_input))
    y_up = int(np.ceil(y_input))
    x_down = int(np.floor(x_input))
    y_down = int(np.floor(y_input))
    if x_up == x_down and y_up == y_down:
        return val[int(x_input)][int(y_input)]
    if x_up == x_down:
        val1 = val[x_up][y_down]
        val2 = val[x_up][y_up]
    else:
        val1 = (val[x_up][y_down] + val[x_down][y_down]) * ((x_input - x_down) / (x_up - x_down))
        val2 = (val[x_up][y_up] + val[x_down][y_up]) * ((x_input - x_down) / (x_up - x_down))
    if y_up == y_down:
        val_out = val1
    else:
        val_out = (val1 + val2) * ((y_input - y_down) / (y_up - y_down))
    return val_out
