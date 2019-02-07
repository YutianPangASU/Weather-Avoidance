import time
import pygrib
import pandas as pd
from utils_weather import *


class weather_cube_generator():

    def __init__(self, cfg):
        self.weather_path = cfg['weather_path']
        self.trajectory_path = cfg['trajectory_path']
        self.cube_size = cfg['cube_size']
        self.date = cfg['date']
        self.call_sign = cfg['call_sign']

        # import weather
        grbs = pygrib.open(self.weather_path)
        a = grbs.select(name='Temperature')[0]
        self.lats, self.lons = a.latlons()
        self.vals_temp = a.values
        a = grbs.select(name='U component of wind')[0]
        self.vals_U_wind = a.values
        a = grbs.select(name='V component of wind')[0]
        self.vals_V_wind = a.values

        # import trajectory
        self.traj = pd.read_csv(self.trajectory_path)

        # define grid
        self.weather_grid_temp = np.zeros(shape=(len(self.traj) - 1, self.cube_size, self.cube_size))
        self.weather_grid_U_wind = np.zeros(shape=(len(self.traj) - 1, self.cube_size, self.cube_size))
        self.weather_grid_V_wind = np.zeros(shape=(len(self.traj) - 1, self.cube_size, self.cube_size))

    def generate(self):

        start = time.time()

        for i in range(1, len(self.traj)):
            # compute index
            print("Working......")
            t = i
            lat = self.traj['LATITUDE'][t]
            lon = self.traj['LONGITUDE'][t]
            x, y = find_position(lat, lon, self.lats, self.lons)  # find the index in weather data of a trajectory point
            x_low = x - self.cube_size // 2
            x_up = x + self.cube_size // 2
            y_up = y + self.cube_size
            x_grid = list(range(x_low, x_up))
            y_grid = list(range(y, y_up))

            # compute slope
            lat_start = self.traj['LATITUDE'][t - 1]
            lon_start = abs(self.traj['LONGITUDE'][t - 1])  # west longitude need to calculate absolute value
            lat_end = lat
            lon_end = abs(lon)
            slope = slope_cal(lat_start, lon_start, lat_end, lon_end)  # calculate the slope of current point

            # calculate the value in each position
            for x_input in x_grid:
                for y_input in y_grid:
                    x_output, y_output = rorate(x_input, y_input, x, y, slope)
                    # compute temprature grid
                    a = interpolate(x_output, y_output, self.vals_temp)
                    self.weather_grid_temp[i-1][int(x_input - x + self.cube_size / 2)][int(y_input - y)] = a
                    # compute U wind grid
                    a = interpolate(x_output, y_output, self.vals_U_wind)
                    self.weather_grid_U_wind[i-1][int(x_input - x + self.cube_size / 2)][int(y_input - y)] = a
                    # compute V wind grid
                    a = interpolate(x_output, y_output, self.vals_V_wind)
                    self.weather_grid_V_wind[i-1][int(x_input - x + self.cube_size / 2)][int(y_input - y)] = a

        data = np.stack((self.weather_grid_temp, self.weather_grid_V_wind, self.weather_grid_U_wind), axis=-1)

        end = time.time()
        print("Total time for one trajectory is: ", end - start)

        # save data
        np.save('weather data/FEATURE_CUBE/{}_{}'.format(self.date, self.call_sign), data)
        np.save('weather data/TEMP/{}_{}'.format(self.date, self.call_sign), self.weather_grid_temp)
        np.save('weather data/WIND_U/{}_{}'.format(self.date, self.call_sign), self.weather_grid_U_wind)
        np.save('weather data/WIND_V/{}_{}'.format(self.date, self.call_sign), self.weather_grid_V_wind)


if __name__ == '__main__':

    cfg ={'cube_size': 20,
          'date': 20170405,
          'call_sign': 'AAL1',
          'weather_path': '/mnt/data/Research/data/NOAA/namanl_218_20170405_0000_000.grb'}
    cfg['trajectory_path'] = 'track_point_{}_JFK2LAX/{}_{}.csv'.format(cfg['date'], cfg['call_sign'], cfg['date'])

    fun = weather_cube_generator(cfg)
    fun.generate()
