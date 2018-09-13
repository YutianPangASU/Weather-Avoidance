from jpype import *
from FAA_parser import FAA_Parser
from CIWS_parser import load_ET
import matplotlib.pyplot as plt
from utils import *
import os
import csv


class FAA_ENGINE(object):

    def __init__(self, call_sign, date):  # this engine takes explicitly two inputs, date and flight call sign

        self.time = date
        self.call_sign = call_sign
        self.threshold = 0.2
        self.lon = np.load('lon.npy')
        self.lat = np.load('lat.npy')

    def run_parser_and_save_files(self):

        self.flight_plan_sequence_change_time, self.flight_plan_change_sequence, self.traj = \
            FAA_Parser(self.call_sign, self.time).get_flight_plan()  # get flight plan info

        self.datetime = unixtime_to_datetime(self.flight_plan_sequence_change_time)  # transfer unix time to utc

        save_csv(self.traj, self.call_sign, self.time)  # save real trajectory in a csv file

        save_trx(self.flight_plan_change_sequence, self.call_sign, self.time)  # save trx files

    def weather_contour(self):

        func = load_ET(self.time)
        func.load_labels()
        # fun.save_pics()
        for i in range(len(self.flight_plan_sequence_change_time)):
            func.plot_weather_contour(self.flight_plan_sequence_change_time[i], self.call_sign)
        plt.hold(False)

    def run_NATS(self, draw_traj = False):

        os.environ['NATS_CLIENT_HOME']='/mnt/data/NATS/NATS_Client/'
        classpath = "/mnt/data/NATS/NATS_Client/dist/nats-client.jar:/mnt/data/NATS/NATS_Client/dist/nats-shared.jar"

        startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=%s" % classpath)

        FLIGHT_MODE_PREDEPARTURE = JPackage('com').osi.util.Constants.FLIGHT_MODE_PREDEPARTURE
        FLIGHT_MODE_CLIMB = JPackage('com').osi.util.Constants.FLIGHT_MODE_CLIMB
        FLIGHT_MODE_CRUISE = JPackage('com').osi.util.Constants.FLIGHT_MODE_CRUISE
        FLIGHT_MODE_DESCENT = JPackage('com').osi.util.Constants.FLIGHT_MODE_DESCENT
        FLIGHT_MODE_LANDED = JPackage('com').osi.util.Constants.FLIGHT_MODE_LANDED
        FLIGHT_MODE_HOLDING = JPackage('com').osi.util.Constants.FLIGHT_MODE_HOLDING

        NATS_SIMULATION_STATUS_READY = JPackage('com').osi.util.Constants.NATS_SIMULATION_STATUS_READY
        NATS_SIMULATION_STATUS_START = JPackage('com').osi.util.Constants.NATS_SIMULATION_STATUS_START
        NATS_SIMULATION_STATUS_PAUSE = JPackage('com').osi.util.Constants.NATS_SIMULATION_STATUS_PAUSE
        NATS_SIMULATION_STATUS_RESUME = JPackage('com').osi.util.Constants.NATS_SIMULATION_STATUS_RESUME
        NATS_SIMULATION_STATUS_STOP = JPackage('com').osi.util.Constants.NATS_SIMULATION_STATUS_STOP
        NATS_SIMULATION_STATUS_ENDED = JPackage('com').osi.util.Constants.NATS_SIMULATION_STATUS_ENDED

        NATSClientFactory = JClass('NATSClientFactory')

        natsClient = NATSClientFactory.getNATSClient()
        sim = natsClient.getSimulationInterface()

        # Get EquipmentInterface
        equipmentInterface = natsClient.getEquipmentInterface()
        # Get AircraftInterface
        aircraftInterface = equipmentInterface.getAircraftInterface()

        # Get EnvironmentInterface
        environmentInterface = natsClient.getEnvironmentInterface()
        # Get AirportInterface
        airportInterface = environmentInterface.getAirportInterface()
        # Get TerminalAreaInterface
        terminalAreaInterface = environmentInterface.getTerminalAreaInterface()

        # default command
        sim.clear_trajectory()
        environmentInterface.load_rap("share/tg/rap")

        # load trx files
        aircraftInterface.load_aircraft('/mnt/data/WeatherCNN/sherlock/cache/' + self.time + "_" + self.call_sign +
                                        ".trx", '/mnt/data/WeatherCNN/sherlock/cache/' + self.time + "_" +
                                        self.call_sign + "_mfl.trx")
        # default command
        aclist = aircraftInterface.getAllAircraftId()

        # coord = terminalAreaInterface.getWaypoint_Latitude_Longitude_deg('')  # get one waypoint coords

        for i in range(len(aclist)):
            ac = aircraftInterface.select_aircraft(aclist[i])
            lon = ac.getFlight_plan_longitude_array()
            lat = ac.getFlight_plan_latitude_array()
            # save original flightplan waypoint coords as a csv file
            if i == 0:
                np.savetxt("flight_plan_coords/" + self.call_sign + "_" + str(i) + ".csv",
                           np.asarray([lon, lat]).T, delimiter=",")

            plt.plot(lon, lat)
            plt.hold(True)

        plt.legend(self.datetime)
        plt.savefig('flight_plan_plot/flight_plan_' + self.call_sign + '_' + self.time)

        # draw real trajectory along with flight plans
        if draw_traj is True:
            traj = np.genfromtxt('traj_csv/' + self.time + '_' + self.call_sign + '.csv',
                                 delimiter=",")  # load csv file
            plt.plot(traj[:, 1], traj[:, 2], 'k--')
            # plt.legend(["real trajectory"])
            plt.savefig('traj_plot/traj_' + self.call_sign + '_' + self.time)

        plt.hold(False)
        # plt.show()

    def fetch_data(self):

        waypoints = np.genfromtxt("flight_plan_coords/" + self.call_sign + '_' + str(0) + '.csv', delimiter=",")

        # delete too close waypoints, usually happened during departure and landing process, not useful for cruise
        wp_idx = np.unique(np.round(waypoints, 2), axis=0, return_index=True)[1]
        waypoints = waypoints[np.sort(wp_idx)]

        print "Found " + str(len(wp_idx)) + " waypoints from the flight plan of flight " + self.call_sign

        trajectory = np.genfromtxt("traj_csv/" + self.time + "_" + self.call_sign + '.csv', delimiter=",")[:, -2:]

        max_distance = []  # maximum distance
        max_point = np.empty((0, 2))  # maximum point
        for i in range(0, len(waypoints)-1):
            closest_point_start_idx = spatial.KDTree(trajectory).query(waypoints[i, :])[1]
            closest_point_end_idx = spatial.KDTree(trajectory).query(waypoints[i+1, :])[1]

            if closest_point_start_idx >= closest_point_end_idx:
                traj_max_point, traj_max_distance = [0, 0], 0
            else:
                traj_max_point, traj_max_distance = calculate_max_distance(waypoints[i, :], waypoints[i+1, :],
                                                        trajectory[closest_point_start_idx:closest_point_end_idx, :])

            max_point = np.vstack([max_point, traj_max_point])
            max_distance = np.append(max_distance, traj_max_distance)

        # get the idx of points maximum distance greater than the given threshold value
        # then range it to get the start waypoint and the end waypoint to get the weather plot

        wp_range = ranges(np.squeeze(np.where(max_distance > self.threshold)))

        print "Found " + str(len(wp_range)) + " useful data points from the database of flight " + self.call_sign

        # save picture depending on the plot range information
        for i in range(len(wp_range)):

            start_pt = waypoints[wp_range[i][0]]
            end_pt = waypoints[wp_range[i][1] + 1]
            print "start waypoint: " + str(start_pt)  # start point of the weather contour
            print "return waypoint: " + str(end_pt)  # end point of the weather contour

            lon_start_idx = find_nearest_index(self.lon, start_pt[0])
            lon_end_idx = find_nearest_index(self.lon, end_pt[0])
            lat_start_idx = find_nearest_index(self.lat, start_pt[1])
            lat_end_idx = find_nearest_index(self.lat, end_pt[1])

            lon_start_idx, lon_end_idx = sorted([lon_start_idx, lon_end_idx])
            lat_start_idx, lat_end_idx = sorted([lat_start_idx, lat_end_idx])

            if lon_start_idx == lon_end_idx:
                lon_end_idx = lon_end_idx + 1
            if lat_start_idx == lat_end_idx:
                lat_end_idx = lat_end_idx + 1

            # save y_train
            y_train = np.asarray(get_y_train(wp_range[i], max_point, start_pt, end_pt))
            with open('y_train.csv', 'a') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                csvwriter.writerow(y_train)

            # save x_train
            load_ET(self.time).crop_weather_contour(i, self.flight_plan_sequence_change_time[0], self.call_sign,
                                                    lat_start_idx[0], lat_end_idx[0], lon_start_idx[0], lon_end_idx[0],
                                                    y_train, hold=True)


if __name__ == '__main__':

    date = '20170406'
    call_sign = 'ASA19'

    np.warnings.filterwarnings('ignore')  # ignore matplotlib warnings

    fun = FAA_ENGINE(call_sign, date)
    fun.run_parser_and_save_files()
    #fun.weather_contour()
    fun.run_NATS(draw_traj=True)
    fun.fetch_data()
