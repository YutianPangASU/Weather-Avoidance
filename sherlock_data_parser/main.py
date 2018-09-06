from jpype import *
from FAA_parser import FAA_Parser
from CIWS_parser import load_ET
import matplotlib.pyplot as plt
import numpy as np
from utils import *
import os


class FAA_ENGINE(object):

    def __init__(self, call_sign, date):  # this engine takes explicitly two inputs, date and flight call sign

        self.time = date
        self.call_sign = call_sign
        self.threshold = 0.2

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

    def run_NATS(self):

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
        aircraftInterface.load_aircraft('/mnt/data/WeatherCNN/sherlock/cache/' + self.time + "_" + self.call_sign + ".trx",
                                        '/mnt/data/WeatherCNN/sherlock/cache/' + self.time + "_" + self.call_sign + "_mfl.trx")
        # default command
        aclist = aircraftInterface.getAllAircraftId()

        # coord = terminalAreaInterface.getWaypoint_Latitude_Longitude_deg('')  # get one waypoint coords

        for i in range(len(aclist)):
            ac = aircraftInterface.select_aircraft(aclist[i])
            lon = ac.getFlight_plan_longitude_array()
            lat = ac.getFlight_plan_latitude_array()
            # save original flightplan waypoint coords as a csv file
            if i == 0:
                np.savetxt("flight_plan_coords/" + self.call_sign + "_" + str(i) + ".csv", np.asarray([lon, lat]).T, delimiter=",")

            plt.plot(lon, lat)
            plt.hold(True)

        plt.legend(self.datetime)
        plt.savefig('flight_plan_plot/flight_plan_' + self.call_sign + '_' + self.time)
        # plt.hold(False)
        # plt.show()

    def draw_traj(self):
        traj = np.genfromtxt('traj_csv/' + self.time + '_' + self.call_sign + '.csv', delimiter=",")  # load csv file
        plt.plot(traj[:, 1], traj[:, 2], 'k--')
        # plt.legend(["real trajectory"])
        plt.savefig('traj_plot/traj_' + self.call_sign + '_' + self.time)
        plt.hold(False)
        # plt.show()

    def fetch_data(self):
        waypoints = np.genfromtxt("flight_plan_coords/" + self.call_sign + '_' + str(0) + '.csv', delimiter=",")
        # delete too close waypoints, usually happened during departure and landing process, not useful for cruise state
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
                continue
            else:
                traj_max_point, traj_max_distance = calculate_max_distance(waypoints[i, :], waypoints[i+1, :],
                                                        trajectory[closest_point_start_idx:closest_point_end_idx, :])
                max_point = np.vstack([max_point, traj_max_point])
                max_distance = np.append(max_distance, traj_max_distance)

        # get the idx of points maximum distance greater than the given threshold value
        # then range it to get the start waypoint and the end waypoint to get the weather plot
        wp_range = ranges(np.squeeze(np.where(max_distance > self.threshold)))

        print "Found " + str(len(wp_range)) + " useful data points from the database of flight " + self.call_sign

        # save picture depending on the point information
        for i in range(len(wp_range)):
            print "start point: " + str(waypoints[wp_range[i][0]])  # start point of the weather contour
            print "end point: " + str(waypoints[wp_range[i][1]])  # end point of the  weather contour


if __name__ == '__main__':

    date = '20170406'
    call_sign = 'FDX1'

    np.warnings.filterwarnings('ignore')  # ignore matplotlib warnings

    fun = FAA_ENGINE(call_sign, date)
    #fun.run_parser_and_save_files()
    #fun.weather_contour()
    #fun.run_NATS()
    #fun.draw_traj()
    fun.fetch_data()
