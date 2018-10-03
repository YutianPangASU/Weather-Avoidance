import pandas as pd
import numpy as np


class FAA_Parser(object):

    def __init__(self, call_sign, time, chunk_size):

        self.time = time
        self.call_sign = call_sign
        self.chunk_size = chunk_size

        # specific row numbers to keep
        self.rows = []

        # specific colomn numbers to keep
        # cols = [0, 1, 7, 17]  # flightID, time, flight number, flight plan
        self.cols = range(0, 18)  # include lat and lon

        # track point is the array to store trajectories
        # self.track_point = []

    def get_flight_plan(self):

        # chunk number index
        i = 0

        df = pd.read_csv('data/IFF_USA_' + self.time + '_050000_86396.csv', chunksize=self.chunk_size, iterator=True,
                         names=range(0, 18), low_memory=False)

        self.track_point = np.empty((0, 18))

        for chunk in df:

            i = i + 1
            print "reading chunk number " + str(i)

            self.rows = []
            self.rows.extend(chunk.index[chunk[7] == self.call_sign])

            if len(self.rows) != 0:
                data = chunk.ix[self.rows][self.cols]
                self.track_point = np.concatenate((self.track_point, data), axis=0)

    def save_csv(self):

        my_df = pd.DataFrame(self.track_point)
        my_df.to_csv(self.time + "_" + self.call_sign + '.csv', index=False,
                     header=False)


if __name__ == '__main__':

    call_sign = 'AAL717'
    time = '20170406'

    chunk_size = 1e6

    fun = FAA_Parser(call_sign, time, chunk_size)
    fun.get_flight_plan()
    fun.save_csv()