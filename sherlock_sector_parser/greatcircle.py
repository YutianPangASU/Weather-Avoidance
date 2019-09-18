#! /home/ypang6/anaconda3/bin/python
#-*- coding: utf-8 -*-

"""
Created on Thu May  9 07:09:00 2019

@author: Nan Xu
This Python script is used to generate the great circle route based on the start and end coordinate in WGS84 coordinates.


@Last Modified by: Yutian Pang
@Last Modified date: 2019-09-18
"""
import numpy as np
import math
import pickle


class GreatCircleRoute(object):
    
    def __init__(self, lon1, lat1, lon2, lat2):
        # WGS84
        self.ra = 6378137.0
        self.rb = 6356752.3142
        self.rmajor = (2 * self.ra + self.rb) / 3.
        self.rminor = (2 * self.ra + self.rb) / 3.

        # start and end points
        self.lon1 = math.radians(lon1)
        self.lat1 = math.radians(lat1)
        self.lon2 = math.radians(lon2)
        self.lat2 = math.radians(lat2)
        
        self.f = (self.rmajor - self.rminor) / self.rmajor
        self.distance, self.azimuth12, self.azimuth21 = self.vinc_dist(self.f, self.rmajor, self.lat1, self.lon1, self.lon2, self.lat2)

        # Great Circle Arc Length in Radians
        self.gcarclen = 2. * math.asin(math.sqrt((math.sin((lat1 - lat2) / 2)) ** 2 + math.cos(lat1) * math.cos(lat2) * (math.sin((lon1 - lon2) / 2)) ** 2))
        
        # Check Antipodal
        if self.gcarclen == math.pi:
            self.antipodal = True
        else:
            self.antipodal = False

    def points(self, npoints):
        # get points
        d = self.gcarclen

        delta = 1.0 / (npoints - 1)

        f = delta * np.arange(npoints)  # f=0 is point 1, f=1 is point 2.

        incdist = self.distance / (npoints - 1)

        lat1 = self.lat1
        lat2 = self.lat2
        lon1 = self.lon1
        lon2 = self.lon2

        # perfect sphere, use great circle formula
        if self.f == 0.:
            A = np.sin((1 - f) * d) / math.sin(d)
            B = np.sin(f * d) / math.sin(d)
            x = A * math.cos(lat1) * math.cos(lon1) + B * math.cos(lat2) * math.cos(lon2)
            y = A * math.cos(lat1) * math.sin(lon1) + B * math.cos(lat2) * math.sin(lon2)
            z = A * math.sin(lat1) + B * math.sin(lat2)
            lats = np.arctan2(z, np.sqrt(x ** 2 + y ** 2))
            lons = np.arctan2(y, x)
            lons = map(math.degrees, lons.tolist())
            lats = map(math.degrees, lats.tolist())
        # use ellipsoid formulas
        else:
            latpt = self.lat1
            lonpt = self.lon1
            azimuth = self.azimuth12
            lons = [math.degrees(lonpt)]
            lats = [math.degrees(latpt)]
            for n in range(npoints - 2):
                latptnew, lonptnew, alpha21 = self.vinc_pt(self.f, self.rmajor, latpt, lonpt, azimuth, incdist)
                d, azimuth, a21 = self.vinc_dist(self.f, self.rmajor, latptnew, lonptnew, lat2, lon2)
                lats.append(math.degrees(latptnew))
                lons.append(math.degrees(lonptnew))
                latpt = latptnew
                lonpt = lonptnew
            lons.append(math.degrees(self.lon2))
            lats.append(math.degrees(self.lat2))

        return np.column_stack((np.asarray(list(lons)), np.asarray(list(lats))))
        
    def vinc_dist(self, f, a, phi1, lembda1, phi2, lembda2):
        
        if (abs(phi2 - phi1) < 1e-8) and (abs(lembda2 - lembda1) < 1e-8):
            return 0.0, 0.0, 0.0

        two_pi = 2.0 * math.pi
        b = a * (1.0 - f)
        TanU1 = (1 - f) * math.tan(phi1)
        TanU2 = (1 - f) * math.tan(phi2)
        U1 = math.atan(TanU1)
        U2 = math.atan(TanU2)
        lembda = lembda2 - lembda1
        last_lembda = -4000000.0  # an impossibe value
        omega = lembda
        
        while (last_lembda < -3000000.0 or lembda != 0 and abs((last_lembda - lembda) / lembda) > 1.0e-9):
            sqr_sin_sigma = pow(math.cos(U2) * math.sin(lembda), 2) + pow((math.cos(U1) * math.sin(U2) - math.sin(U1) * math.cos(U2) * math.cos(lembda)), 2)
            Sin_sigma = math.sqrt(sqr_sin_sigma)
            Cos_sigma = math.sin(U1) * math.sin(U2) + math.cos(U1) * math.cos(U2) * math.cos(lembda)
            sigma = math.atan2(Sin_sigma, Cos_sigma)
            Sin_alpha = math.cos(U1) * math.cos(U2) * math.sin(lembda) / math.sin(sigma)
            alpha = math.asin(Sin_alpha)
            Cos2sigma_m = math.cos(sigma) - (2 * math.sin(U1) * math.sin(U2) / pow(math.cos(alpha), 2))
            C = (f / 16) * pow(math.cos(alpha), 2) * (4 + f * (4 - 3 * pow(math.cos(alpha), 2)))
            last_lembda = lembda
            lembda = omega + (1 - C) * f * math.sin(alpha) * (sigma + C * math.sin(sigma) * (Cos2sigma_m + C * math.cos(sigma) * (-1 + 2 * pow(Cos2sigma_m, 2))))

        u2 = pow(math.cos(alpha), 2) * (a * a - b * b) / (b * b)
        A = 1 + (u2 / 16384) * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
        B = (u2 / 1024) * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))
        delta_sigma = B * Sin_sigma * (Cos2sigma_m + (B / 4) * (Cos_sigma * (-1 + 2 * pow(Cos2sigma_m, 2)) - (B / 6) * Cos2sigma_m * (-3 + 4 * sqr_sin_sigma) * (-3 + 4 * pow(Cos2sigma_m, 2))))
        s = b * A * (sigma - delta_sigma)
        alpha12 = math.atan2((math.cos(U2) * math.sin(lembda)), (math.cos(U1) * math.sin(U2) - math.sin(U1) * math.cos(U2) * math.cos(lembda)))
        alpha21 = math.atan2((math.cos(U1) * math.sin(lembda)), (-math.sin(U1) * math.cos(U2) + math.cos(U1) * math.sin(U2) * math.cos(lembda)))

        if (alpha12 < 0.0):
            alpha12 = alpha12 + two_pi
    
        if (alpha12 > two_pi):
            alpha12 = alpha12 - two_pi

        alpha21 = alpha21 + two_pi / 2.0
    
        if (alpha21 < 0.0):
            alpha21 = alpha21 + two_pi
        
        if (alpha21 > two_pi):
            alpha21 = alpha21 - two_pi

        return s, alpha12, alpha21
    
    def vinc_pt(self, f, a, phi1, lembda1, alpha12, s):

        two_pi = 2.0 * math.pi

        if (alpha12 < 0.0):
            alpha12 = alpha12 + two_pi
    
        if (alpha12 > two_pi):
            alpha12 = alpha12 - two_pi

        b = a * (1.0 - f)
        TanU1 = (1 - f) * math.tan(phi1)
        U1 = math.atan(TanU1)
        sigma1 = math.atan2(TanU1, math.cos(alpha12))
        Sinalpha = math.cos(U1) * math.sin(alpha12)
        cosalpha_sq = 1.0 - Sinalpha * Sinalpha
        u2 = cosalpha_sq * (a * a - b * b) / (b * b)
    
        A = 1.0 + (u2 / 16384) * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
        B = (u2 / 1024) * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))

        # Starting with the approximation
        sigma = (s / (b * A))
        last_sigma = 2.0 * sigma + 2.0  # something impossible

        while (abs((last_sigma - sigma) / sigma) > 1.0e-9):
            two_sigma_m = 2 * sigma1 + sigma
            delta_sigma = B * math.sin(sigma) * (math.cos(two_sigma_m) + (B / 4) * (math.cos(sigma) * (-1 + 2 * math.pow(math.cos(two_sigma_m), 2)-
                                                           (B / 6) * math.cos(two_sigma_m) * (-3 + 4 * math.pow(math.sin(sigma), 2)) * (-3 + 4 * math.pow(math.cos(two_sigma_m), 2)))))
            last_sigma = sigma
            sigma = (s / (b * A)) + delta_sigma

        phi2 = math.atan2((math.sin(U1) * math.cos(sigma) + math.cos(U1) * math.sin(sigma) * math.cos(alpha12)),
                          ((1 - f) * math.sqrt(math.pow(Sinalpha, 2) + pow(math.sin(U1) * math.sin(sigma) - math.cos(U1) * math.cos(
                                               sigma) * math.cos(alpha12), 2))))

        lembda = math.atan2((math.sin(sigma) * math.sin(alpha12)), (math.cos(U1) * math.cos(sigma) - math.sin(U1) * math.sin(sigma) * math.cos(alpha12)))
        C = (f / 16) * cosalpha_sq * (4 + f * (4 - 3 * cosalpha_sq))
        omega = lembda - (1 - C) * f * Sinalpha * (sigma + C * math.sin(sigma) * (math.cos(two_sigma_m) + C * math.cos(sigma) * (-1 + 2 * math.pow(math.cos(two_sigma_m), 2))))
        lembda2 = lembda1 + omega
        alpha21 = math.atan2(Sinalpha, (-math.sin(U1) * math.sin(sigma) + math.cos(U1) * math.cos(sigma) * math.cos(alpha12)))
        alpha21 = alpha21 + two_pi / 2.0
    
        if (alpha21 < 0.0):
            alpha21 = alpha21 + two_pi
            
        if (alpha21 > two_pi):
            alpha21 = alpha21 - two_pi

        return phi2, lembda2, alpha21

        
if __name__ == "__main__":
    sector = 'ZTL'
    date = 20190624
    number_of_points_gc = 50

    dict_fp = pickle.load(open('FP_{}_{}.p'.format(sector, date), 'rb'))

    dict_gc = {}
    for flight_id, fp in dict_fp.items():
        dict_gc[flight_id] = GreatCircleRoute(fp.iloc[0][2], fp.iloc[0][1], fp.iloc[-1][2], fp.iloc[-1][1]).points(number_of_points_gc)

    pickle.dump(dict_gc, open('GC_{}_{}.p'.format(sector, date), 'wb'))

    # # lat/lon of DC.
    # lat1 =  40.64781#40.78
    # lon1 = -73.79528#-73.98
    # # lat/lon of LA.
    # lat2 = 33.93975#51.53
    # lon2 = -118.40543#0.08
    # gc = GreatCircleRoute(lon1, lat1, lon2, lat2).points(50)
    # print(gc)