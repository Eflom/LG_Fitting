# -*- coding: utf-8 -*-
"""
Created on Thu May 18 16:08:37 2017

@author: EFlom
"""

import numpy as np
from numpy import mean, sqrt, square, average
import numpy.ma as ma
import pylab
import matplotlib.pyplot as plt
import scipy
from scipy import special
from mpl_toolkits.mplot3d.axes3d import Axes3D
import cmath
import pandas as pd
import scipy.optimize as opt
from scipy.odr import ODR, odr, Model, Data, RealData, Output



#specifycrop vallues for the data -- IMPORTANT!! the data must be a square
xmin = 350
xmax = 739 
ymin = 570
ymax = 959 
x0 = (xmin + xmax)/2
y0 = (ymin + ymax)/2
ranges_inch = np.arange(0,10,1)
ranges = (ranges_inch)*25.4 + 5

#debug use only, used for making simulated LG data 'noisy'
xerr = scipy.random.random(200)
yerr = scipy.random.random(200)
zerr = scipy.random.random(200)*10.0 -5.0

#read in data from file, cropping it using the values above

data1_1 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/2_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data1_2 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/2_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data1_3 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/2_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data1_4 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/2_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data1_5 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/2_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data1_6 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/33_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data1_7 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/33_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data1_8 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/33_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data1_9 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/33_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data1_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/33_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))


data2_1 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/3_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data2_2 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/3_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data2_3 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/3_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data2_4 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/3_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data2_5 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/3_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data2_6 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/2_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data2_7 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/2_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data2_8 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/2_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data2_9 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/2_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data2_10 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/2_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))


data3_1 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/4_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data3_2 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/4_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data3_3 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/4_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data3_4 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/4_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data3_5 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/4_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data3_6 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/3_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data3_7 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/3_00007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data3_8 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/3_00008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data3_9 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/3_00009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data3_10 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/3_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))


data4_1 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/5_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data4_2 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/5_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data4_3 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/5_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data4_4 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/5_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data4_5 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/5_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data4_6 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/2-1/Optic2_24_00006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data4_7 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/4_00007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data4_8 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/4_00008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data4_9 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/4_00009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data4_10 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/4_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))


data5_1 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/6_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data5_2 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/6_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data5_3 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/6_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data5_4 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/6_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data5_5 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/6_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data5_6 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/5_00006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data5_7 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/5_00007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data5_8 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/5_00008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data5_9 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/5_00009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data5_10 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/5_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data6_1 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/11_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data6_2 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/11_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data6_3 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/11_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data6_4 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/11_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data6_5 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/11_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data6_6 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/6_00006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data6_7 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/6_00007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data6_8 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/6_00008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data6_9 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/6_00009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data6_10 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/6_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data7_1 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/13_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data7_2 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/13_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data7_3 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/13_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data7_4 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/13_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data7_5 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/13_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data7_6 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/2-1/Optic2_24_00006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
##data7_7 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/2-1/August_5_2/7_00007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data7_8 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/7_00008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data7_9 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/7_00009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data7_10 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/7_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data8_1 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/15_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data8_2 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/15_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data8_3 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/15_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data8_4 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/15_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data8_5 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/15_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data8_6 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/2-1/Optic2_218_00006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data8_7 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/8_00007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data8_8 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/8_00008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data8_9 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/8_00009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data8_10 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/8_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data9_1 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/17_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data9_2 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/17_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data9_3 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/17_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data9_4 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/17_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data9_5 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/17_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data9_6 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/2-1/Optic2_249_00006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data9_7 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/9_00007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data9_8 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/9_00008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data9_9 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/9_00009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data9_10 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/9_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data10_1 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/19_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data10_2 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/19_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data10_3 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/19_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data10_4 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/19_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data10_5 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/19_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data10_6 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/2-1/Optic2_210_00006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data10_7 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/10_00007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data10_8 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/10_00008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data10_9 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/10_00009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data10_10 = np.array(pd.read_table('C:/Users/EFlom/Documents/Data/10-1/Normal/10_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

print 'data read in complete'


data1 = (data1_1 + data1_2 + data1_3 + data1_4 + data1_5)/5.# + data1_6 + data1_7 + data1_8 + data1_9 + data1_10)/10.
data2 = (data2_1 + data2_2 + data2_3 + data2_4 + data2_5)/5.# + data2_6 + data2_7 + data2_8 + data2_9 + data2_10)/10.
data3 = (data3_1 + data3_2 + data3_3 + data3_4 + data3_5)/5.# + data3_6 + data3_7 + data3_8 + data3_9 + data3_10)/10.
data4 = (data4_1 + data4_2 + data4_3 + data4_4 + data4_5)/5.# + data4_6 + data4_7 + data4_8 + data4_9 + data4_10)/10.
data5 = (data5_1 + data5_2 + data5_3 + data5_4 + data5_5)/5.# + data5_6 + data5_7 + data5_8 + data5_9 + data5_10)/10.
data6 = (data6_1 + data6_2 + data6_3 + data6_4 + data6_5)/5.# + data6_6 + data6_7 + data6_8 + data6_9 + data6_10)/10.
data7 = (data7_1 + data7_2 + data7_3 + data7_4 + data7_5)/5.# + data7_6 + data7_7 + data7_8 + data7_9 + data7_10)/10.
data8 = (data8_1 + data8_2 + data8_3 + data8_4 + data8_5)/5.# + data8_6 + data8_7 + data8_8 + data8_9 + data8_10)/10.
data9 = (data9_1 + data9_2 + data9_3 + data9_4 + data9_5)/5.# + data9_6 + data9_7 + data9_8 + data9_9 + data9_10)/10.
data10 = (data10_1 + data10_2 + data10_3 + data10_4 + data10_5)/5.# + data10_6 + data10_7 + data10_8 + data10_9 + data10_10)/10.

print 'data averaging complete'

#data1_sd = np.std(data1_1, data1_2, data1_3, data1_4, data1_5)#, data1_6, data1_7])#, data1_8, data1_9, data1_10], axis = 0)
#data2_sd = np.std(data2_1, data2_2, data2_3, data2_4, data2_5)#, data2_6, data2_7])#, data2_8, data2_9, data2_10], axis = 0)
#data3_sd = np.std(data3_1, data3_2, data3_3, data3_4, data3_5)#, data3_6, data3_7])#, data3_8, data3_9, data3_10], axis = 0)
#data4_sd = np.std(data4_1, data4_2, data4_3, data4_4, data4_5)#, data4_6, data4_7])#, data4_8, data4_9, data4_10], axis = 0)
#data5_sd = np.std(data5_1, data5_2, data5_3, data5_4, data5_5)#, data5_6, data5_7])#, data5_8, data5_9, data5_10], axis = 0)
#data6_sd = np.std(data6_1, data6_2, data6_3, data6_4, data6_5)#, data6_6, data6_7])#, data6_8, data6_9, data6_10], axis = 0)
#data7_sd = np.std(data7_1, data7_2, data7_3, data7_4, data7_5)#, data7_6, data7_7])#, data7_8, data7_9, data7_10], axis = 0)
#data8_sd = np.std(data8_1, data8_2, data8_3, data8_4, data8_5)#, data8_6, data8_7])#, data8_8, data8_9, data8_10], axis = 0)
#data9_sd = np.std(data9_1, data9_2, data9_3, data9_4, data9_5)#, data9_6, data9_7])#, data9_8, data9_9, data9_10], axis = 0)
#data10_sd = np.std(data10_1, data10_2, data10_3, data10_4, data10_5)#, data10_6])#, data10_7, data10_8, data10_9, data10_10], axis = 0)

#print data1_sd

#print 'standard deviation calculation complete'

#generate a regular x-y space as independent variables

x = np.arange(xmin, xmax, 1)
y = np.arange(ymin, ymax, 1)

#specify radial and azimuthal modes of the LG Beam

l=10.
p=1.

#generate the 2D grid for plotting

X, Y = np.meshgrid(x - x0, y -y0)

#define these terms b/c I'm lazy and don't want to keep typing them

Q = x - x0
W = y - y0

#Debug use only, used for generating simulated LG cross-sections

A = 100.
w = 30.
B = 1


r = (X**2 + Y**2)**0.5

Z1 = ((2.*(r**2)/w**2))**l
Z2 = ((l+p - ((2.*r**2)/w**2.))**(p))**2.  ##use Mathematica LaguerreL [bottom, top, function] to generate this term
Z3 = np.exp((-(2.))*(r**2/w**2.))
Z4 = A*Z1*Z2*Z3 + B

#routine to mask data to remove bottom fraction of values so the "shoulder" doesn't interfere with fitting
Crop_range = 0.2 #Specify as decimal

zmin1 = np.max(np.max(data1))*Crop_range
zmin2 = np.max(np.max(data2))*Crop_range
zmin3 = np.max(np.max(data3))*Crop_range
zmin4 = np.max(np.max(data4))*Crop_range
zmin5 = np.max(np.max(data5))*Crop_range
zmin6 = np.max(np.max(data6))*Crop_range
zmin7 = np.max(np.max(data7))*Crop_range
zmin8 = np.max(np.max(data8))*Crop_range
zmin9 = np.max(np.max(data9))*Crop_range
zmin10 = np.max(np.max(data10))*Crop_range


maskeddata1 = np.where(data1 > zmin1, data1, 100)
maskeddata2 = np.where(data2 > zmin2, data2, 100)
maskeddata3 = np.where(data3 > zmin3, data3, 100)
maskeddata4 = np.where(data4 > zmin4, data4, 100)
maskeddata5 = np.where(data5 > zmin5, data5, 100)
maskeddata6 = np.where(data6 > zmin6, data6, 100)
maskeddata7 = np.where(data7 > zmin7, data7, 100)
maskeddata8 = np.where(data8 > zmin8, data8, 100)
maskeddata9 = np.where(data9 > zmin9, data9, 100)
maskeddata10 = np.where(data10 > zmin10, data10, 100)



#define the funciton in terms of the 5 paramters, so that the ODR can process them
def function(params, data):
	scale = params[0]
	baseline = params[2] #=620 ## Keep looking at the fit outputs. If you're having fit problems (like if the fit isn't getting very close to the correct baseline), one thing to do is look at the data and try to ballpark the average value of the noise, then pin the baseline it to that one value. 
	width = params[1]
	y_0 = params[3]
	x_0 = params[4]
		
	return ((((scale))*(2.*((X - x_0)**2. + (Y - y_0)**2.)/(width)**2.))**l)*((l+p - ((2.*((X - x_0)**2. + (Y - y_0)**2.))/(width)**2.))**2.)*(np.exp((-2.)*(((X - x_0)**2. + (Y - y_0)**2.)/(width)**2.))) + baseline


#The meat of the ODR program. set "guesses" to a rough initial guess for the data <-- IMPORTANT

myData1 = Data([Q, W], data1)
myModel = Model(function)
guesses1 = [1, 30, 600, -0, -50] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr1 = ODR(myData1, myModel, guesses1, maxit=2000)
odr1.set_job(fit_type=2)
output1 = odr1.run()
output1.pprint()
Fit_out1 = (((((output1.beta[0]))*(2.*((X - output1.beta[4])**2. + (Y - output1.beta[3])**2.)/(output1.beta[1])**2.)))**l)*((l+p - ((2.*((X - output1.beta[4])**2. + (Y - output1.beta[3])**2.))/(output1.beta[1])**2))**2)*(np.exp((-2.)*(((X - output1.beta[4])**2. + (Y - output1.beta[3])**2.)/(output1.beta[1])**2.))) + output1.beta[2]
print 'done1'

myData2 = Data([Q, W], data2)
myModel = Model(function)
guesses2 = [1, 30, 600, -20, 2] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr2 = ODR(myData2, myModel, guesses2, maxit=2000)
odr2.set_job(fit_type=2)
output2 = odr2.run()
output2.pprint()
Fit_out2 = (((((output2.beta[0]))*(2.*((X - output2.beta[4])**2. + (Y - output2.beta[3])**2.)/(output2.beta[1])**2.)))**l)*((l+p - ((2.*((X - output2.beta[4])**2. + (Y - output2.beta[3])**2.))/(output2.beta[1])**2))**2)*(np.exp((-2.)*(((X - output2.beta[4])**2. + (Y - output2.beta[3])**2.)/(output2.beta[1])**2.))) + output2.beta[2]
print 'done2'

myData3 = Data([Q, W], data3)
myModel = Model(function)
guesses3 = [1, 40, 600, 20, -40]
odr3 = ODR(myData3, myModel, guesses3, maxit=2000)
odr3.set_job(fit_type=2)
output3 = odr3.run()
output3.pprint()
Fit_out3 = (((((output3.beta[0]))*(2.*((X - output3.beta[4])**2. + (Y - output3.beta[3])**2.)/(output3.beta[1])**2.)))**l)*((l+p - ((2.*((X - output3.beta[4])**2. + (Y - output3.beta[3])**2.))/(output3.beta[1])**2))**2)*(np.exp((-2.)*(((X - output3.beta[4])**2. + (Y - output3.beta[3])**2.)/(output3.beta[1])**2.))) + output3.beta[2]
print 'done3'

myData4 = Data([Q, W], data4)
myModel = Model(function)
guesses4 = [1, 40, 600, -30, -40] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr4 = ODR(myData4, myModel, guesses4, maxit=2000)
odr4.set_job(fit_type=2)
output4 = odr4.run()
output4.pprint()
Fit_out4 = (((((output4.beta[0]))*(2.*((X - output4.beta[4])**2. + (Y - output4.beta[3])**2.)/(output4.beta[1])**2.)))**l)*((l+p - ((2.*((X - output4.beta[4])**2. + (Y - output4.beta[3])**2.))/(output4.beta[1])**2))**2)*(np.exp((-2.)*(((X - output4.beta[4])**2. + (Y - output4.beta[3])**2.)/(output4.beta[1])**2.))) + output4.beta[2]
print 'done4'

myData5 = Data([Q, W], data5)
myModel = Model(function)
guesses5 = [1, 50, 600, 0, 0] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr5 = ODR(myData5, myModel, guesses5, maxit=2000)
odr5.set_job(fit_type=2)
output5 = odr5.run()
output5.pprint()
Fit_out5 = (((((output5.beta[0]))*(2.*((X - output5.beta[4])**2. + (Y - output5.beta[3])**2.)/(output5.beta[1])**2.)))**l)*((l+p - ((2.*((X - output5.beta[4])**2. + (Y - output5.beta[3])**2.))/(output5.beta[1])**2))**2)*(np.exp((-2.)*(((X - output5.beta[4])**2. + (Y - output5.beta[3])**2.)/(output5.beta[1])**2.))) + output5.beta[2]
print 'done5'

myData6 = Data([Q, W], data6)
myModel = Model(function)
guesses6 = [100, 60, 600, 0, 0] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr6 = ODR(myData6, myModel, guesses6, maxit=2000)
odr6.set_job(fit_type=2)
output6 = odr6.run()
output6.pprint()
Fit_out6 = (((((output6.beta[0]))*(2.*((X - output6.beta[4])**2. + (Y - output6.beta[3])**2.)/(output6.beta[1])**2.)))**l)*((l+p - ((2.*((X - output6.beta[4])**2. + (Y - output6.beta[3])**2.))/(output6.beta[1])**2))**2)*(np.exp((-2.)*(((X - output6.beta[4])**2. + (Y - output6.beta[3])**2.)/(output6.beta[1])**2.))) + output6.beta[2]
print 'done6'

myData7 = Data([Q, W], maskeddata7)
myModel = Model(function)
guesses7 = [100, 70, 600, 0, 0] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr7 = ODR(myData7, myModel, guesses7, maxit=2000)
odr7.set_job(fit_type=2)
output7 = odr7.run()
output7.pprint()
Fit_out7 = (((((output7.beta[0]))*(2.*((X - output7.beta[4])**2. + (Y - output7.beta[3])**2.)/(output7.beta[1])**2.)))**l)*((l+p - ((2.*((X - output7.beta[4])**2. + (Y - output7.beta[3])**2.))/(output7.beta[1])**2))**2)*(np.exp((-2.)*(((X - output7.beta[4])**2. + (Y - output7.beta[3])**2.)/(output7.beta[1])**2.))) + output7.beta[2]
print 'done7'

myData8 = Data([Q, W], maskeddata8)
myModel = Model(function)
guesses8 = [100, 80, 600, 0, 0] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr8 = ODR(myData8, myModel, guesses8, maxit=2000)
odr8.set_job(fit_type=2)
output8 = odr8.run()
output8.pprint()
Fit_out8 = (((((output8.beta[0]))*(2.*((X - output8.beta[4])**2. + (Y - output8.beta[3])**2.)/(output8.beta[1])**2.)))**l)*((l+p - ((2.*((X - output8.beta[4])**2. + (Y - output8.beta[3])**2.))/(output8.beta[1])**2))**2)*(np.exp((-2.)*(((X - output8.beta[4])**2. + (Y - output8.beta[3])**2.)/(output8.beta[1])**2.))) + output8.beta[2]
print 'done8'

myData9 = Data([Q, W], maskeddata9)
myModel = Model(function)
guesses9 = [100, 90, 600, 00, 0] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr9 = ODR(myData9, myModel, guesses9, maxit=2000)
odr9.set_job(fit_type=2)
output9 = odr9.run()
output9.pprint()
Fit_out9 = (((((output9.beta[0]))*(2.*((X - output9.beta[4])**2. + (Y - output9.beta[3])**2.)/(output9.beta[1])**2.)))**l)*((l+p - ((2.*((X - output9.beta[4])**2. + (Y - output9.beta[3])**2.))/(output9.beta[1])**2))**2)*(np.exp((-2.)*(((X - output9.beta[4])**2. + (Y - output9.beta[3])**2.)/(output9.beta[1])**2.))) + output9.beta[2]
print 'done9'

myData10 = Data([Q, W], maskeddata10)
myModel = Model(function)
guesses10 = [100, 100, 600, 0, 0] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr10 = ODR(myData10, myModel, guesses10, maxit=2000)
odr10.set_job(fit_type=2)
output10 = odr10.run()
output10.pprint()
Fit_out10 = (((((output10.beta[0]))*(2.*((X - output10.beta[4])**2. + (Y - output10.beta[3])**2.)/(output10.beta[1])**2.)))**l)*((l+p - ((2.*((X - output10.beta[4])**2. + (Y - output10.beta[3])**2.))/(output10.beta[1])**2))**2)*(np.exp((-2.)*(((X - output10.beta[4])**2. + (Y - output10.beta[3])**2.)/(output10.beta[1])**2.))) + output10.beta[2]
print 'done10'

maskedfit1 = np.where(data1 > zmin1, Fit_out1, 100)
maskedfit2 = np.where(data2 > zmin2, Fit_out2, 100)
maskedfit3 = np.where(data3 > zmin3, Fit_out3, 100)
maskedfit4 = np.where(data4 > zmin4, Fit_out4, 100)
maskedfit5 = np.where(data5 > zmin5, Fit_out5, 100)
maskedfit6 = np.where(data6 > zmin6, Fit_out6, 100)
maskedfit7 = np.where(data7 > zmin7, Fit_out7, 100)
maskedfit8 = np.where(data8 > zmin8, Fit_out8, 100)
maskedfit9 = np.where(data9 > zmin9, Fit_out9, 100)
maskedfit10 = np.where(data10 > zmin10, Fit_out10, 100)

print 'masking data done'

Chisq1 = np.sum(np.sum((((maskeddata1 - maskedfit1))**2)/(maskedfit1)))
Chisq2 = np.sum(np.sum((((maskeddata2 - maskedfit2))**2)/(maskedfit2)))
Chisq3 = np.sum(np.sum((((maskeddata3 - maskedfit3))**2)/(maskedfit3)))
Chisq4 = np.sum(np.sum((((maskeddata4 - maskedfit4))**2)/(maskedfit4)))
Chisq5 = np.sum(np.sum((((maskeddata5 - maskedfit5))**2)/(maskedfit5)))
Chisq6 = np.sum(np.sum((((maskeddata6 - maskedfit6))**2)/(maskedfit6)))
Chisq7 = np.sum(np.sum((((maskeddata7 - maskedfit7))**2)/(maskedfit7)))
Chisq8 = np.sum(np.sum((((maskeddata8 - maskedfit8))**2)/(maskedfit8)))
Chisq9 = np.sum(np.sum((((maskeddata9 - maskedfit9))**2)/(maskedfit9)))
Chisq10 = np.sum(np.sum((((maskeddata10 - maskedfit10))**2)/(maskedfit10)))

print 'Chi Squared analysis done'

z1 = np.count_nonzero(np.where(z1 < num_sd, z1, 0))
z2 = np.count_nonzero(np.where(z2 < num_sd, z2, 0))
z3 = np.count_nonzero(np.where(z3 < num_sd, z3, 0))
z4 = np.count_nonzero(np.where(z4 < num_sd, z4, 0))
z5 = np.count_nonzero(np.where(z5 < num_sd, z5, 0))
#z6 = np.count_nonzero(np.where(z6 < num_sd, z6, 0))
#z7 = np.count_nonzero(np.where(z7 < num_sd, z7, 0))
#z8 = np.count_nonzero(np.where(z8 < num_sd, z8, 0))
#z9 = np.count_nonzero(np.where(z9 < num_sd, z9, 0))
#z10 = np.count_nonzero(np.where(z10 < num_sd, z10, 0))

N1 = np.count_nonzero((maskeddata1) - (maskedfit1))
N2 = np.count_nonzero((maskeddata2) - (maskedfit2))
N3 = np.count_nonzero((maskeddata3) - (maskedfit3))
N4 = np.count_nonzero((maskeddata4) - (maskedfit4))
N5 = np.count_nonzero((maskeddata5) - (maskedfit5))
#N6 = np.count_nonzero((maskeddata6) - (maskedfit6))
#N7 = np.count_nonzero((maskeddata7) - (maskedfit7))
#N8 = np.count_nonzero((maskeddata8) - (maskedfit8))
#N9 = np.count_nonzero((maskeddata9) - (maskedfit9))
#N10 = np.count_nonzero((maskeddata10) - (maskedfit10))

print (z1, N1)
print (z1, N1)
print (z2, N2)
print (z3, N3) 
print (z4, N4)
print (np.float(z1)/np.float(N1))


print 'standard deviation analysis done'

adj_chi1 = [Chisq1/N1, Chisq2/N2, Chisq3/N3, Chisq4/N4, Chisq5/N5]#, Chisq6/N6, Chisq7/N7, Chisq8/N8, Chisq9/N9, Chisq10/N10]
Mode_Purity = [np.float64(z1)/np.float64(N1), np.float64(z2)/np.float64(N2), np.float64(z3)/np.float64(N3), np.float64(z4)/np.float64(N4), np.float64(z5)/np.float64(N5)]#, np.float64(z6)/np.float64(N6), np.float64(z7)/np.float64(N7), np.float64(z8)/np.float64(N8), np.float64(z9)/np.float64(N9), np.float64(z10)/np.float64(N10)]
#diameters = [output1.beta[1], output2.beta[1], output3.beta[1], output4.beta[1], output5.beta[1], output6.beta[1], output7.beta[1], output8.beta[1], output9.beta[1], output10.beta[1]]

#Set up the display window for the plots and the plots themselves

fig1 = plt.figure()
fig2 = plt.figure()
fig3 = plt.figure()
fig4 = plt.figure()
fig5 = plt.figure()
fig6 = plt.figure()
fig7 = plt.figure()
fig8 = plt.figure()
fig9 = plt.figure()
fig10 = plt.figure()
#fig11 = plt.figure()
#fig12 = plt.figure()
#fig13 = plt.figure()
#fig14 = plt.figure()
#fig15 = plt.figure()
#fig16 = plt.figure()
#fig17 = plt.figure()
#fig18 = plt.figure()
#fig19 = plt.figure()
#fig20 = plt.figure()
#fig21 = plt.figure()
#fig22 = plt.figure()


plot1_1=fig1.add_subplot(121, projection='3d')
plot1_2=fig1.add_subplot(122, projection='3d')

plot2_1=fig2.add_subplot(121, projection='3d')
plot2_2=fig2.add_subplot(122, projection='3d')

plot3_1=fig3.add_subplot(121, projection='3d')
plot3_2=fig3.add_subplot(122, projection='3d')

plot4_1=fig4.add_subplot(121, projection='3d')
plot4_2=fig4.add_subplot(122, projection='3d')

plot5_1=fig5.add_subplot(121, projection='3d')
plot5_2=fig5.add_subplot(122, projection='3d')

plot6_1=fig6.add_subplot(121, projection='3d')
plot6_2=fig6.add_subplot(122, projection='3d')

plot7_1=fig7.add_subplot(121, projection='3d')
plot7_2=fig7.add_subplot(122, projection='3d')

plot8_1=fig8.add_subplot(121, projection='3d')
plot8_2=fig8.add_subplot(122, projection='3d')

plot9_1=fig9.add_subplot(121, projection='3d')
plot9_2=fig9.add_subplot(122, projection='3d')

plot10_1=fig10.add_subplot(121, projection='3d')
plot10_2=fig10.add_subplot(122, projection='3d')

#plot11_1=fig11.add_subplot(121, projection='3d')
#plot11_2=fig11.add_subplot(122, projection='3d')

#plot12_1=fig12.add_subplot(121, projection='3d')
#plot12_2=fig12.add_subplot(122, projection='3d')

#plot13_1=fig13.add_subplot(121, projection='3d')
#plot13_2=fig13.add_subplot(122, projection='3d')

#plot14_1=fig14.add_subplot(121, projection='3d')
#plot14_2=fig14.add_subplot(122, projection='3d')

#plot15_1=fig15.add_subplot(121, projection='3d')
#plot15_2=fig15.add_subplot(122, projection='3d')

#plot16_1=fig16.add_subplot(121, projection='3d')
#plot16_2=fig16.add_subplot(122, projection='3d')

#plot17_1=fig17.add_subplot(121, projection='3d')
#plot17_2=fig17.add_subplot(122, projection='3d')

#plot18_1=fig18.add_subplot(121, projection='3d')
#plot18_2=fig18.add_subplot(122, projection='3d')

#plot19_1=fig19.add_subplot(121, projection='3d')
#plot19_2=fig19.add_subplot(122, projection='3d')

#plot20_1=fig20.add_subplot(121, projection='3d')
#plot20_2=fig20.add_subplot(122, projection='3d')

#plot21=fig21.add_subplot(111)
#plot22=fig22.add_subplot(111)


#plot1_1.set_title('LG 1-0 mode at 25mm')
#plot2_1.set_title('LG 1-0 mode at 150mm')
#plot3_1.set_title('LG 1-0 mode at 1100mm')

#plot21.set_title('Difference Squared/N vs. Distance')
#plot22.set_title('# of Standard Dev. Sq./N vs. Distance')

plot1_1.plot_surface(Y, X, maskedfit1, rstride = 12, cstride = 12, linewidth = 0.05, cmap = 'cool')
plot1_2.plot_surface(Y, X, maskeddata1, rstride = 4, cstride = 4, cmap = 'hot', linewidth = 0.05) 

plot2_1.plot_surface(Y, X, maskedfit2, rstride = 12, cstride = 12, linewidth = 0.05, cmap = 'cool')
plot2_2.plot_surface(Y, X, maskeddata2, rstride = 4, cstride = 4, cmap = 'hot', linewidth = 0.05) 

plot3_1.plot_surface(Y, X, maskedfit3, rstride = 12, cstride = 12, linewidth = 0.05, cmap = 'cool')
plot3_2.plot_surface(Y, X, maskeddata3, rstride = 4, cstride = 4, cmap = 'hot', linewidth = 0.05) 

plot4_1.plot_surface(Y, X, maskedfit4, rstride = 12, cstride = 12, linewidth = 0.05, cmap = 'cool')
plot4_2.plot_surface(Y, X, maskeddata4, rstride = 4, cstride = 4, cmap = 'hot', linewidth = 0.05) 

plot5_1.plot_surface(Y, X, maskedfit5, rstride = 12, cstride = 12, linewidth = 0.05, cmap = 'cool')
plot5_2.plot_surface(Y, X, maskeddata5, rstride = 4, cstride = 4, cmap = 'hot', linewidth = 0.05) 

#plot6_1.plot_surface(Y, X, maskedfit6, rstride = 12, cstride = 12, linewidth = 0.05, cmap = 'cool')
#plot6_2.plot_surface(Y, X, maskeddata6, rstride = 12, cstride = 12, cmap = 'hot', linewidth = 0.05) 

#plot7_1.plot_surface(Y, X, maskedfit7, rstride = 12, cstride = 12, linewidth = 0.05, cmap = 'cool')
#plot7_2.plot_surface(Y, X, maskeddata7, rstride = 12, cstride = 12, cmap = 'hot', linewidth = 0.05) 

#plot8_1.plot_surface(Y, X, maskedfit8, rstride = 12, cstride = 12, linewidth = 0.05, cmap = 'cool')
#plot8_2.plot_surface(Y, X, maskeddata8, rstride = 12, cstride = 12, cmap = 'hot', linewidth = 0.05)

#plot9_1.plot_surface(Y, X, maskedfit9, rstride = 12, cstride = 12, linewidth = 0.05, cmap = 'cool')
#plot9_2.plot_surface(Y, X, maskeddata9, rstride = 12, cstride = 12, cmap = 'hot', linewidth = 0.05) 

#plot10_1.plot_surface(Y, X, maskedfit10, rstride = 12, cstride = 12, linewidth = 0.05, cmap = 'cool')
#plot10_2.plot_surface(Y, X, maskeddata10, rstride = 12, cstride = 12, cmap = 'hot', linewidth = 0.05) 

#plot11_1.plot_surface(Y, X, maskedfit11, rstride = 12, cstride = 12, linewidth = 0.05, cmap = 'cool')
#plot11_2.plot_surface(Y, X, maskeddata11, rstride = 12, cstride = 12, cmap = 'hot', linewidth = 0.05) 

#plot12_1.plot_surface(Y, X, maskedfit12, rstride = 12, cstride = 12, linewidth = 0.05, cmap = 'cool')
#plot12_2.plot_surface(Y, X, maskeddata12, rstride = 12, cstride = 12, cmap = 'hot', linewidth = 0.05) 

#plot13_1.plot_surface(Y, X, maskedfit13, rstride = 12, cstride = 12, linewidth = 0.05, cmap = 'cool')
#plot13_2.plot_surface(Y, X, maskeddata13, rstride = 12, cstride = 12, cmap = 'hot', linewidth = 0.05) 

#plot14_1.plot_surface(Y, X, maskedfit14, rstride = 12, cstride = 12, linewidth = 0.05, cmap = 'cool')
#plot14_2.plot_surface(Y, X, maskeddata14, rstride = 12, cstride = 12, cmap = 'hot', linewidth = 0.05) 

#plot15_1.plot_surface(Y, X, maskedfit15, rstride = 12, cstride = 12, linewidth = 0.05, cmap = 'cool')
#plot15_2.plot_surface(Y, X, maskeddata15, rstride = 12, cstride = 12, cmap = 'hot', linewidth = 0.05) 

#plot16_1.plot_surface(Y, X, maskedfit16, rstride = 12, cstride = 12, linewidth = 0.05, cmap = 'cool')
#plot16_2.plot_surface(Y, X, maskeddata16, rstride = 12, cstride = 12, cmap = 'hot', linewidth = 0.05) 

#plot17_1.plot_surface(Y, X, maskedfit17, rstride = 12, cstride = 12, linewidth = 0.05, cmap = 'cool')
#plot17_2.plot_surface(Y, X, maskeddata17, rstride = 12, cstride = 12, cmap = 'hot', linewidth = 0.05) 

#plot18_1.plot_surface(Y, X, maskedfit18, rstride = 12, cstride = 12, linewidth = 0.05, cmap = 'cool')
#plot18_2.plot_surface(Y, X, maskeddata18, rstride = 12, cstride = 12, cmap = 'hot', linewidth = 0.05) 

#plot19_1.plot_surface(Y, X, maskedfit19, rstride = 12, cstride = 12, linewidth = 0.05, cmap = 'cool')
#plot19_2.plot_surface(Y, X, maskeddata19, rstride = 12, cstride = 12, cmap = 'hot', linewidth = 0.05) 

#plot20_1.plot_surface(Y, X, maskedfit20, rstride = 12, cstride = 12, linewidth = 0.05, cmap = 'cool')
#plot20_2.plot_surface(Y, X, maskeddata20, rstride = 12, cstride = 12, cmap = 'hot', linewidth = 0.05) 

#plot21.scatter(ranges, np.array(adj_chi1))

#plot21.plot(xfit, Fit_out51)

#prints and labels all five parameters in the terminal, generates the plot in a new window.

plt.show()

print (output1.beta[3], output1.beta[4])
print (output2.beta[3], output2.beta[4])
print (output3.beta[3], output3.beta[4])
print (output4.beta[3], output4.beta[4])
print (output5.beta[3], output5.beta[4])
#print (output6.beta[3], output6.beta[4])
#print (output7.beta[3], output7.beta[4])
#print (output8.beta[3], output8.beta[4])
#print (output9.beta[3], output9.beta[4])
#print (output10.beta[3], output10.beta[4])

print 'adjusted Chi Squared'
print adj_chi1
print 'Mode Purity Statistic'
print Mode_Purity
print 'radii (pixels)'
#print diameters


####Library of Laguerre Polynomials for substitution in Z2
## 1, 0       1
## 1, 1       (2 - ((2*(X**2 + Y**2))/w**2))**2
## 2, 1       (3 - ((2*(X**2 + Y**2))/w**2))**2
## 5, 0       1
## 

