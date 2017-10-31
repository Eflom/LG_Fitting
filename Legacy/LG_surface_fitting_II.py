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

np.set_printoptions(threshold=np.nan)

#specify crop vallues for the data -- IMPORTANT!! the data must be a square
xmin = 350
xmax = 800 
ymin = 550
ymax = 1000 
x0 = (xmin + xmax)/2
y0 = (ymin + ymax)/2
ranges = [00, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285]

#debug use only, used for making simulated LG data 'noisy'
xerr = scipy.random.random(200)
yerr = scipy.random.random(200)
zerr = scipy.random.random(200)*10.0 -5.0

#read in data from file, cropping it using the values above

data1_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/1_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data1_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/1_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data1_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/1_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data1_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/1_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data1_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/1_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data1_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/1_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data1_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/1_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data1_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/1_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data1_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/1_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data1_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/1_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))


data2_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/2_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data2_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/2_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data2_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/2_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data2_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/2_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data2_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/2_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data2_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/2_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data2_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/2_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data2_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/2_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data2_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/2_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data2_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/2_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))


data3_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/3_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data3_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/3_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data3_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/3_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data3_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/3_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data3_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/3_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data3_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/3_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data3_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/3_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data3_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/3_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data3_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/3_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data3_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/3_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))


data4_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/4_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data4_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/4_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data4_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/4_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data4_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/4_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data4_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/4_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data4_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/4_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data4_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/4_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data4_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/4_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data4_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/4_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data4_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/4_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))


data5_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/5_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data5_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/5_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data5_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/5_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data5_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/5_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data5_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/5_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data5_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/5_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data5_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/5_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data5_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/5_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data5_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/5_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data5_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/5_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))


data6_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/6_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data6_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/6_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data6_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/6_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data6_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/6_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data6_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/6_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data6_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/6_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data6_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/6_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data6_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/6_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data6_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/6_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data6_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/6_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))


data7_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/7_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data7_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/7_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data7_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/7_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data7_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/7_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data7_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/7_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data7_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/7_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data7_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/7_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data7_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/7_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data7_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/7_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data7_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/7_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))


data8_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/8_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data8_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/8_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data8_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/8_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data8_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/8_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data8_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/8_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data8_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/8_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data8_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/8_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data8_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/8_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data8_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/8_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data8_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/8_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data9_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/9_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data9_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/9_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data9_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/9_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data9_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/9_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data9_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/9_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data9_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/9_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data9_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/9_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data9_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/9_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data9_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/9_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data9_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/9_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data10_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/10_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data10_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/10_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data10_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/10_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data10_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/10_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data10_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/10_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data10_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/10_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data10_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/10_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data10_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/10_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data10_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/10_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data10_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/10_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data11_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/11_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data11_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/11_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data11_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/11_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data11_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/11_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data11_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/11_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data11_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/11_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data11_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/11_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data11_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/11_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data11_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/11_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data11_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/11_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data12_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/12_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data12_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/12_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data12_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/12_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data12_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/12_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data12_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/12_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data12_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/12_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data12_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/12_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data12_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/12_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data12_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/12_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data12_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/12_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data13_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/13_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data13_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/13_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data13_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/13_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data13_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/13_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data13_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/13_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data13_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/13_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data13_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/13_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data13_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/13_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data13_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/13_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data13_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/13_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data14_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/14_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data14_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/14_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data14_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/14_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data14_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/14_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data14_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/14_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data14_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/14_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data14_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/14_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data14_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/14_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data14_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/14_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data14_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/14_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data15_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/15_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data15_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/15_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data15_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/15_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data15_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/15_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data15_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/15_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data15_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/15_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data15_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/15_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data15_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/15_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data15_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/15_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data15_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/15_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data16_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/16_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data16_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/16_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data16_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/16_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data16_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/16_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data16_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/16_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data16_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/16_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data16_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/16_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data16_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/16_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data16_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/16_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data16_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/16_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data17_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/17_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data17_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/17_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data17_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/17_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data17_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/17_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data17_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/17_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data17_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/17_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data17_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/17_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data17_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/17_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data17_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/17_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data17_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/17_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data18_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/18_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data18_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/18_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data18_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/18_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data18_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/18_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data18_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/18_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data18_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/18_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data18_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/18_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data18_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/18_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data18_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/18_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data18_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/18_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data19_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/19_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data19_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/19_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data19_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/19_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data19_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/19_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data19_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/19_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data19_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/19_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data19_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/19_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data19_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/19_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data19_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/19_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data19_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/19_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data20_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/20_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data20_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/20_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data20_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/20_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data20_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/20_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data20_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/20_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data20_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/20_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data20_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/20_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data20_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/20_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data20_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/20_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data20_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_11/20_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data1 = (data1_1 + data1_2 + data1_3 + data1_4 + data1_5 + data1_6 + data1_7 + data1_8 + data1_9 + data1_10)/10.
data2 = (data2_1 + data2_2 + data2_3 + data2_4 + data2_5 + data2_6 + data2_7 + data2_8 + data2_9 + data2_10)/10.
data3 = (data3_1 + data3_2 + data3_3 + data3_4 + data3_5 + data3_6 + data3_7 + data3_8 + data3_9 + data3_10)/10.
data4 = (data4_1 + data4_2 + data4_3 + data4_4 + data4_5 + data4_6 + data4_7 + data4_8 + data4_9 + data4_10)/10.
data5 = (data5_1 + data5_2 + data5_3 + data5_4 + data5_5 + data5_6 + data5_7 + data5_8 + data5_9 + data5_10)/10.
data6 = (data6_1 + data6_2 + data6_3 + data6_4 + data6_5 + data6_6 + data6_7 + data6_8 + data6_9 + data6_10)/10.
data7 = (data7_1 + data7_2 + data7_3 + data7_4 + data7_5 + data7_6 + data7_7 + data7_8 + data7_9 + data7_10)/10.
data8 = (data8_1 + data8_2 + data8_3 + data8_4 + data8_5 + data8_6 + data8_7 + data8_8 + data8_9 + data8_10)/10.
data9 = (data9_1 + data9_2 + data9_3 + data9_4 + data9_5 + data9_6 + data9_7 + data9_8 + data9_9 + data9_10)/10.
data10 = (data10_1 + data10_2 + data10_3 + data10_4 + data10_5 + data10_6 + data10_7 + data10_8 + data10_9 + data10_10)/10.
data11 = (data11_1 + data11_2 + data11_3 + data11_4 + data11_5 + data11_6 + data11_7 + data11_8 + data11_9 + data11_10)/10.
data12 = (data12_1 + data12_2 + data12_3 + data12_4 + data12_5 + data12_6 + data12_7 + data12_8 + data12_9 + data12_10)/10.
data13 = (data13_1 + data13_2 + data13_3 + data13_4 + data13_5 + data13_6 + data13_7 + data13_8 + data13_9 + data13_10)/10.
data14 = (data14_1 + data14_2 + data14_3 + data14_4 + data14_5 + data14_6 + data14_7 + data14_8 + data14_9 + data14_10)/10.
data15 = (data15_1 + data15_2 + data15_3 + data15_4 + data15_5 + data15_6 + data15_7 + data15_8 + data15_9 + data15_10)/10.
data16 = (data16_1 + data16_2 + data16_3 + data16_4 + data16_5 + data16_6 + data16_7 + data16_8 + data16_9 + data16_10)/10.
data17 = (data17_1 + data17_2 + data17_3 + data17_4 + data17_5 + data17_6 + data17_7 + data17_8 + data17_9 + data17_10)/10.
data18 = (data18_1 + data18_2 + data18_3 + data18_4 + data18_5 + data18_6 + data18_7 + data18_8 + data18_9 + data18_10)/10.
data19 = (data19_1 + data19_2 + data19_3 + data19_4 + data19_5 + data19_6 + data19_7 + data19_8 + data19_9 + data19_10)/10.
data20 = (data20_1 + data20_2 + data20_3 + data20_4 + data20_5 + data20_6 + data20_7 + data20_8 + data20_9 + data20_10)/10.

data1_sd = np.std([data1_1, data1_2, data1_3, data1_4, data1_5, data1_6, data1_7, data1_8, data1_9, data1_10], axis = 0)
data2_sd = np.std([data2_1, data2_2, data2_3, data2_4, data2_5, data2_6, data2_7, data2_8, data2_9, data2_10], axis = 0)
data3_sd = np.std([data3_1, data3_2, data3_3, data3_4, data3_5, data3_6, data3_7, data3_8, data3_9, data3_10], axis = 0)
data4_sd = np.std([data4_1, data4_2, data4_3, data4_4, data4_5, data4_6, data4_7, data4_8, data4_9, data4_10], axis = 0)
data5_sd = np.std([data5_1, data5_2, data5_3, data5_4, data5_5, data5_6, data5_7, data5_8, data5_9, data5_10], axis = 0)
data6_sd = np.std([data6_1, data6_2, data6_3, data6_4, data6_5, data6_6, data6_7, data6_8, data6_9, data6_10], axis = 0)
data7_sd = np.std([data7_1, data7_2, data7_3, data7_4, data7_5, data7_6, data7_7, data7_8, data7_9, data7_10], axis = 0)
data8_sd = np.std([data8_1, data8_2, data8_3, data8_4, data8_5, data8_6, data8_7, data8_8, data8_9, data8_10], axis = 0)
data9_sd = np.std([data9_1, data9_2, data9_3, data9_4, data9_5, data9_6, data9_7, data9_8, data9_9, data9_10], axis = 0)
data10_sd = np.std([data10_1, data10_2, data10_3, data10_4, data10_5, data10_6, data10_7, data10_8, data10_9, data10_10], axis = 0)
data11_sd = np.std([data11_1, data11_2, data11_3, data11_4, data11_5, data11_6, data11_7, data11_8, data11_9, data11_10], axis = 0)
data12_sd = np.std([data12_1, data12_2, data12_3, data12_4, data12_5, data12_6, data12_7, data12_8, data12_9, data12_10], axis = 0)
data13_sd = np.std([data13_1, data13_2, data13_3, data13_4, data13_5, data13_6, data13_7, data13_8, data13_9, data13_10], axis = 0)
data14_sd = np.std([data14_1, data14_2, data14_3, data14_4, data14_5, data14_6, data14_7, data14_8, data14_9, data14_10], axis = 0)
data15_sd = np.std([data15_1, data15_2, data15_3, data15_4, data15_5, data15_6, data15_7, data15_8, data15_9, data15_10], axis = 0)
data16_sd = np.std([data16_1, data16_2, data16_3, data16_4, data16_5, data16_6, data16_7, data16_8, data16_9, data16_10], axis = 0)
data17_sd = np.std([data17_1, data17_2, data17_3, data17_4, data17_5, data17_6, data17_7, data17_8, data17_9, data17_10], axis = 0)
data18_sd = np.std([data18_1, data18_2, data18_3, data18_4, data18_5, data18_6, data18_7, data18_8, data18_9, data18_10], axis = 0)
data19_sd = np.std([data19_1, data19_2, data19_3, data19_4, data19_5, data19_6, data19_7, data19_8, data19_9, data19_10], axis = 0)
data20_sd = np.std([data20_1, data20_2, data20_3, data20_4, data20_5, data20_6, data20_7, data20_8, data20_9, data20_10], axis = 0)

#generate a regular x-y space as independent variables

x = np.arange(xmin, xmax, 1)
y = np.arange(ymin, ymax, 1)

#specify radial and azimuthal modes of the LG Beam

l=1.
p=0.

#generate the 2D grid for plotting

X, Y = np.meshgrid(x - x0, y -y0)

#define these terms b/c I'm lazy and don't want to keep typing them

Q = x - x0
W = y - y0

#Debug use only, used for generating simulated LG cross-sections

A = 100.
w = 30.
B = 1

#Here's the fit function, broken into three terms for the sake of debugging. Used both for generating simulated cross-sections
Z1 = np.sqrt(A)*(2.*(X**2. + Y**2.)/(w**2.))**l
Z2 = (3. - ((2.*(X**2. + Y**2.))/w**2.))**2.  ##use Mathematica LaguerreL [bottom, top, function] to generate this term
Z3 = np.exp((-2.)*((X**2. + Y**2.)/w**2.)) 
Z4 = Z1*Z2*Z3 + B

#routine to mask data to remove bottom fraction of values
Crop_range = 0.25

zmin1 = np.max(np.max(data1))*Crop_range
zmin2 = np.max(np.max(data2))*Crop_range
zmin3 = np.max(np.max(data3))*Crop_range
zmin4 = np.max(np.max(data4))*Crop_range
zmin5 = np.max(np.max(data5))*Crop_range
zmin6 = np.max(np.max(data6))*Crop_range
zmin7 = np.max(np.max(data7))*Crop_range
zmin8 = np.max(np.max(data8))*Crop_range
zmin9 = np.max(np.max(data8))*Crop_range
zmin10 = np.max(np.max(data8))*Crop_range
zmin11 = np.max(np.max(data8))*Crop_range
zmin12 = np.max(np.max(data8))*Crop_range
zmin13 = np.max(np.max(data8))*Crop_range
zmin14 = np.max(np.max(data8))*Crop_range
zmin15 = np.max(np.max(data8))*Crop_range
zmin16 = np.max(np.max(data8))*Crop_range
zmin17 = np.max(np.max(data8))*Crop_range
zmin18 = np.max(np.max(data8))*Crop_range
zmin19 = np.max(np.max(data8))*Crop_range
zmin20 = np.max(np.max(data8))*Crop_range

maskeddata1 = np.where(data1 > zmin1, data1, 1)
maskedfitdata1 = np.where(data1 > zmin1, data1, 0.0)

maskeddata2 = np.where(data2 > zmin2, data2, 1)
maskedfitdata2 = np.where(data2 > zmin2, data2, 0.0)

maskeddata3 = np.where(data3 > zmin3, data3, 1)
maskedfitdata3 = np.where(data3 > zmin3, data3, 0.0)

maskeddata4 = np.where(data4 > zmin4, data4, 1)
maskedfitdata4 = np.where(data4 > zmin4, data4, 0.0)

maskeddata4 = np.where(data4 > zmin4, data4, 1)
maskedfitdata4 = np.where(data4 > zmin4, data4, 0.0)

maskeddata5 = np.where(data5 > zmin5, data5, 1)
maskedfitdata5 = np.where(data5 > zmin5, data5, 0.0)

maskeddata6 = np.where(data6 > zmin6, data6, 1)
maskedfitdata6 = np.where(data6 > zmin6, data6, 0.0)

maskeddata7 = np.where(data7 > zmin7, data7, 1)
maskedfitdata7 = np.where(data7 > zmin7, data7, 0.0)

maskeddata8 = np.where(data8 > zmin8, data8, 1)
maskedfitdata8 = np.where(data8 > zmin8, data8, 0.0)

maskeddata9 = np.where(data9 > zmin9, data9, 1)
maskedfitdata9 = np.where(data9 > zmin9, data9, 0.0)

maskeddata10 = np.where(data10 > zmin10, data10, 1)
maskedfitdata10 = np.where(data10 > zmin10, data10, 0.0)

maskeddata11 = np.where(data11 > zmin11, data11, 1)
maskedfitdata11 = np.where(data11 > zmin11, data11, 0.0)

maskeddata12 = np.where(data12 > zmin12, data12, 1)
maskedfitdata12 = np.where(data12 > zmin12, data12, 0.0)

maskeddata13 = np.where(data13 > zmin13, data13, 1)
maskedfitdata13 = np.where(data13 > zmin13, data13, 0.0)

maskeddata14 = np.where(data14 > zmin14, data14, 1)
maskedfitdata14 = np.where(data14 > zmin14, data14, 0.0)

maskeddata15 = np.where(data15 > zmin15, data15, 1)
maskedfitdata15 = np.where(data15 > zmin15, data15, 0.0)

maskeddata16 = np.where(data16 > zmin16, data16, 1)
maskedfitdata16 = np.where(data16 > zmin16, data16, 0.0)

maskeddata17 = np.where(data17 > zmin17, data17, 1)
maskedfitdata17 = np.where(data17 > zmin17, data17, 0.0)

maskeddata18 = np.where(data18 > zmin18, data18, 1)
maskedfitdata18 = np.where(data18 > zmin18, data18, 0.0)

maskeddata19 = np.where(data19 > zmin19, data19, 1)
maskedfitdata19 = np.where(data19 > zmin19, data19, 0.0)

maskeddata20 = np.where(data20 > zmin20, data20, 1)
maskedfitdata20 = np.where(data20 > zmin20, data20, 0.0)

N1 = np.count_nonzero(maskeddata1)
N2 = np.count_nonzero(maskeddata2)
N3 = np.count_nonzero(maskeddata3)
N4 = np.count_nonzero(maskeddata4)
N5 = np.count_nonzero(maskeddata5)
N6 = np.count_nonzero(maskeddata6)
N7 = np.count_nonzero(maskeddata7)
N8 = np.count_nonzero(maskeddata8)
N9 = np.count_nonzero(maskeddata9)
N10 = np.count_nonzero(maskeddata10)
N11 = np.count_nonzero(maskeddata11)
N12 = np.count_nonzero(maskeddata12)
N13 = np.count_nonzero(maskeddata13)
N14 = np.count_nonzero(maskeddata14)
N15 = np.count_nonzero(maskeddata15)
N16 = np.count_nonzero(maskeddata16)
N17 = np.count_nonzero(maskeddata17)
N18 = np.count_nonzero(maskeddata18)
N19 = np.count_nonzero(maskeddata19)
N20 = np.count_nonzero(maskeddata20)



#define the funciton in terms of the 5 paramters, so that the ODR can process them
def function(params, data):
	scale = params[0] #= 9000
	baseline = params[2] #= 850
	width = params[1] #= 30
	y_0 = params[3] #=0
	x_0 = params[4] #=20
		
	return ((((scale)*((2.)*((X - x_0)**2. + (Y - y_0)**2.)/(width)**2.)))**l)*(np.exp((-2.)*(((X - x_0)**2. + (Y - y_0)**2.)/(width)**2.))) + baseline


#The meat of the ODR program. set "guesses" to a rough initial guess for the data <-- IMPORTANT

myData1 = Data([Q, W], data1)
myModel = Model(function)
guesses1 = [25000, 20, 850, -50, -50] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr1 = ODR(myData1, myModel, guesses1, maxit=1000)
odr1.set_job(fit_type=2)
output1 = odr1.run()
#output1.pprint()
Fit_out1 = (((((output1.beta[0]))*(2.*((X - output1.beta[4])**2. + (Y - output1.beta[3])**2.)/(output1.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output1.beta[4])**2. + (Y - output1.beta[3])**2.)/(output1.beta[1])**2.))) + output1.beta[2]
print 'done1'

myData2 = Data([Q, W], data2)
myModel = Model(function)
guesses2 = [25000, 20, 800, -50, -30] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr2 = ODR(myData2, myModel, guesses1, maxit=100)
odr2.set_job(fit_type=2)
output2 = odr2.run()
#output2.pprint()
Fit_out2 = (((((output2.beta[0]))*(2.*((X - output2.beta[4])**2. + (Y - output2.beta[3])**2.)/(output2.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output2.beta[4])**2. + (Y - output2.beta[3])**2.)/(output2.beta[1])**2.))) + output2.beta[2]
print 'done2'

myData3 = Data([Q, W], data3)
myModel = Model(function)
guesses3 = [25000, 20, 850, 0, 0] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr3 = ODR(myData3, myModel, guesses3, maxit=100)
odr3.set_job(fit_type=2)
output3 = odr3.run()
#output3.pprint()
Fit_out3 = (((((output3.beta[0]))*(2.*((X - output3.beta[4])**2. + (Y - output3.beta[3])**2.)/(output3.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output3.beta[4])**2. + (Y - output3.beta[3])**2.)/(output3.beta[1])**2.))) + output3.beta[2]
print 'done3'

myData4 = Data([Q, W], data4)
myModel = Model(function)
guesses4 = [25000, 20, 850, 0, 0] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr4 = ODR(myData4, myModel, guesses4, maxit=100)
odr4.set_job(fit_type=2)
output4 = odr4.run()
#output4.pprint()
Fit_out4 = (((((output4.beta[0]))*(2.*((X - output4.beta[4])**2. + (Y - output4.beta[3])**2.)/(output4.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output4.beta[4])**2. + (Y - output4.beta[3])**2.)/(output4.beta[1])**2.))) + output4.beta[2]
print 'done4'

myData5 = Data([Q, W], data5)
myModel = Model(function)
guesses5 = [25000, 20, 850, 0, 0] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr5 = ODR(myData5, myModel, guesses5, maxit=100)
odr5.set_job(fit_type=2)
output5 = odr5.run()
#output5.pprint()
Fit_out5 = (((((output5.beta[0]))*(2.*((X - output5.beta[4])**2. + (Y - output5.beta[3])**2.)/(output5.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output5.beta[4])**2. + (Y - output5.beta[3])**2.)/(output5.beta[1])**2.))) + output5.beta[2]
print 'done5'

myData6 = Data([Q, W], data6)
myModel = Model(function)
guesses6 = [25000, 20, 850, 0, 0] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr6 = ODR(myData6, myModel, guesses6, maxit=100)
odr6.set_job(fit_type=2)
output6 = odr6.run()
#output6.pprint()
Fit_out6 = (((((output6.beta[0]))*(2.*((X - output6.beta[4])**2. + (Y - output6.beta[3])**2.)/(output6.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output6.beta[4])**2. + (Y - output6.beta[3])**2.)/(output6.beta[1])**2.))) + output6.beta[2]
print 'done6'

myData7 = Data([Q, W], data7)
myModel = Model(function)
guesses7 = [25000, 20, 850, 25, -20] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr7 = ODR(myData7, myModel, guesses7, maxit=100)
odr7.set_job(fit_type=2)
output7 = odr7.run()
#output7.pprint()
Fit_out7 = (((((output7.beta[0]))*(2.*((X - output7.beta[4])**2. + (Y - output7.beta[3])**2.)/(output7.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output7.beta[4])**2. + (Y - output7.beta[3])**2.)/(output7.beta[1])**2.))) + output7.beta[2]
print 'done7'

myData8 = Data([Q, W], data8)
myModel = Model(function)
guesses8 = [25000, 20, 850, 0, 0] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr8 = ODR(myData8, myModel, guesses8, maxit=100)
odr8.set_job(fit_type=2)
output8 = odr8.run()
#output8.pprint()
Fit_out8 = (((((output8.beta[0]))*(2.*((X - output8.beta[4])**2. + (Y - output8.beta[3])**2.)/(output8.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output8.beta[4])**2. + (Y - output8.beta[3])**2.)/(output8.beta[1])**2.))) + output8.beta[2]
print 'done8'

myData9 = Data([Q, W], data9)
myModel = Model(function)
guesses9 = [25000, 20, 150, 00, 00] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr9 = ODR(myData9, myModel, guesses9, maxit=1000)
odr9.set_job(fit_type=2)
output9 = odr9.run()
#output1.pprint()
Fit_out9 = (((((output9.beta[0]))*(2.*((X - output9.beta[4])**2. + (Y - output9.beta[3])**2.)/(output9.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output9.beta[4])**2. + (Y - output9.beta[3])**2.)/(output9.beta[1])**2.))) + output9.beta[2]
print 'done9'

myData10 = Data([Q, W], data10)
myModel = Model(function)
guesses10 = [25000, 20, 850, 50, -20] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr10 = ODR(myData10, myModel, guesses10, maxit=1000)
odr10.set_job(fit_type=2)
output10 = odr10.run()
#output1.pprint()
Fit_out10 = (((((output10.beta[0]))*(2.*((X - output10.beta[4])**2. + (Y - output10.beta[3])**2.)/(output10.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output10.beta[4])**2. + (Y - output10.beta[3])**2.)/(output10.beta[1])**2.))) + output10.beta[2]
print 'done10'

myData11 = Data([Q, W], data11)
myModel = Model(function)
guesses11 = [25000, 20, 850, 75, -20] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr11 = ODR(myData11, myModel, guesses11, maxit=1000)
odr11.set_job(fit_type=2)
output11 = odr11.run()
#output1.pprint()
Fit_out11 = (((((output11.beta[0]))*(2.*((X - output11.beta[4])**2. + (Y - output11.beta[3])**2.)/(output11.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output11.beta[4])**2. + (Y - output11.beta[3])**2.)/(output11.beta[1])**2.))) + output11.beta[2]
print 'done11'

myData12 = Data([Q, W], data12)
myModel = Model(function)
guesses12 = [25000, 20, 850, 100, -50] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr12 = ODR(myData12, myModel, guesses12, maxit=1000)
odr12.set_job(fit_type=2)
output12 = odr12.run()
#output1.pprint()
Fit_out12 = (((((output12.beta[0]))*(2.*((X - output12.beta[4])**2. + (Y - output12.beta[3])**2.)/(output12.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output12.beta[4])**2. + (Y - output12.beta[3])**2.)/(output12.beta[1])**2.))) + output12.beta[2]
print 'done12'

myData13 = Data([Q, W], data13)
myModel = Model(function)
guesses13 = [25000, 20, 850, 120, -50] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr13 = ODR(myData13, myModel, guesses13, maxit=1000)
odr13.set_job(fit_type=2)
output13 = odr13.run()
#output1.pprint()
Fit_out13 = (((((output13.beta[0]))*(2.*((X - output13.beta[4])**2. + (Y - output13.beta[3])**2.)/(output13.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output13.beta[4])**2. + (Y - output13.beta[3])**2.)/(output13.beta[1])**2.))) + output13.beta[2]
print 'done13'

myData14 = Data([Q, W], data14)
myModel = Model(function)
guesses14 = [25000, 20, 850, 00, 00] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr14 = ODR(myData14, myModel, guesses14, maxit=1000)
odr14.set_job(fit_type=2)
output14 = odr14.run()
#output1.pprint()
Fit_out14 = (((((output14.beta[0]))*(2.*((X - output14.beta[4])**2. + (Y - output14.beta[3])**2.)/(output14.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output14.beta[4])**2. + (Y - output14.beta[3])**2.)/(output14.beta[1])**2.))) + output14.beta[2]
print 'done14'

myData15 = Data([Q, W], data15)
myModel = Model(function)
guesses15 = [25000, 20, 850, 75, -20] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr15 = ODR(myData15, myModel, guesses15, maxit=1000)
odr15.set_job(fit_type=2)
output15 = odr15.run()
#output1.pprint()
Fit_out15 = (((((output15.beta[0]))*(2.*((X - output15.beta[4])**2. + (Y - output15.beta[3])**2.)/(output15.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output15.beta[4])**2. + (Y - output15.beta[3])**2.)/(output15.beta[1])**2.))) + output15.beta[2]
print 'done15'

myData16 = Data([Q, W], data16)
myModel = Model(function)
guesses16 = [25000, 20, 850, 00, 00] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr16 = ODR(myData16, myModel, guesses16, maxit=1000)
odr16.set_job(fit_type=2)
output16 = odr16.run()
#output1.pprint()
Fit_out16 = (((((output16.beta[0]))*(2.*((X - output16.beta[4])**2. + (Y - output16.beta[3])**2.)/(output16.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output16.beta[4])**2. + (Y - output16.beta[3])**2.)/(output16.beta[1])**2.))) + output16.beta[2]
print 'done16'

myData17 = Data([Q, W], data17)
myModel = Model(function)
guesses17 = [25000, 20, 850, 00, 00] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr17 = ODR(myData17, myModel, guesses17, maxit=1000)
odr17.set_job(fit_type=2)
output17 = odr17.run()
#output1.pprint()
Fit_out17 = (((((output17.beta[0]))*(2.*((X - output17.beta[4])**2. + (Y - output17.beta[3])**2.)/(output17.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output17.beta[4])**2. + (Y - output17.beta[3])**2.)/(output17.beta[1])**2.))) + output17.beta[2]
print 'done17'

myData18 = Data([Q, W], data18)
myModel = Model(function)
guesses18 = [25000, 20, 850, 00, 00] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr18 = ODR(myData18, myModel, guesses18, maxit=1000)
odr18.set_job(fit_type=2)
output18 = odr18.run()
#output1.pprint()
Fit_out18 = (((((output18.beta[0]))*(2.*((X - output18.beta[4])**2. + (Y - output18.beta[3])**2.)/(output18.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output18.beta[4])**2. + (Y - output18.beta[3])**2.)/(output18.beta[1])**2.))) + output18.beta[2]
print 'done18'

myData19 = Data([Q, W], data19)
myModel = Model(function)
guesses19 = [25000, 20, 850, 00, 00] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr19 = ODR(myData19, myModel, guesses19, maxit=1000)
odr19.set_job(fit_type=2)
output19 = odr19.run()
#output19.pprint()
Fit_out19 = (((((output19.beta[0]))*(2.*((X - output19.beta[4])**2. + (Y - output19.beta[3])**2.)/(output19.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output19.beta[4])**2. + (Y - output19.beta[3])**2.)/(output19.beta[1])**2.))) + output19.beta[2]
print 'done19'

myData20 = Data([Q, W], data20)
myModel = Model(function)
guesses20 = [25000, 20, 850, 00, 00] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr20 = ODR(myData20, myModel, guesses20, maxit=1000)
odr20.set_job(fit_type=2)
output20 = odr20.run()
#output20.pprint()
Fit_out20 = (((((output20.beta[0]))*(2.*((X - output20.beta[4])**2. + (Y - output20.beta[3])**2.)/(output20.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output20.beta[4])**2. + (Y - output20.beta[3])**2.)/(output20.beta[1])**2.))) + output20.beta[2]
print 'done20'

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
maskedfit11 = np.where(data11 > zmin11, Fit_out11, 100)
maskedfit12 = np.where(data12 > zmin12, Fit_out12, 100)
maskedfit13 = np.where(data13 > zmin13, Fit_out13, 100)
maskedfit14 = np.where(data14 > zmin14, Fit_out14, 100)
maskedfit15 = np.where(data15 > zmin15, Fit_out15, 100)
maskedfit16 = np.where(data16 > zmin16, Fit_out16, 100)
maskedfit17 = np.where(data17 > zmin17, Fit_out17, 100)
maskedfit18 = np.where(data18 > zmin18, Fit_out18, 100)
maskedfit19 = np.where(data19 > zmin19, Fit_out19, 100)
maskedfit20 = np.where(data20 > zmin20, Fit_out20, 100)

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
Chisq11 = np.sum(np.sum((((maskeddata11 - maskedfit11))**2)/(maskedfit11)))
Chisq12 = np.sum(np.sum((((maskeddata12 - maskedfit12))**2)/(maskedfit12)))
Chisq13 = np.sum(np.sum((((maskeddata13 - maskedfit13))**2)/(maskedfit13)))
Chisq14 = np.sum(np.sum((((maskeddata14 - maskedfit14))**2)/(maskedfit14)))
Chisq15 = np.sum(np.sum((((maskeddata15 - maskedfit15))**2)/(maskedfit15)))
Chisq16 = np.sum(np.sum((((maskeddata16 - maskedfit16))**2)/(maskedfit16)))
Chisq17 = np.sum(np.sum((((maskeddata17 - maskedfit17))**2)/(maskedfit17)))
Chisq18 = np.sum(np.sum((((maskeddata18 - maskedfit18))**2)/(maskedfit18)))
Chisq19 = np.sum(np.sum((((maskeddata19 - maskedfit19))**2)/(maskedfit19)))
Chisq20 = np.sum(np.sum((((maskeddata20 - maskedfit20))**2)/(maskedfit20)))

zsq1 = np.sum(np.sum(((maskeddata1 - maskedfit1)/(data1_sd))**2))
zsq2 = np.sum(np.sum(((maskeddata2 - maskedfit2)/(data2_sd))**2))
zsq3 = np.sum(np.sum(((maskeddata3 - maskedfit3)/(data3_sd))**2))
zsq4 = np.sum(np.sum(((maskeddata4 - maskedfit4)/(data4_sd))**2))
zsq5 = np.sum(np.sum(((maskeddata5 - maskedfit5)/(data5_sd))**2))
zsq6 = np.sum(np.sum(((maskeddata6 - maskedfit6)/(data6_sd))**2))
zsq7 = np.sum(np.sum(((maskeddata7 - maskedfit7)/(data7_sd))**2))
zsq8 = np.sum(np.sum(((maskeddata8 - maskedfit8)/(data8_sd))**2))
zsq9 = np.sum(np.sum(((maskeddata9 - maskedfit9)/(data9_sd))**2))
zsq10 = np.sum(np.sum(((maskeddata10 - maskedfit10)/(data10_sd))**2))
zsq11 = np.sum(np.sum(((maskeddata11 - maskedfit11)/(data11_sd))**2))
zsq12 = np.sum(np.sum(((maskeddata12 - maskedfit12)/(data12_sd))**2))
zsq13 = np.sum(np.sum(((maskeddata13 - maskedfit13)/(data13_sd))**2))
zsq14 = np.sum(np.sum(((maskeddata14 - maskedfit14)/(data14_sd))**2))
zsq15 = np.sum(np.sum(((maskeddata15 - maskedfit15)/(data15_sd))**2))
zsq16 = np.sum(np.sum(((maskeddata16 - maskedfit16)/(data16_sd))**2))
zsq17 = np.sum(np.sum(((maskeddata17 - maskedfit17)/(data17_sd))**2))
zsq18 = np.sum(np.sum(((maskeddata18 - maskedfit18)/(data18_sd))**2))
zsq19 = np.sum(np.sum(((maskeddata19 - maskedfit19)/(data19_sd))**2))
zsq20 = np.sum(np.sum(((maskeddata20 - maskedfit20)/(data20_sd))**2))


scale1 = output1.beta[0]
scale2 = output2.beta[0]
scale3 = output3.beta[0]
scale4 = output4.beta[0]
scale5 = output5.beta[0]
scale6 = output6.beta[0]
scale7 = output7.beta[0]
scale8 = output8.beta[0]
scale9 = output9.beta[0]
scale10 = output10.beta[0]
scale11 = output11.beta[0]
scale12 = output12.beta[0]
scale13 = output13.beta[0]
scale14 = output14.beta[0]
scale15 = output15.beta[0]
scale16 = output16.beta[0]
scale17 = output17.beta[0]
scale18 = output18.beta[0]
scale19 = output19.beta[0]
scale20 = output20.beta[0]

adj_chi1 = [Chisq1/N1, Chisq2/N2, Chisq3/N3, Chisq4/N4, Chisq5/N5, Chisq6/N6, Chisq7/N7, Chisq8/N8, Chisq9/N9, Chisq10/N10, Chisq11/N11, Chisq12/N12, Chisq13/N13, Chisq14/N14, Chisq15/N15, Chisq16/N16, Chisq17/N17, Chisq18/N18, Chisq19/N19, Chisq20/N20]
adj_z = [zsq1/N1, zsq2/N2, zsq3/N3, zsq4/N4, zsq5/N5, zsq6/N6, zsq7/N7, zsq8/N8, zsq9/N9, zsq10/N10, zsq11/N11, zsq12/N12, zsq13/N13, zsq14/N14, zsq15/N15, zsq16/N16, zsq17/N17, zsq18/N18, zsq19/N19, zsq20/N20]

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
fig11 = plt.figure()
fig12 = plt.figure()
fig13 = plt.figure()
fig14 = plt.figure()
fig15 = plt.figure()
fig16 = plt.figure()
fig17 = plt.figure()
fig18 = plt.figure()
fig19 = plt.figure()
fig20 = plt.figure()
fig21 = plt.figure()
fig22 = plt.figure()

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

plot11_1=fig11.add_subplot(121, projection='3d')
plot11_2=fig11.add_subplot(122, projection='3d')

plot12_1=fig12.add_subplot(121, projection='3d')
plot12_2=fig12.add_subplot(122, projection='3d')

plot13_1=fig13.add_subplot(121, projection='3d')
plot13_2=fig13.add_subplot(122, projection='3d')

plot14_1=fig14.add_subplot(121, projection='3d')
plot14_2=fig14.add_subplot(122, projection='3d')

plot15_1=fig15.add_subplot(121, projection='3d')
plot15_2=fig15.add_subplot(122, projection='3d')

plot16_1=fig16.add_subplot(121, projection='3d')
plot16_2=fig16.add_subplot(122, projection='3d')

plot17_1=fig17.add_subplot(121, projection='3d')
plot17_2=fig17.add_subplot(122, projection='3d')

plot18_1=fig18.add_subplot(121, projection='3d')
plot18_2=fig18.add_subplot(122, projection='3d')

plot19_1=fig19.add_subplot(121, projection='3d')
plot19_2=fig19.add_subplot(122, projection='3d')

plot20_1=fig20.add_subplot(121, projection='3d')
plot20_2=fig20.add_subplot(122, projection='3d')

plot21=fig21.add_subplot(121)
plot22=fig21.add_subplot(122)

plot1_1.set_title('Plane Fit 00mm')
plot1_2.set_title('Data 00mm')

plot2_1.set_title('Plane Fit 15mm')
plot2_2.set_title('Data 15mm')

plot3_1.set_title('Plane Fit 30mm')
plot3_2.set_title('Data 30mm')

plot4_1.set_title('Plane Fit 45mm')
plot4_2.set_title('Data 45mm')

plot5_1.set_title('Plane Fit 60mm')
plot5_2.set_title('Data 60mm')

plot6_1.set_title('Plane Fit 75mm')
plot6_2.set_title('Data 75mm')

plot7_1.set_title('Plane Fit 90mm')
plot7_2.set_title('Data 90mm')

plot8_1.set_title('Plane Fit 105mm')
plot8_2.set_title('Data 105mm')

plot21.set_title('Chi Sq./N vs. Distance')
plot22.set_title('Z-score Sq. vs. Distance')

plot1_1.plot_surface(Y, X, maskedfit1, rstride = 8, cstride = 8, linewidth = 0.05, cmap = 'cool')
plot1_2.plot_surface(Y, X, maskeddata1, rstride = 8, cstride = 8, cmap = 'hot', linewidth = 0.05) 

plot2_1.plot_surface(Y, X, maskedfit2, rstride = 8, cstride = 8, linewidth = 0.05, cmap = 'cool')
plot2_2.plot_surface(Y, X, maskeddata2, rstride = 8, cstride = 8, cmap = 'hot', linewidth = 0.05) 

plot3_1.plot_surface(Y, X, maskedfit3, rstride = 8, cstride = 8, linewidth = 0.05, cmap = 'cool')
plot3_2.plot_surface(Y, X, maskeddata3, rstride = 8, cstride = 8, cmap = 'hot', linewidth = 0.05) 

plot4_1.plot_surface(Y, X, maskedfit4, rstride = 8, cstride = 8, linewidth = 0.05, cmap = 'cool')
plot4_2.plot_surface(Y, X, maskeddata4, rstride = 8, cstride = 8, cmap = 'hot', linewidth = 0.05) 

plot5_1.plot_surface(Y, X, maskedfit5, rstride = 8, cstride = 8, linewidth = 0.05, cmap = 'cool')
plot5_2.plot_surface(Y, X, maskeddata5, rstride = 8, cstride = 8, cmap = 'hot', linewidth = 0.05) 

plot6_1.plot_surface(Y, X, maskedfit6, rstride = 8, cstride = 8, linewidth = 0.05, cmap = 'cool')
plot6_2.plot_surface(Y, X, maskeddata6, rstride = 8, cstride = 8, cmap = 'hot', linewidth = 0.05) 

plot7_1.plot_surface(Y, X, maskedfit7, rstride = 8, cstride = 8, linewidth = 0.05, cmap = 'cool')
plot7_2.plot_surface(Y, X, maskeddata7, rstride = 8, cstride = 8, cmap = 'hot', linewidth = 0.05) 

plot8_1.plot_surface(Y, X, maskedfit8, rstride = 8, cstride = 8, linewidth = 0.05, cmap = 'cool')
plot8_2.plot_surface(Y, X, maskeddata8, rstride = 8, cstride = 8, cmap = 'hot', linewidth = 0.05)

plot9_1.plot_surface(Y, X, maskedfit9, rstride = 8, cstride = 8, linewidth = 0.05, cmap = 'cool')
plot9_2.plot_surface(Y, X, maskeddata9, rstride = 8, cstride = 8, cmap = 'hot', linewidth = 0.05) 

plot10_1.plot_surface(Y, X, maskedfit10, rstride = 8, cstride = 8, linewidth = 0.05, cmap = 'cool')
plot10_2.plot_surface(Y, X, maskeddata10, rstride = 8, cstride = 8, cmap = 'hot', linewidth = 0.05) 

plot11_1.plot_surface(Y, X, maskedfit11, rstride = 8, cstride = 8, linewidth = 0.05, cmap = 'cool')
plot11_2.plot_surface(Y, X, maskeddata11, rstride = 8, cstride = 8, cmap = 'hot', linewidth = 0.05) 

plot12_1.plot_surface(Y, X, maskedfit12, rstride = 8, cstride = 8, linewidth = 0.05, cmap = 'cool')
plot12_2.plot_surface(Y, X, maskeddata12, rstride = 8, cstride = 8, cmap = 'hot', linewidth = 0.05) 

plot13_1.plot_surface(Y, X, maskedfit13, rstride = 8, cstride = 8, linewidth = 0.05, cmap = 'cool')
plot13_2.plot_surface(Y, X, maskeddata13, rstride = 8, cstride = 8, cmap = 'hot', linewidth = 0.05) 

plot14_1.plot_surface(Y, X, maskedfit14, rstride = 8, cstride = 8, linewidth = 0.05, cmap = 'cool')
plot14_2.plot_surface(Y, X, maskeddata14, rstride = 8, cstride = 8, cmap = 'hot', linewidth = 0.05) 

plot15_1.plot_surface(Y, X, maskedfit15, rstride = 8, cstride = 8, linewidth = 0.05, cmap = 'cool')
plot15_2.plot_surface(Y, X, maskeddata15, rstride = 8, cstride = 8, cmap = 'hot', linewidth = 0.05) 

plot16_1.plot_surface(Y, X, maskedfit16, rstride = 8, cstride = 8, linewidth = 0.05, cmap = 'cool')
plot16_2.plot_surface(Y, X, maskeddata16, rstride = 8, cstride = 8, cmap = 'hot', linewidth = 0.05) 

plot17_1.plot_surface(Y, X, maskedfit17, rstride = 8, cstride = 8, linewidth = 0.05, cmap = 'cool')
plot17_2.plot_surface(Y, X, maskeddata17, rstride = 8, cstride = 8, cmap = 'hot', linewidth = 0.05) 

plot18_1.plot_surface(Y, X, maskedfit18, rstride = 8, cstride = 8, linewidth = 0.05, cmap = 'cool')
plot18_2.plot_surface(Y, X, maskeddata18, rstride = 8, cstride = 8, cmap = 'hot', linewidth = 0.05) 

plot19_1.plot_surface(Y, X, maskedfit19, rstride = 8, cstride = 8, linewidth = 0.05, cmap = 'cool')
plot19_2.plot_surface(Y, X, maskeddata19, rstride = 8, cstride = 8, cmap = 'hot', linewidth = 0.05) 

plot20_1.plot_surface(Y, X, maskedfit20, rstride = 8, cstride = 8, linewidth = 0.05, cmap = 'cool')
plot20_2.plot_surface(Y, X, maskeddata20, rstride = 8, cstride = 8, cmap = 'hot', linewidth = 0.05) 


plot21.scatter(ranges, adj_chi1)
plot22.scatter(ranges, adj_z)

def function2(params2, ranges):
	constant = params2[0]
	linear = params2[1]
		
	return (constant + linear*ranges)
	
def function3(params4, ranges):
	constant = params4[0]
	linear = params4[1]
	quadratic = params4[2]
	
	
	return (params4[0] + params4[1]*(ranges) + params4[2]*(ranges**2.0))
	
xfit = np.arange(1, 1000, 0.5)
	
myData21 = RealData(ranges, adj_chi1, sx = 5, sy = 1)
myModel2 = Model(function2)
guesses21 = [0.001, .00020]
odr21 = ODR(myData21, myModel2, guesses21, maxit=1)
odr21.set_job(fit_type=0)
output21 = odr21.run()
#output21.pprint()
Fit_out21 = output21.beta[1]*xfit + output21.beta[0]

myData22 = RealData(ranges, adj_chi1, sx = 5, sy = 0.00000005)
myModel3 = Model(function3)
guesses22 = [0., .000005, .00000000000005]
odr22 = ODR(myData22, myModel3, guesses22, maxit=1)
odr22.set_job(fit_type=0)
output22 = odr22.run()
#output22.pprint()

Fit_out10 = output22.beta[1]*(xfit) + output22.beta[0] + output22.beta[2]*(xfit**2)

#plot9.plot(xfit, Fit_out9)
#plot10.plot(xfit, Fit_out10)

#prints and labels all five parameters in the terminal, generates the plot in a new window.

plt.show()

####Library of Laguerre Polynomials for substitution in Z2
## 1, 0       1
## 1, 1       (2 - ((2*(X**2 + Y**2))/w**2))**2
## 2, 1       (3 - ((2*(X**2 + Y**2))/w**2))**2
## 5, 0       1
## 

