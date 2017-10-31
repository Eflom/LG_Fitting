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
xmin = 225
xmax = 825 
ymin = 325
ymax = 925 
x0 = (xmin + xmax)/2
y0 = (ymin + ymax)/2
ranges = [15, 50, 100, 200, 400, 600, 800, 1000]

#debug use only, used for making simulated LG data 'noisy'
xerr = scipy.random.random(200)
yerr = scipy.random.random(200)
zerr = scipy.random.random(200)*10.0 -5.0

#read in data from file, cropping it using the values above

data1_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_1_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data1_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_1_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data1_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_1_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data1_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_1_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data1_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_1_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data1_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_1_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data1_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_1_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data1_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_1_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data1_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_1_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data1_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_1_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))


data2_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_2_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data2_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_2_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data2_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_2_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data2_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_2_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data2_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_2_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data2_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_2_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data2_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_2_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data2_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_2_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data2_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_2_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data2_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_2_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))


data3_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_3_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data3_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_3_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data3_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_3_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data3_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_3_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data3_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_3_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data3_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_3_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data3_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_3_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data3_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_3_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data3_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_3_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data3_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_3_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))


data4_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_4_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data4_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_4_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data4_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_4_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data4_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_4_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data4_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_4_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data4_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_4_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data4_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_4_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data4_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_4_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data4_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_4_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data4_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_4_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))


data5_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_5_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data5_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_5_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data5_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_5_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data5_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_5_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data5_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_5_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data5_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_5_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data5_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_5_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data5_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_5_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data5_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_5_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data5_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_5_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))


data6_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_6_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data6_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_6_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data6_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_6_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data6_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_6_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data6_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_6_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data6_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_6_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data6_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_6_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data6_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_6_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data6_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_6_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data6_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_6_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))


data7_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_7_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data7_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_7_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data7_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_7_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data7_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_7_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data7_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_7_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data7_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_7_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data7_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_7_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data7_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_7_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data7_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_7_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data7_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_7_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))


data8_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_8_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data8_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_8_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data8_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_8_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data8_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_8_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data8_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_8_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data8_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_8_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data8_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_8_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data8_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_8_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data8_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_8_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data8_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/July_5/Post_8_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))


data1 = (data1_1 + data1_2 + data1_3 + data1_4 + data1_5 + data1_6 + data1_7 + data1_8 + data1_9 + data1_10)/10.
data2 = (data2_1 + data2_2 + data2_3 + data2_4 + data2_5 + data2_6 + data2_7 + data2_8 + data2_9 + data2_10)/10.
data3 = (data3_1 + data3_2 + data3_3 + data3_4 + data3_5 + data3_6 + data3_7 + data3_8 + data3_9 + data3_10)/10.
data4 = (data4_1 + data4_2 + data4_3 + data4_4 + data4_5 + data4_6 + data4_7 + data4_8 + data4_9 + data4_10)/10.
data5 = (data5_1 + data5_2 + data5_3 + data5_4 + data5_5 + data5_6 + data5_7 + data5_8 + data5_9 + data5_10)/10.
data6 = (data6_1 + data6_2 + data6_3 + data6_4 + data6_5 + data6_6 + data6_7 + data6_8 + data6_9 + data6_10)/10.
data7 = (data7_1 + data7_2 + data7_3 + data7_4 + data7_5 + data7_6 + data7_7 + data7_8 + data7_9 + data7_10)/10.
data8 = (data8_1 + data8_2 + data8_3 + data8_4 + data8_5 + data8_6 + data8_7 + data8_8 + data8_9 + data8_10)/10.


data1_sd = np.std([data1_1, data1_2, data1_3, data1_4, data1_5, data1_6, data1_7, data1_8, data1_9, data1_10], axis = 0)
data2_sd = np.std([data2_1, data2_2, data2_3, data2_4, data2_5, data2_6, data2_7, data2_8, data2_9, data2_10], axis = 0)
data3_sd = np.std([data3_1, data3_2, data3_3, data3_4, data3_5, data3_6, data3_7, data3_8, data3_9, data3_10], axis = 0)
data4_sd = np.std([data4_1, data4_2, data4_3, data4_4, data4_5, data4_6, data4_7, data4_8, data4_9, data4_10], axis = 0)
data5_sd = np.std([data5_1, data5_2, data5_3, data5_4, data5_5, data5_6, data5_7, data5_8, data5_9, data5_10], axis = 0)
data6_sd = np.std([data6_1, data6_2, data6_3, data6_4, data6_5, data6_6, data6_7, data6_8, data6_9, data6_10], axis = 0)
data7_sd = np.std([data7_1, data7_2, data7_3, data7_4, data7_5, data7_6, data7_7, data7_8, data7_9, data7_10], axis = 0)
data8_sd = np.std([data8_1, data8_2, data8_3, data8_4, data8_5, data8_6, data8_7, data8_8, data8_9, data8_10], axis = 0)


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
Crop_range = 0.3

zmin1 = np.max(np.max(data1))*Crop_range
zmin2 = np.max(np.max(data2))*Crop_range
zmin3 = np.max(np.max(data3))*Crop_range
zmin4 = np.max(np.max(data4))*Crop_range
zmin5 = np.max(np.max(data5))*Crop_range
zmin6 = np.max(np.max(data6))*Crop_range
zmin7 = np.max(np.max(data7))*Crop_range
zmin8 = np.max(np.max(data8))*Crop_range

maskeddata1 = np.where(data1 > zmin1, data1, 0.0)
maskedfitdata1 = np.where(data1 > zmin1, data1, 0.0)

maskeddata2 = np.where(data2 > zmin2, data2, 0.0)
maskedfitdata2 = np.where(data2 > zmin2, data2, 0.0)

maskeddata3 = np.where(data3 > zmin3, data3, 0.0)
maskedfitdata3 = np.where(data3 > zmin3, data3, 0.0)

maskeddata4 = np.where(data4 > zmin4, data4, 0.0)
maskedfitdata4 = np.where(data4 > zmin4, data4, 0.0)

maskeddata4 = np.where(data4 > zmin4, data4, 0.0)
maskedfitdata4 = np.where(data4 > zmin4, data4, 0.0)

maskeddata5 = np.where(data5 > zmin5, data5, 0.0)
maskedfitdata5 = np.where(data5 > zmin5, data5, 0.0)

maskeddata6 = np.where(data6 > zmin6, data6, 0.0)
maskedfitdata6 = np.where(data6 > zmin6, data6, 0.0)

maskeddata7 = np.where(data7 > zmin7, data7, 0.0)
maskedfitdata7 = np.where(data7 > zmin7, data7, 0.0)

maskeddata8 = np.where(data8 > zmin8, data8, 0.0)
maskedfitdata8 = np.where(data8 > zmin8, data8, 0.0)

N1 = np.count_nonzero(maskeddata1)
N2 = np.count_nonzero(maskeddata2)
N3 = np.count_nonzero(maskeddata3)
N4 = np.count_nonzero(maskeddata4)
N5 = np.count_nonzero(maskeddata5)
N6 = np.count_nonzero(maskeddata6)
N7 = np.count_nonzero(maskeddata7)
N8 = np.count_nonzero(maskeddata8)

#define the funciton in terms of the 5 paramters, so that the ODR can process them
def function(params, maskedfitdata1):
	scale = params[0] #= 9000
	baseline = params[2] #= 850
	width = params[1] #= 30
	y_0 = params[3] #=0
	x_0 = params[4] #=20
		
	return ((((scale)*((2.)*((X - x_0)**2. + (Y - y_0)**2.)/(width)**2.)))**l)*(np.exp((-2.)*(((X - x_0)**2. + (Y - y_0)**2.)/(width)**2.))) + baseline


#The meat of the ODR program. set "guesses" to a rough initial guess for the data <-- IMPORTANT

myData1 = Data([Q, W], data1)
myModel = Model(function)
guesses1 = [25000, 20, 150, 00, 00] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr1 = ODR(myData1, myModel, guesses1, maxit=1000)
odr1.set_job(fit_type=2)
output1 = odr1.run()
#output1.pprint()
Fit_out1 = (((((output1.beta[0]))*(2.*((X - output1.beta[4])**2. + (Y - output1.beta[3])**2.)/(output1.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output1.beta[4])**2. + (Y - output1.beta[3])**2.)/(output1.beta[1])**2.))) + output1.beta[2]
print 'done1'

myData2 = Data([Q, W], data2)
myModel = Model(function)
guesses2 = [25000, 20, 400, 07, -77] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr2 = ODR(myData2, myModel, guesses1, maxit=100)
odr2.set_job(fit_type=2)
output2 = odr2.run()
#output2.pprint()
Fit_out2 = (((((output2.beta[0]))*(2.*((X - output2.beta[4])**2. + (Y - output2.beta[3])**2.)/(output2.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output2.beta[4])**2. + (Y - output2.beta[3])**2.)/(output2.beta[1])**2.))) + output2.beta[2]
print 'done2'

myData3 = Data([Q, W], data3)
myModel = Model(function)
guesses3 = [25000, 20, 250, 0, 0] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr3 = ODR(myData3, myModel, guesses3, maxit=100)
odr3.set_job(fit_type=2)
output3 = odr3.run()
#output3.pprint()
Fit_out3 = (((((output3.beta[0]))*(2.*((X - output3.beta[4])**2. + (Y - output3.beta[3])**2.)/(output3.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output3.beta[4])**2. + (Y - output3.beta[3])**2.)/(output3.beta[1])**2.))) + output3.beta[2]
print 'done3'

myData4 = Data([Q, W], data4)
myModel = Model(function)
guesses4 = [25000, 40, 850, 0, 0] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr4 = ODR(myData4, myModel, guesses4, maxit=100)
odr4.set_job(fit_type=2)
output4 = odr4.run()
#output4.pprint()
Fit_out4 = (((((output4.beta[0]))*(2.*((X - output4.beta[4])**2. + (Y - output4.beta[3])**2.)/(output4.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output4.beta[4])**2. + (Y - output4.beta[3])**2.)/(output4.beta[1])**2.))) + output4.beta[2]
print 'done4'

myData5 = Data([Q, W], data5)
myModel = Model(function)
guesses5 = [25000, 40, 850, 0, 0] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr5 = ODR(myData5, myModel, guesses5, maxit=100)
odr5.set_job(fit_type=2)
output5 = odr5.run()
#output5.pprint()
Fit_out5 = (((((output5.beta[0]))*(2.*((X - output5.beta[4])**2. + (Y - output5.beta[3])**2.)/(output5.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output5.beta[4])**2. + (Y - output5.beta[3])**2.)/(output5.beta[1])**2.))) + output5.beta[2]
print 'done5'

myData6 = Data([Q, W], data6)
myModel = Model(function)
guesses6 = [25000, 60, 850, 0, 0] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr6 = ODR(myData6, myModel, guesses6, maxit=100)
odr6.set_job(fit_type=2)
output6 = odr6.run()
#output6.pprint()
Fit_out6 = (((((output6.beta[0]))*(2.*((X - output6.beta[4])**2. + (Y - output6.beta[3])**2.)/(output6.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output6.beta[4])**2. + (Y - output6.beta[3])**2.)/(output6.beta[1])**2.))) + output6.beta[2]
print 'done6'

myData7 = Data([Q, W], data7)
myModel = Model(function)
guesses7 = [25000, 150, 850, 0, 0] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr7 = ODR(myData7, myModel, guesses7, maxit=100)
odr7.set_job(fit_type=2)
output7 = odr7.run()
#output7.pprint()
Fit_out7 = (((((output7.beta[0]))*(2.*((X - output7.beta[4])**2. + (Y - output7.beta[3])**2.)/(output7.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output7.beta[4])**2. + (Y - output7.beta[3])**2.)/(output7.beta[1])**2.))) + output7.beta[2]
print 'done7'

myData8 = Data([Q, W], data8)
myModel = Model(function)
guesses8 = [25000, 150, 850, 0, 0] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr8 = ODR(myData8, myModel, guesses8, maxit=100)
odr8.set_job(fit_type=2)
output8 = odr8.run()
#output8.pprint()
Fit_out8 = (((((output8.beta[0]))*(2.*((X - output8.beta[4])**2. + (Y - output8.beta[3])**2.)/(output8.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output8.beta[4])**2. + (Y - output8.beta[3])**2.)/(output8.beta[1])**2.))) + output8.beta[2]
print 'done8'

maskedfit1 = np.where(Fit_out1 > zmin1, Fit_out1, 0)
maskedfit2 = np.where(Fit_out2 > zmin2, Fit_out2, 0)
maskedfit3 = np.where(Fit_out3 > zmin3, Fit_out3, 0)
maskedfit4 = np.where(Fit_out4 > zmin4, Fit_out4, 0)
maskedfit5 = np.where(Fit_out5 > zmin5, Fit_out5, 0)
maskedfit6 = np.where(Fit_out6 > zmin6, Fit_out6, 0)
maskedfit7 = np.where(Fit_out7 > zmin7, Fit_out7, 0)
maskedfit8 = np.where(Fit_out8 > zmin8, Fit_out8, 0)

Chisq1 = np.sum(np.sum((((maskeddata1 - maskedfit1))**2)/(maskedfit1+.01)))
Chisq2 = np.sum(np.sum((((maskeddata2 - maskedfit2))**2)/(maskedfit2+.01)))
Chisq3 = np.sum(np.sum((((maskeddata3 - maskedfit3))**2)/(maskedfit3+.01)))
Chisq4 = np.sum(np.sum((((maskeddata4 - maskedfit4))**2)/(maskedfit4+.01)))
Chisq5 = np.sum(np.sum((((maskeddata5 - maskedfit5))**2)/(maskedfit5+.01)))
Chisq6 = np.sum(np.sum((((maskeddata6 - maskedfit6))**2)/(maskedfit6+.01)))
Chisq7 = np.sum(np.sum((((maskeddata7 - maskedfit7))**2)/(maskedfit7+.01)))
Chisq8 = np.sum(np.sum((((maskeddata8 - maskedfit8))**2)/(maskedfit8+.01)))

scale1 = output1.beta[0]
scale2 = output2.beta[0]
scale3 = output3.beta[0]
scale4 = output4.beta[0]
scale5 = output5.beta[0]
scale6 = output6.beta[0]
scale7 = output7.beta[0]
scale8 = output8.beta[0]

Chi_values = [Chisq1, Chisq2, Chisq3, Chisq4, Chisq5, Chisq6, Chisq7, Chisq8]
N_values = [N1, N2, N3, N4, N5, N6, N7, N8]
adj_chi1 = [Chisq1/N1, Chisq2/N2, Chisq3/N3, Chisq4/N4, Chisq5/N5, Chisq6/N6, Chisq7/N7, Chisq8/N8]
adj_chi2 = np.array([Chisq1/N1/scale1, Chisq2/N2/scale2, Chisq3/N3/scale3, Chisq4/N4/scale4, Chisq5/N5/scale5, Chisq6/N6/scale6, Chisq7/N7/scale7, Chisq8/N8/scale8])
adj_chi3 = [Chisq1/scale1, Chisq2/scale2, Chisq3/scale3, Chisq4/scale4, Chisq5/scale5, Chisq6/scale6, Chisq7/scale7, Chisq8/scale8]


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



plot9=fig9.add_subplot(111)
plot10=fig10.add_subplot(111)

plot1_1.set_title('Plane Fit 15mm')
plot1_2.set_title('Data 15 mm')

plot2_1.set_title('Plane Fit 50mm')
plot2_2.set_title('Data 50mm')

plot3_1.set_title('Plane Fit 100mm')
plot3_2.set_title('Data 100mm')

plot4_1.set_title('Plane Fit 200mm')
plot4_2.set_title('Data 200mm')

plot5_1.set_title('Plane Fit 400mm')
plot5_2.set_title('Data 400mm')

plot6_1.set_title('Plane Fit 600mm')
plot6_2.set_title('Data 600mm')

plot7_1.set_title('Plane Fit 800mm')
plot7_2.set_title('Data 800mm')

plot8_1.set_title('Plane Fit 1000mm')
plot8_2.set_title('Data 1000mm')

plot9.set_title('Chi Sq./N vs. Distance, adjusted')
plot10.set_title('Chi Sq. vs. Distance, adjusted')

plot1_1.plot_surface(Y, X, maskedfit1, rstride = 2, cstride = 2, linewidth = 0.05, cmap = 'cool')
plot1_2.plot_surface(Y, X, maskeddata1, rstride = 2, cstride = 2, cmap = 'hot', linewidth = 0.05) 

plot2_1.plot_surface(Y, X, maskedfit2, rstride = 10, cstride = 10, linewidth = 0.05, cmap = 'cool')
plot2_2.plot_surface(Y, X, maskeddata2, rstride = 1, cstride = 1, cmap = 'hot', linewidth = 0.05) 

plot3_1.plot_surface(Y, X, maskedfit3, rstride = 20, cstride = 20, linewidth = 0.05, cmap = 'cool')
plot3_2.plot_surface(Y, X, maskeddata3, rstride = 20, cstride = 20, cmap = 'hot', linewidth = 0.05) 

plot4_1.plot_surface(Y, X, maskedfit4, rstride = 20, cstride = 20, linewidth = 0.05, cmap = 'cool')
plot4_2.plot_surface(Y, X, maskeddata4, rstride = 20, cstride = 20, cmap = 'hot', linewidth = 0.05) 

plot5_1.plot_surface(Y, X, maskedfit5, rstride = 20, cstride = 20, linewidth = 0.05, cmap = 'cool')
plot5_2.plot_surface(Y, X, maskeddata5, rstride = 20, cstride = 20, cmap = 'hot', linewidth = 0.05) 

plot6_1.plot_surface(Y, X, maskedfit6, rstride = 20, cstride = 20, linewidth = 0.05, cmap = 'cool')
plot6_2.plot_surface(Y, X, maskeddata6, rstride = 20, cstride = 20, cmap = 'hot', linewidth = 0.05) 

plot7_1.plot_surface(Y, X, maskedfit7, rstride = 20, cstride = 20, linewidth = 0.05, cmap = 'cool')
plot7_2.plot_surface(Y, X, maskeddata7, rstride = 20, cstride = 20, cmap = 'hot', linewidth = 0.05) 

plot8_1.plot_surface(Y, X, maskedfit8, rstride = 20, cstride = 20, linewidth = 0.05, cmap = 'cool')
plot8_2.plot_surface(Y, X, maskeddata8, rstride = 20, cstride = 20, cmap = 'hot', linewidth = 0.05)

plot9.scatter(ranges, adj_chi1)
plot10.scatter(ranges, adj_chi1)

plot2_2.set_xlim(-50, 50)
plot2_2.set_ylim(-50, 50)
plot2_1.set_zlim(10, 1400)
plot2_2.set_zlim(10, 1400)
plot3_1.set_zlim(10, 1400)
plot3_2.set_zlim(10, 1400)
plot4_1.set_zlim(10, 1400)
plot4_2.set_zlim(10, 1400)
plot5_1.set_zlim(10, 1400)
plot5_2.set_zlim(10, 1400)
plot6_1.set_zlim(10, 1400)
plot6_2.set_zlim(10, 1400)
plot7_1.set_zlim(10, 1400)
plot7_2.set_zlim(10, 1400)
plot8_1.set_zlim(10, 1400)
plot8_2.set_zlim(10, 1400)

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
	
myData9 = RealData(ranges, adj_chi1, sx = 5, sy = 1)
myModel2 = Model(function2)
guesses9 = [0.001, .00020]
odr9 = ODR(myData9, myModel2, guesses9, maxit=1)
odr9.set_job(fit_type=0)
output9 = odr9.run()
#output9.pprint()
Fit_out9 = output9.beta[1]*xfit + output9.beta[0]

myData10 = RealData(ranges, adj_chi1, sx = 5, sy = 0.00000005)
myModel3 = Model(function3)
guesses10 = [0., .000005, .00000000000005]
odr10 = ODR(myData10, myModel3, guesses10, maxit=1)
odr10.set_job(fit_type=0)
output10 = odr10.run()
#output10.pprint()

Fit_out10 = output10.beta[1]*(xfit) + output10.beta[0] + output10.beta[2]*(xfit**2)
#plot9.plot(xfit, Fit_out9)
#plot10.plot(xfit, Fit_out10)

#prints and labels all five parameters in the terminal, generates the plot in a new window.

print N1
print N2
print N3
print N4
print N5
print N6
print N7
print N8

print Chisq1
print Chisq2
print Chisq3
print Chisq4
print Chisq5
print Chisq6
print Chisq7
print Chisq8

plt.show()

####Library of Laguerre Polynomials for substitution in Z2
## 1, 0       1
## 1, 1       (2 - ((2*(X**2 + Y**2))/w**2))**2
## 2, 1       (3 - ((2*(X**2 + Y**2))/w**2))**2
## 5, 0       1
## 

