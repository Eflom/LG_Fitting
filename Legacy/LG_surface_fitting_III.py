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
ranges_inch = np.arange(0,50,1)
ranges = (ranges_inch)*25.4 + 5

#debug use only, used for making simulated LG data 'noisy'
xerr = scipy.random.random(200)
yerr = scipy.random.random(200)
zerr = scipy.random.random(200)*10.0 -5.0

#read in data from file, cropping it using the values above

data1_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/1_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data1_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/1_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data1_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/1_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data1_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/1_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data1_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/1_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data1_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/1_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data1_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/1_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data1_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/1_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data1_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/1_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data1_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/1_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))


data2_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/2_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data2_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/2_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data2_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/2_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data2_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/2_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data2_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/2_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data2_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/2_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data2_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/2_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data2_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/2_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data2_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/2_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data2_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/2_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))


data3_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/3_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data3_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/3_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data3_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/3_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data3_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/3_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data3_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/3_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data3_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/3_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data3_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/3_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data3_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/3_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data3_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/3_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data3_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/3_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))


data4_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/4_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data4_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/4_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data4_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/4_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data4_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/4_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data4_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/4_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data4_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/4_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data4_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/4_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data4_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/4_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data4_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/4_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data4_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/4_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))


data5_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/5_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data5_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/5_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data5_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/5_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data5_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/5_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data5_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/5_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data5_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/5_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data5_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/5_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data5_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/5_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data5_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/5_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data5_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/5_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))


data6_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/6_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data6_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/6_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data6_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/6_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data6_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/6_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data6_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/6_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data6_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/6_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data6_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/6_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data6_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/6_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data6_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/6_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data6_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/6_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))


data7_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/7_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data7_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/7_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data7_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/7_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data7_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/7_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data7_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/7_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data7_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/7_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data7_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/7_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data7_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/7_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data7_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/7_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data7_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/7_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))


data8_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/8_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data8_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/8_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data8_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/8_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data8_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/8_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data8_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/8_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data8_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/8_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data8_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/8_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data8_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/8_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data8_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/8_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data8_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/8_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data9_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/9_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data9_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/9_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data9_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/9_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data9_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/9_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data9_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/9_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data9_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/9_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data9_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/9_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data9_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/9_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data9_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/9_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data9_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/9_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data10_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/10_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data10_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/10_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data10_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/10_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data10_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/10_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data10_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/10_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data10_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/10_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data10_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/10_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data10_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/10_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data10_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/10_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data10_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/10_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data11_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/11_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data11_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/11_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data11_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/11_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data11_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/11_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data11_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/11_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data11_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/11_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data11_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/11_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data11_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/11_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data11_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/11_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data11_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/11_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data12_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/12_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data12_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/12_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data12_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/12_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data12_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/12_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data12_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/12_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data12_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/12_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data12_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/12_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data12_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/12_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data12_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/12_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data12_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/12_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data13_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/13_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data13_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/13_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data13_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/13_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data13_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/13_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data13_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/13_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data13_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/13_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data13_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/13_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data13_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/13_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data13_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/13_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data13_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/13_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data14_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/14_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data14_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/14_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data14_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/14_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data14_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/14_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data14_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/14_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data14_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/14_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data14_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/14_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data14_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/14_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data14_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/14_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data14_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/14_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data15_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/15_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data15_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/15_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data15_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/15_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data15_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/15_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data15_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/15_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data15_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/15_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data15_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/15_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data15_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/15_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data15_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/15_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data15_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/15_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data16_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/16_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data16_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/16_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data16_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/16_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data16_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/16_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data16_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/16_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data16_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/16_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data16_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/16_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data16_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/16_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data16_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/16_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data16_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/16_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data17_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/17_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data17_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/17_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data17_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/17_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data17_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/17_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data17_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/17_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data17_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/17_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data17_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/17_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data17_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/17_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data17_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/17_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data17_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/17_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data18_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/18_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data18_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/18_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data18_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/18_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data18_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/18_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data18_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/18_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data18_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/18_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data18_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/18_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data18_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/18_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data18_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/18_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data18_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/18_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data19_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/19_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data19_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/19_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data19_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/19_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data19_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/19_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data19_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/19_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data19_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/19_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data19_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/19_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data19_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/19_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data19_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/19_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data19_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/19_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))


data20_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/20_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data20_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/20_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data20_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/20_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data20_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/20_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data20_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/20_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data20_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/20_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data20_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/20_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data20_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/20_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data20_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/20_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data20_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/20_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data21_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/21_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data21_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/21_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data21_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/21_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data21_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/21_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data21_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/21_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data21_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/21_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data21_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/21_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data21_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/21_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data21_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/21_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data21_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/21_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data22_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/22_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data22_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/22_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data22_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/22_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data22_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/22_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data22_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/22_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data22_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/22_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data22_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/22_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data22_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/22_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data22_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/22_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data22_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/22_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data23_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/23_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data23_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/23_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data23_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/23_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data23_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/23_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data23_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/23_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data23_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/23_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data23_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/23_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data23_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/23_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data23_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/23_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data23_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/23_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data24_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/24_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data24_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/24_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data24_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/24_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data24_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/24_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data24_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/24_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data24_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/24_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data24_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/24_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data24_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/24_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data24_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/24_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data24_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/24_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data25_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/25_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data25_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/25_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data25_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/25_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data25_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/25_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data25_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/25_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data25_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/25_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data25_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/25_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data25_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/25_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data25_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/25_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data25_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/25_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data26_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/26_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data26_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/26_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data26_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/26_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data26_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/26_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data26_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/26_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data26_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/26_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data26_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/26_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data26_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/26_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data26_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/26_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data26_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/26_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data27_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/27_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data27_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/27_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data27_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/27_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data27_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/27_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data27_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/27_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data27_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/27_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data27_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/27_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data27_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/27_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data27_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/27_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data27_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/27_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data28_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/28_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data28_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/28_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data28_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/28_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data28_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/28_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data28_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/28_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data28_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/28_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data28_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/28_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data28_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/28_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data28_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/28_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data28_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/28_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data29_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/29_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data29_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/29_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data29_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/29_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data29_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/29_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data29_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/29_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data29_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/29_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data29_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/29_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data29_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/29_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data29_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/29_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data29_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/29_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data30_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/30_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data30_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/30_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data30_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/30_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data30_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/30_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data30_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/30_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data30_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/30_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data30_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/30_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data30_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/30_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data30_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/30_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data30_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/30_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data31_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/31_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data31_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/31_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data31_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/31_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data31_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/31_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data31_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/31_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data31_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/31_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data31_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/31_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data31_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/31_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data31_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/31_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data31_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/31_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data32_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/32_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data32_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/32_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data32_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/32_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data32_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/32_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data32_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/32_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data32_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/32_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data32_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/32_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data32_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/32_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data32_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/32_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data32_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/32_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data33_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/33_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data33_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/33_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data33_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/33_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data33_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/33_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data33_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/33_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data33_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/33_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data33_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/33_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data33_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/33_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data33_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/33_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data33_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/33_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data34_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/34_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data34_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/34_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data34_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/34_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data34_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/34_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data34_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/34_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data34_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/34_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data34_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/34_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data34_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/34_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data34_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/34_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data34_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/34_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data35_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/35_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data35_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/35_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data35_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/35_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data35_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/35_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data35_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/35_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data35_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/35_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data35_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/35_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data35_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/35_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data35_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/35_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data35_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/35_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data36_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/36_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data36_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/36_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data36_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/36_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data36_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/36_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data36_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/36_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data36_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/36_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data36_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/36_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data36_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/36_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data36_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/36_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data36_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/36_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data37_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/37_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data37_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/37_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data37_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/37_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data37_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/37_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data37_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/37_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data37_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/37_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data37_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/37_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data37_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/37_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data37_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/37_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data37_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/37_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data38_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/38_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data38_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/38_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data38_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/38_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data38_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/38_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data38_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/38_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data38_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/38_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data38_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/38_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data38_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/38_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data38_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/38_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data38_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/38_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data39_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/39_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data39_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/39_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data39_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/39_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data39_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/39_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data39_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/39_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data39_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/39_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data39_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/39_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data39_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/39_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data39_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/39_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data39_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/39_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data40_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/40_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data40_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/40_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data40_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/40_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data40_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/40_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data40_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/40_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data40_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/40_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data40_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/40_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data40_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/40_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data40_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/40_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data40_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/40_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data41_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/41_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data41_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/41_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data41_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/41_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data41_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/41_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data41_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/41_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data41_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/41_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data41_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/41_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data41_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/41_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data41_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/41_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data41_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/41_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data42_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/42_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data42_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/42_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data42_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/42_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data42_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/42_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data42_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/42_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data42_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/42_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data42_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/42_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data42_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/42_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data42_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/42_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data42_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/42_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data43_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/43_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data43_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/43_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data43_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/43_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data43_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/43_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data43_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/43_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data43_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/43_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data43_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/43_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data43_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/43_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data43_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/43_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data43_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/43_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data44_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/44_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data44_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/44_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data44_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/44_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data44_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/44_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data44_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/44_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data44_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/44_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data44_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/44_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data44_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/44_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data44_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/44_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data44_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/44_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data45_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/45_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data45_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/45_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data45_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/45_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data45_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/45_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data45_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/45_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data45_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/45_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data45_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/45_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data45_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/45_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data45_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/45_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data45_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/45_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data46_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/46_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data46_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/46_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data46_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/46_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data46_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/46_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data46_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/46_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data46_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/46_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data46_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/46_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data46_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/46_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data46_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/46_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data46_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/46_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data47_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/47_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data47_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/47_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data47_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/47_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data47_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/47_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data47_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/47_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data47_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/47_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data47_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/47_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data47_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/47_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data47_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/47_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data47_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/47_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data48_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/48_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data48_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/48_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data48_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/48_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data48_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/48_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data48_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/48_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data48_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/48_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data48_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/48_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data48_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/48_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data48_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/48_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data48_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/48_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data49_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/49_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data49_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/49_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data49_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/49_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data49_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/49_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data49_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/49_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data49_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/49_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data49_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/49_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data49_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/49_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data49_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/49_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data49_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/49_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data50_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/50_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data50_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/50_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data50_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/50_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data50_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/50_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data50_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/50_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data50_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/50_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data50_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/50_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data50_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/50_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data50_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/50_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data50_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/50_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

print 'data read in complete'


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
data21 = (data21_1 + data21_2 + data21_3 + data21_4 + data21_5 + data21_6 + data21_7 + data21_8 + data21_9 + data21_10)/10.
data22 = (data22_1 + data22_2 + data22_3 + data22_4 + data22_5 + data22_6 + data22_7 + data22_8 + data22_9 + data22_10)/10.
data23 = (data23_1 + data23_2 + data23_3 + data23_4 + data23_5 + data23_6 + data23_7 + data23_8 + data23_9 + data23_10)/10.
data24 = (data24_1 + data24_2 + data24_3 + data24_4 + data24_5 + data24_6 + data24_7 + data24_8 + data24_9 + data24_10)/10.
data25 = (data25_1 + data25_2 + data25_3 + data25_4 + data25_5 + data25_6 + data25_7 + data25_8 + data25_9 + data25_10)/10.
data26 = (data26_1 + data26_2 + data26_3 + data26_4 + data26_5 + data26_6 + data26_7 + data26_8 + data26_9 + data26_10)/10.
data27 = (data27_1 + data27_2 + data27_3 + data27_4 + data27_5 + data27_6 + data27_7 + data27_8 + data27_9 + data27_10)/10.
data28 = (data28_1 + data28_2 + data28_3 + data28_4 + data28_5 + data28_6 + data28_7 + data28_8 + data28_9 + data28_10)/10.
data29 = (data29_1 + data29_2 + data29_3 + data29_4 + data29_5 + data29_6 + data29_7 + data29_8 + data29_9 + data29_10)/10.
data30 = (data30_1 + data30_2 + data30_3 + data30_4 + data30_5 + data30_6 + data30_7 + data30_8 + data30_9 + data30_10)/10.
data31 = (data31_1 + data31_2 + data31_3 + data31_4 + data31_5 + data31_6 + data31_7 + data31_8 + data31_9 + data31_10)/10.
data32 = (data32_1 + data32_2 + data32_3 + data32_4 + data32_5 + data32_6 + data32_7 + data32_8 + data32_9 + data32_10)/10.
data33 = (data33_1 + data33_2 + data33_3 + data33_4 + data33_5 + data33_6 + data33_7 + data33_8 + data33_9 + data33_10)/10.
data34 = (data34_1 + data34_2 + data34_3 + data34_4 + data34_5 + data34_6 + data34_7 + data34_8 + data34_9 + data34_10)/10.
data35 = (data35_1 + data35_2 + data35_3 + data35_4 + data35_5 + data35_6 + data35_7 + data35_8 + data35_9 + data35_10)/10.
data36 = (data36_1 + data36_2 + data36_3 + data36_4 + data36_5 + data36_6 + data36_7 + data36_8 + data36_9 + data36_10)/10.
data37 = (data37_1 + data37_2 + data37_3 + data37_4 + data37_5 + data37_6 + data37_7 + data37_8 + data37_9 + data37_10)/10.
data38 = (data38_1 + data38_2 + data38_3 + data38_4 + data38_5 + data38_6 + data38_7 + data38_8 + data38_9 + data38_10)/10.
data39 = (data39_1 + data39_2 + data39_3 + data39_4 + data39_5 + data39_6 + data39_7 + data39_8 + data39_9 + data39_10)/10.
data40 = (data40_1 + data40_2 + data40_3 + data40_4 + data40_5 + data40_6 + data40_7 + data40_8 + data40_9 + data40_10)/10.
data41 = (data41_1 + data41_2 + data41_3 + data41_4 + data41_5 + data41_6 + data41_7 + data41_8 + data41_9 + data41_10)/10.
data42 = (data42_1 + data42_2 + data42_3 + data42_4 + data42_5 + data42_6 + data42_7 + data42_8 + data42_9 + data42_10)/10.
data43 = (data43_1 + data43_2 + data43_3 + data43_4 + data43_5 + data43_6 + data43_7 + data43_8 + data43_9 + data43_10)/10.
data44 = (data44_1 + data44_2 + data44_3 + data44_4 + data44_5 + data44_6 + data44_7 + data44_8 + data44_9 + data44_10)/10.
data45 = (data45_1 + data45_2 + data45_3 + data45_4 + data45_5 + data45_6 + data45_7 + data45_8 + data45_9 + data45_10)/10.
data46 = (data46_1 + data46_2 + data46_3 + data46_4 + data46_5 + data46_6 + data46_7 + data46_8 + data46_9 + data46_10)/10.
data47 = (data47_1 + data47_2 + data47_3 + data47_4 + data47_5 + data47_6 + data47_7 + data47_8 + data47_9 + data47_10)/10.
data48 = (data48_1 + data48_2 + data48_3 + data48_4 + data48_5 + data48_6 + data48_7 + data48_8 + data48_9 + data48_10)/10.
data49 = (data49_1 + data49_2 + data49_3 + data49_4 + data49_5 + data49_6 + data49_7 + data49_8 + data49_9 + data49_10)/10.
data50 = (data50_1 + data50_2 + data50_3 + data50_4 + data50_5 + data50_6 + data50_7 + data50_8 + data50_9 + data50_10)/10.

print 'data averaging complete'

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
data21_sd = np.std([data21_1, data21_2, data21_3, data21_4, data21_5, data21_6, data21_7, data21_8, data21_9, data21_10], axis = 0)
data22_sd = np.std([data22_1, data22_2, data22_3, data22_4, data22_5, data22_6, data22_7, data22_8, data22_9, data22_10], axis = 0)
data23_sd = np.std([data23_1, data23_2, data23_3, data23_4, data23_5, data23_6, data23_7, data23_8, data23_9, data23_10], axis = 0)
data24_sd = np.std([data24_1, data24_2, data24_3, data24_4, data24_5, data24_6, data24_7, data24_8, data24_9, data24_10], axis = 0)
data25_sd = np.std([data25_1, data25_2, data25_3, data25_4, data25_5, data25_6, data25_7, data25_8, data25_9, data25_10], axis = 0)
data26_sd = np.std([data26_1, data26_2, data26_3, data26_4, data26_5, data26_6, data26_7, data26_8, data26_9, data26_10], axis = 0)
data27_sd = np.std([data27_1, data27_2, data27_3, data27_4, data27_5, data27_6, data27_7, data27_8, data27_9, data27_10], axis = 0)
data28_sd = np.std([data28_1, data28_2, data28_3, data28_4, data28_5, data28_6, data28_7, data28_8, data28_9, data28_10], axis = 0)
data29_sd = np.std([data29_1, data29_2, data29_3, data29_4, data29_5, data29_6, data29_7, data29_8, data29_9, data29_10], axis = 0)
data30_sd = np.std([data30_1, data30_2, data30_3, data30_4, data30_5, data30_6, data30_7, data30_8, data30_9, data30_10], axis = 0)
data31_sd = np.std([data31_1, data31_2, data31_3, data31_4, data31_5, data31_6, data31_7, data31_8, data31_9, data31_10], axis = 0)
data32_sd = np.std([data32_1, data32_2, data32_3, data32_4, data32_5, data32_6, data32_7, data32_8, data32_9, data32_10], axis = 0)
data33_sd = np.std([data33_1, data33_2, data33_3, data33_4, data33_5, data33_6, data33_7, data33_8, data33_9, data33_10], axis = 0)
data34_sd = np.std([data34_1, data34_2, data34_3, data34_4, data34_5, data34_6, data34_7, data34_8, data34_9, data34_10], axis = 0)
data35_sd = np.std([data35_1, data35_2, data35_3, data35_4, data35_5, data35_6, data35_7, data35_8, data35_9, data35_10], axis = 0)
data36_sd = np.std([data36_1, data36_2, data36_3, data36_4, data36_5, data36_6, data36_7, data36_8, data36_9, data36_10], axis = 0)
data37_sd = np.std([data37_1, data37_2, data37_3, data37_4, data37_5, data37_6, data37_7, data37_8, data37_9, data37_10], axis = 0)
data38_sd = np.std([data38_1, data38_2, data38_3, data38_4, data38_5, data38_6, data38_7, data38_8, data38_9, data38_10], axis = 0)
data39_sd = np.std([data39_1, data39_2, data39_3, data39_4, data39_5, data39_6, data39_7, data39_8, data39_9, data39_10], axis = 0)
data40_sd = np.std([data40_1, data40_2, data40_3, data40_4, data40_5, data40_6, data40_7, data40_8, data40_9, data40_10], axis = 0)
data41_sd = np.std([data41_1, data41_2, data41_3, data41_4, data41_5, data41_6, data41_7, data41_8, data41_9, data41_10], axis = 0)
data42_sd = np.std([data42_1, data42_2, data42_3, data42_4, data42_5, data42_6, data42_7, data42_8, data42_9, data42_10], axis = 0)
data43_sd = np.std([data43_1, data43_2, data43_3, data43_4, data43_5, data43_6, data43_7, data43_8, data43_9, data43_10], axis = 0)
data44_sd = np.std([data44_1, data44_2, data44_3, data44_4, data44_5, data44_6, data44_7, data44_8, data44_9, data44_10], axis = 0)
data45_sd = np.std([data45_1, data45_2, data45_3, data45_4, data45_5, data45_6, data45_7, data45_8, data45_9, data45_10], axis = 0)
data46_sd = np.std([data46_1, data46_2, data46_3, data46_4, data46_5, data46_6, data46_7, data46_8, data46_9, data46_10], axis = 0)
data47_sd = np.std([data47_1, data47_2, data47_3, data47_4, data47_5, data47_6, data47_7, data47_8, data47_9, data47_10], axis = 0)
data48_sd = np.std([data48_1, data48_2, data48_3, data48_4, data48_5, data48_6, data48_7, data48_8, data48_9, data48_10], axis = 0)
data49_sd = np.std([data49_1, data49_2, data49_3, data49_4, data49_5, data49_6, data49_7, data49_8, data49_9, data49_10], axis = 0)
data50_sd = np.std([data50_1, data50_2, data50_3, data50_4, data50_5, data50_6, data50_7, data50_8, data50_9, data50_10], axis = 0)

print 'standard deviation calculation complete'

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
Crop_range = 0.2

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
zmin11 = np.max(np.max(data11))*Crop_range
zmin12 = np.max(np.max(data12))*Crop_range
zmin13 = np.max(np.max(data13))*Crop_range
zmin14 = np.max(np.max(data14))*Crop_range
zmin15 = np.max(np.max(data15))*Crop_range
zmin16 = np.max(np.max(data16))*Crop_range
zmin17 = np.max(np.max(data17))*Crop_range
zmin18 = np.max(np.max(data18))*Crop_range
zmin19 = np.max(np.max(data19))*Crop_range
zmin20 = np.max(np.max(data20))*Crop_range
zmin21 = np.max(np.max(data21))*Crop_range
zmin22 = np.max(np.max(data22))*Crop_range
zmin23 = np.max(np.max(data23))*Crop_range
zmin24 = np.max(np.max(data24))*Crop_range
zmin25 = np.max(np.max(data25))*Crop_range
zmin26 = np.max(np.max(data26))*Crop_range
zmin27 = np.max(np.max(data27))*Crop_range
zmin28 = np.max(np.max(data28))*Crop_range
zmin29 = np.max(np.max(data29))*Crop_range
zmin30 = np.max(np.max(data30))*Crop_range
zmin31 = np.max(np.max(data31))*Crop_range
zmin32 = np.max(np.max(data32))*Crop_range
zmin33 = np.max(np.max(data33))*Crop_range
zmin34 = np.max(np.max(data34))*Crop_range
zmin35 = np.max(np.max(data35))*Crop_range
zmin36 = np.max(np.max(data36))*Crop_range
zmin37 = np.max(np.max(data37))*Crop_range
zmin38 = np.max(np.max(data38))*Crop_range
zmin39 = np.max(np.max(data39))*Crop_range
zmin40 = np.max(np.max(data40))*Crop_range
zmin41 = np.max(np.max(data41))*Crop_range
zmin42 = np.max(np.max(data42))*Crop_range
zmin43 = np.max(np.max(data43))*Crop_range
zmin44 = np.max(np.max(data44))*Crop_range
zmin45 = np.max(np.max(data45))*Crop_range
zmin46 = np.max(np.max(data46))*Crop_range
zmin47 = np.max(np.max(data47))*Crop_range
zmin48 = np.max(np.max(data48))*Crop_range
zmin49 = np.max(np.max(data49))*Crop_range
zmin50 = np.max(np.max(data50))*Crop_range


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
maskeddata11 = np.where(data11 > zmin11, data11, 100)
maskeddata12 = np.where(data12 > zmin12, data12, 100)
maskeddata13 = np.where(data13 > zmin13, data13, 100)
maskeddata14 = np.where(data14 > zmin14, data14, 100)
maskeddata15 = np.where(data15 > zmin15, data15, 100)
maskeddata16 = np.where(data16 > zmin16, data16, 100)
maskeddata17 = np.where(data17 > zmin17, data17, 100)
maskeddata18 = np.where(data18 > zmin18, data18, 100)
maskeddata19 = np.where(data19 > zmin19, data19, 100)
maskeddata20 = np.where(data20 > zmin20, data20, 100)
maskeddata21 = np.where(data21 > zmin21, data21, 100)
maskeddata22 = np.where(data22 > zmin22, data22, 100)
maskeddata23 = np.where(data23 > zmin23, data23, 100)
maskeddata24 = np.where(data24 > zmin24, data24, 100)
maskeddata25 = np.where(data25 > zmin25, data25, 100)
maskeddata26 = np.where(data26 > zmin26, data26, 100)
maskeddata27 = np.where(data27 > zmin27, data27, 100)
maskeddata28 = np.where(data28 > zmin28, data28, 100)
maskeddata29 = np.where(data29 > zmin29, data29, 100)
maskeddata30 = np.where(data30 > zmin30, data30, 100)
maskeddata31 = np.where(data31 > zmin31, data31, 100)
maskeddata32 = np.where(data32 > zmin32, data32, 100)
maskeddata33 = np.where(data33 > zmin33, data33, 100)
maskeddata34 = np.where(data34 > zmin34, data34, 100)
maskeddata35 = np.where(data35 > zmin35, data35, 100)
maskeddata36 = np.where(data36 > zmin36, data36, 100)
maskeddata37 = np.where(data37 > zmin37, data37, 100)
maskeddata38 = np.where(data38 > zmin38, data38, 100)
maskeddata39 = np.where(data39 > zmin39, data39, 100)
maskeddata40 = np.where(data40 > zmin40, data40, 100)
maskeddata41 = np.where(data41 > zmin41, data41, 100)
maskeddata42 = np.where(data42 > zmin42, data42, 100)
maskeddata43 = np.where(data43 > zmin43, data43, 100)
maskeddata44 = np.where(data44 > zmin44, data44, 100)
maskeddata45 = np.where(data45 > zmin45, data45, 100)
maskeddata46 = np.where(data46 > zmin46, data46, 100)
maskeddata47 = np.where(data47 > zmin47, data47, 100)
maskeddata48 = np.where(data48 > zmin48, data48, 100)
maskeddata49 = np.where(data49 > zmin49, data49, 100)
maskeddata50 = np.where(data50 > zmin50, data50, 100)

N1 = np.size(np.where(maskeddata1 > 100)) 
N2 = np.size(np.where(maskeddata2 > 100))
N3 = np.size(np.where(maskeddata3 > 100))
N4 = np.size(np.where(maskeddata4 > 100))
N5 = np.size(np.where(maskeddata5 > 100))
N6 = np.size(np.where(maskeddata6 > 100))
N7 = np.size(np.where(maskeddata7 > 100))
N8 = np.size(np.where(maskeddata8 > 100))
N9 = np.size(np.where(maskeddata9 > 100))
N10 = np.size(np.where(maskeddata10 > 100))
N11 = np.size(np.where(maskeddata11 > 100))
N12 = np.size(np.where(maskeddata12 > 100))
N13 = np.size(np.where(maskeddata13 > 100))
N14 = np.size(np.where(maskeddata14 > 100))
N15 = np.size(np.where(maskeddata15 > 100))
N16 = np.size(np.where(maskeddata16 > 100))
N17 = np.size(np.where(maskeddata17 > 100))
N18 = np.size(np.where(maskeddata18 > 100))
N19 = np.size(np.where(maskeddata19 > 100))
N20 = np.size(np.where(maskeddata20 > 100))
N21 = np.size(np.where(maskeddata21 > 100))
N22 = np.size(np.where(maskeddata22 > 100))
N23 = np.size(np.where(maskeddata23 > 100))
N24 = np.size(np.where(maskeddata24 > 100))
N25 = np.size(np.where(maskeddata25 > 100))
N26 = np.size(np.where(maskeddata26 > 100))
N27 = np.size(np.where(maskeddata27 > 100))
N28 = np.size(np.where(maskeddata28 > 100))
N29 = np.size(np.where(maskeddata29 > 100))
N30 = np.size(np.where(maskeddata30 > 100))
N31 = np.size(np.where(maskeddata31 > 100))
N32 = np.size(np.where(maskeddata32 > 100))
N33 = np.size(np.where(maskeddata33 > 100))
N34 = np.size(np.where(maskeddata34 > 100))
N35 = np.size(np.where(maskeddata35 > 100))
N36 = np.size(np.where(maskeddata36 > 100))
N37 = np.size(np.where(maskeddata37 > 100))
N38 = np.size(np.where(maskeddata38 > 100))
N39 = np.size(np.where(maskeddata39 > 100))
N40 = np.size(np.where(maskeddata40 > 100))
N41 = np.size(np.where(maskeddata41 > 100))
N42 = np.size(np.where(maskeddata42 > 100))
N43 = np.size(np.where(maskeddata43 > 100))
N44 = np.size(np.where(maskeddata44 > 100))
N45 = np.size(np.where(maskeddata45 > 100))
N46 = np.size(np.where(maskeddata46 > 100))
N47 = np.size(np.where(maskeddata47 > 100))
N48 = np.size(np.where(maskeddata48 > 100))
N49 = np.size(np.where(maskeddata49 > 100))
N50 = np.size(np.where(maskeddata50 > 100))


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
guesses1 = [25000, 30, 850, -60, -50] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr1 = ODR(myData1, myModel, guesses1, maxit=1000)
odr1.set_job(fit_type=2)
output1 = odr1.run()
#output1.pprint()
Fit_out1 = (((((output1.beta[0]))*(2.*((X - output1.beta[4])**2. + (Y - output1.beta[3])**2.)/(output1.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output1.beta[4])**2. + (Y - output1.beta[3])**2.)/(output1.beta[1])**2.))) + output1.beta[2]
print 'done1'

myData2 = Data([Q, W], data2)
myModel = Model(function)
guesses2 = [25000, 30, 800, -50, -50] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr2 = ODR(myData2, myModel, guesses1, maxit=1000)
odr2.set_job(fit_type=2)
output2 = odr2.run()
#output2.pprint()
Fit_out2 = (((((output2.beta[0]))*(2.*((X - output2.beta[4])**2. + (Y - output2.beta[3])**2.)/(output2.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output2.beta[4])**2. + (Y - output2.beta[3])**2.)/(output2.beta[1])**2.))) + output2.beta[2]
print 'done2'

myData3 = Data([Q, W], data3)
myModel = Model(function)
guesses3 = [25000, 30, 450, -50, -50] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr3 = ODR(myData3, myModel, guesses3, maxit=1000)
odr3.set_job(fit_type=2)
output3 = odr3.run()
#output3.pprint()
Fit_out3 = (((((output3.beta[0]))*(2.*((X - output3.beta[4])**2. + (Y - output3.beta[3])**2.)/(output3.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output3.beta[4])**2. + (Y - output3.beta[3])**2.)/(output3.beta[1])**2.))) + output3.beta[2]
print 'done3'

myData4 = Data([Q, W], data4)
myModel = Model(function)
guesses4 = [25000, 30, 450, -50, -50] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr4 = ODR(myData4, myModel, guesses4, maxit=1000)
odr4.set_job(fit_type=2)
output4 = odr4.run()
#output4.pprint()
Fit_out4 = (((((output4.beta[0]))*(2.*((X - output4.beta[4])**2. + (Y - output4.beta[3])**2.)/(output4.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output4.beta[4])**2. + (Y - output4.beta[3])**2.)/(output4.beta[1])**2.))) + output4.beta[2]
print 'done4'

myData5 = Data([Q, W], data5)
myModel = Model(function)
guesses5 = [25000, 30, 450, -50, -50] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr5 = ODR(myData5, myModel, guesses5, maxit=1000)
odr5.set_job(fit_type=2)
output5 = odr5.run()
#output5.pprint()
Fit_out5 = (((((output5.beta[0]))*(2.*((X - output5.beta[4])**2. + (Y - output5.beta[3])**2.)/(output5.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output5.beta[4])**2. + (Y - output5.beta[3])**2.)/(output5.beta[1])**2.))) + output5.beta[2]
print 'done5'

myData6 = Data([Q, W], data6)
myModel = Model(function)
guesses6 = [25000, 30, 850, -50, -50] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr6 = ODR(myData6, myModel, guesses6, maxit=1000)
odr6.set_job(fit_type=2)
output6 = odr6.run()
#output6.pprint()
Fit_out6 = (((((output6.beta[0]))*(2.*((X - output6.beta[4])**2. + (Y - output6.beta[3])**2.)/(output6.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output6.beta[4])**2. + (Y - output6.beta[3])**2.)/(output6.beta[1])**2.))) + output6.beta[2]
print 'done6'

myData7 = Data([Q, W], data7)
myModel = Model(function)
guesses7 = [25000, 30, 450, -50, -50] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr7 = ODR(myData7, myModel, guesses7, maxit=1000)
odr7.set_job(fit_type=2)
output7 = odr7.run()
#output7.pprint()
Fit_out7 = (((((output7.beta[0]))*(2.*((X - output7.beta[4])**2. + (Y - output7.beta[3])**2.)/(output7.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output7.beta[4])**2. + (Y - output7.beta[3])**2.)/(output7.beta[1])**2.))) + output7.beta[2]
print 'done7'

myData8 = Data([Q, W], data8)
myModel = Model(function)
guesses8 = [25000, 30, 450, -50, -50] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr8 = ODR(myData8, myModel, guesses8, maxit=1000)
odr8.set_job(fit_type=2)
output8 = odr8.run()
#output8.pprint()
Fit_out8 = (((((output8.beta[0]))*(2.*((X - output8.beta[4])**2. + (Y - output8.beta[3])**2.)/(output8.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output8.beta[4])**2. + (Y - output8.beta[3])**2.)/(output8.beta[1])**2.))) + output8.beta[2]
print 'done8'

myData9 = Data([Q, W], data9)
myModel = Model(function)
guesses9 = [25000, 40, 450, -50, -50] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr9 = ODR(myData9, myModel, guesses9, maxit=1000)
odr9.set_job(fit_type=2)
output9 = odr9.run()
#output1.pprint()
Fit_out9 = (((((output9.beta[0]))*(2.*((X - output9.beta[4])**2. + (Y - output9.beta[3])**2.)/(output9.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output9.beta[4])**2. + (Y - output9.beta[3])**2.)/(output9.beta[1])**2.))) + output9.beta[2]
print 'done9'

myData10 = Data([Q, W], data10)
myModel = Model(function)
guesses10 = [25000, 40, 450, -50, -50] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr10 = ODR(myData10, myModel, guesses10, maxit=1000)
odr10.set_job(fit_type=2)
output10 = odr10.run()
#output1.pprint()
Fit_out10 = (((((output10.beta[0]))*(2.*((X - output10.beta[4])**2. + (Y - output10.beta[3])**2.)/(output10.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output10.beta[4])**2. + (Y - output10.beta[3])**2.)/(output10.beta[1])**2.))) + output10.beta[2]
print 'done10'

myData11 = Data([Q, W], data11)
myModel = Model(function)
guesses11 = [25000, 40, 850, -50, -50] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr11 = ODR(myData11, myModel, guesses11, maxit=1000)
odr11.set_job(fit_type=2)
output11 = odr11.run()
#output1.pprint()
Fit_out11 = (((((output11.beta[0]))*(2.*((X - output11.beta[4])**2. + (Y - output11.beta[3])**2.)/(output11.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output11.beta[4])**2. + (Y - output11.beta[3])**2.)/(output11.beta[1])**2.))) + output11.beta[2]
print 'done11'

myData12 = Data([Q, W], data12)
myModel = Model(function)
guesses12 = [25000, 40, 850, -50, -50] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr12 = ODR(myData12, myModel, guesses12, maxit=1000)
odr12.set_job(fit_type=2)
output12 = odr12.run()
#output1.pprint()
Fit_out12 = (((((output12.beta[0]))*(2.*((X - output12.beta[4])**2. + (Y - output12.beta[3])**2.)/(output12.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output12.beta[4])**2. + (Y - output12.beta[3])**2.)/(output12.beta[1])**2.))) + output12.beta[2]
print 'done12'

myData13 = Data([Q, W], data13)
myModel = Model(function)
guesses13 = [25000, 40, 850, -50, -50] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr13 = ODR(myData13, myModel, guesses13, maxit=1000)
odr13.set_job(fit_type=2)
output13 = odr13.run()
#output1.pprint()
Fit_out13 = (((((output13.beta[0]))*(2.*((X - output13.beta[4])**2. + (Y - output13.beta[3])**2.)/(output13.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output13.beta[4])**2. + (Y - output13.beta[3])**2.)/(output13.beta[1])**2.))) + output13.beta[2]
print 'done13'

myData14 = Data([Q, W], data14)
myModel = Model(function)
guesses14 = [25000, 50, 850, -50, -50] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr14 = ODR(myData14, myModel, guesses14, maxit=1000)
odr14.set_job(fit_type=2)
output14 = odr14.run()
#output1.pprint()
Fit_out14 = (((((output14.beta[0]))*(2.*((X - output14.beta[4])**2. + (Y - output14.beta[3])**2.)/(output14.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output14.beta[4])**2. + (Y - output14.beta[3])**2.)/(output14.beta[1])**2.))) + output14.beta[2]
print 'done14'

myData15 = Data([Q, W], data15)
myModel = Model(function)
guesses15 = [25000, 50, 850, -50, -50] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr15 = ODR(myData15, myModel, guesses15, maxit=1000)
odr15.set_job(fit_type=2)
output15 = odr15.run()
#output1.pprint()
Fit_out15 = (((((output15.beta[0]))*(2.*((X - output15.beta[4])**2. + (Y - output15.beta[3])**2.)/(output15.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output15.beta[4])**2. + (Y - output15.beta[3])**2.)/(output15.beta[1])**2.))) + output15.beta[2]
print 'done15'

myData16 = Data([Q, W], data16)
myModel = Model(function)
guesses16 = [25000, 50, 850, -50, -50] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr16 = ODR(myData16, myModel, guesses16, maxit=1000)
odr16.set_job(fit_type=2)
output16 = odr16.run()
#output1.pprint()
Fit_out16 = (((((output16.beta[0]))*(2.*((X - output16.beta[4])**2. + (Y - output16.beta[3])**2.)/(output16.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output16.beta[4])**2. + (Y - output16.beta[3])**2.)/(output16.beta[1])**2.))) + output16.beta[2]
print 'done16'

myData17 = Data([Q, W], data17)
myModel = Model(function)
guesses17 = [25000, 50, 850, -50, -50] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr17 = ODR(myData17, myModel, guesses17, maxit=1000)
odr17.set_job(fit_type=2)
output17 = odr17.run()
#output1.pprint()
Fit_out17 = (((((output17.beta[0]))*(2.*((X - output17.beta[4])**2. + (Y - output17.beta[3])**2.)/(output17.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output17.beta[4])**2. + (Y - output17.beta[3])**2.)/(output17.beta[1])**2.))) + output17.beta[2]
print 'done17'

myData18 = Data([Q, W], data18)
myModel = Model(function)
guesses18 = [25000, 50, 850, -50, -50] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr18 = ODR(myData18, myModel, guesses18, maxit=1000)
odr18.set_job(fit_type=2)
output18 = odr18.run()
#output1.pprint()
Fit_out18 = (((((output18.beta[0]))*(2.*((X - output18.beta[4])**2. + (Y - output18.beta[3])**2.)/(output18.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output18.beta[4])**2. + (Y - output18.beta[3])**2.)/(output18.beta[1])**2.))) + output18.beta[2]
print 'done18'

myData19 = Data([Q, W], data19)
myModel = Model(function)
guesses19 = [25000, 50, 850, -50, -50] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr19 = ODR(myData19, myModel, guesses19, maxit=1000)
odr19.set_job(fit_type=2)
output19 = odr19.run()
#output19.pprint()
Fit_out19 = (((((output19.beta[0]))*(2.*((X - output19.beta[4])**2. + (Y - output19.beta[3])**2.)/(output19.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output19.beta[4])**2. + (Y - output19.beta[3])**2.)/(output19.beta[1])**2.))) + output19.beta[2]
print 'done19'

myData20 = Data([Q, W], data20)
myModel = Model(function)
guesses20 = [25000, 60, 850, -50, -50] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr20 = ODR(myData20, myModel, guesses20, maxit=1000)
odr20.set_job(fit_type=2)
output20 = odr20.run()
#output20.pprint()
Fit_out20 = (((((output20.beta[0]))*(2.*((X - output20.beta[4])**2. + (Y - output20.beta[3])**2.)/(output20.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output20.beta[4])**2. + (Y - output20.beta[3])**2.)/(output20.beta[1])**2.))) + output20.beta[2]
print 'done20'

myData21 = Data([Q, W], data21)
myModel = Model(function)
guesses21 = [25000, 60, 850, -50, -50] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr21 = ODR(myData21, myModel, guesses21, maxit=1000)
odr21.set_job(fit_type=2)
output21 = odr21.run()
#output21.pprint()
Fit_out21 = (((((output21.beta[0]))*(2.*((X - output21.beta[4])**2. + (Y - output21.beta[3])**2.)/(output21.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output21.beta[4])**2. + (Y - output21.beta[3])**2.)/(output21.beta[1])**2.))) + output21.beta[2]
print 'done21'

myData22 = Data([Q, W], data22)
myModel = Model(function)
guesses22 = [25000, 60, 850, -50, -50] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr22 = ODR(myData22, myModel, guesses22, maxit=1000)
odr22.set_job(fit_type=2)
output22 = odr22.run()
#output22.pprint()
Fit_out22 = (((((output22.beta[0]))*(2.*((X - output22.beta[4])**2. + (Y - output22.beta[3])**2.)/(output22.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output22.beta[4])**2. + (Y - output22.beta[3])**2.)/(output22.beta[1])**2.))) + output22.beta[2]
print 'done22'

myData23 = Data([Q, W], data23)
myModel = Model(function)
guesses23 = [25000, 60, 850, -50, -50] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr23 = ODR(myData23, myModel, guesses23, maxit=1000)
odr23.set_job(fit_type=2)
output23 = odr23.run()
#output23.pprint()
Fit_out23 = (((((output23.beta[0]))*(2.*((X - output23.beta[4])**2. + (Y - output23.beta[3])**2.)/(output23.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output23.beta[4])**2. + (Y - output23.beta[3])**2.)/(output23.beta[1])**2.))) + output23.beta[2]
print 'done23'

myData24 = Data([Q, W], data24)
myModel = Model(function)
guesses24 = [25000, 70, 850, -50, -50] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr24 = ODR(myData24, myModel, guesses24, maxit=1000)
odr24.set_job(fit_type=2)
output24 = odr24.run()
#output24.pprint()
Fit_out24 = (((((output24.beta[0]))*(2.*((X - output24.beta[4])**2. + (Y - output24.beta[3])**2.)/(output24.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output24.beta[4])**2. + (Y - output24.beta[3])**2.)/(output24.beta[1])**2.))) + output24.beta[2]
print 'done24'

myData25 = Data([Q, W], data25)
myModel = Model(function)
guesses25 = [25000, 70, 850, -50, -50] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr25 = ODR(myData25, myModel, guesses25, maxit=1000)
odr25.set_job(fit_type=2)
output25 = odr25.run()
#output25.pprint()
Fit_out25 = (((((output25.beta[0]))*(2.*((X - output25.beta[4])**2. + (Y - output25.beta[3])**2.)/(output25.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output25.beta[4])**2. + (Y - output25.beta[3])**2.)/(output25.beta[1])**2.))) + output25.beta[2]
print 'done25'

myData26 = Data([Q, W], data26)
myModel = Model(function)
guesses26 = [25000, 70, 850, -50, -50] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr26 = ODR(myData26, myModel, guesses26, maxit=1000)
odr26.set_job(fit_type=2)
output26 = odr26.run()
#output26.pprint()
Fit_out26 = (((((output26.beta[0]))*(2.*((X - output26.beta[4])**2. + (Y - output26.beta[3])**2.)/(output26.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output26.beta[4])**2. + (Y - output26.beta[3])**2.)/(output26.beta[1])**2.))) + output26.beta[2]
print 'done26'

myData27 = Data([Q, W], data27)
myModel = Model(function)
guesses27 = [25000, 70, 850, -50, -50] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr27 = ODR(myData27, myModel, guesses27, maxit=1000)
odr27.set_job(fit_type=2)
output27 = odr27.run()
#output27.pprint()
Fit_out27 = (((((output27.beta[0]))*(2.*((X - output27.beta[4])**2. + (Y - output27.beta[3])**2.)/(output27.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output27.beta[4])**2. + (Y - output27.beta[3])**2.)/(output27.beta[1])**2.))) + output27.beta[2]
print 'done27'

myData28 = Data([Q, W], data28)
myModel = Model(function)
guesses28 = [25000, 80, 850, -50, -50] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr28 = ODR(myData28, myModel, guesses28, maxit=1000)
odr28.set_job(fit_type=2)
output28 = odr28.run()
#output28.pprint()
Fit_out28 = (((((output28.beta[0]))*(2.*((X - output28.beta[4])**2. + (Y - output28.beta[3])**2.)/(output28.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output28.beta[4])**2. + (Y - output28.beta[3])**2.)/(output28.beta[1])**2.))) + output28.beta[2]
print 'done28'

myData29 = Data([Q, W], data29)
myModel = Model(function)
guesses29 = [25000, 80, 850, -50, -50] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr29 = ODR(myData29, myModel, guesses29, maxit=1000)
odr29.set_job(fit_type=2)
output29 = odr29.run()
#output29.pprint()
Fit_out29 = (((((output29.beta[0]))*(2.*((X - output29.beta[4])**2. + (Y - output29.beta[3])**2.)/(output29.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output29.beta[4])**2. + (Y - output29.beta[3])**2.)/(output29.beta[1])**2.))) + output29.beta[2]
print 'done29'

myData30 = Data([Q, W], data30)
myModel = Model(function)
guesses30 = [25000, 80, 850, -50, -50] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr30 = ODR(myData30, myModel, guesses30, maxit=1000)
odr30.set_job(fit_type=2)
output30 = odr30.run()
#output30.pprint()
Fit_out30 = (((((output30.beta[0]))*(2.*((X - output30.beta[4])**2. + (Y - output30.beta[3])**2.)/(output30.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output30.beta[4])**2. + (Y - output30.beta[3])**2.)/(output30.beta[1])**2.))) + output30.beta[2]
print 'done30'

myData31 = Data([Q, W], data31)
myModel = Model(function)
guesses31 = [25000, 90, 850, -50, -50] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr31 = ODR(myData31, myModel, guesses31, maxit=1000)
odr31.set_job(fit_type=2)
output31 = odr31.run()
#output31.pprint()
Fit_out31 = (((((output31.beta[0]))*(2.*((X - output31.beta[4])**2. + (Y - output31.beta[3])**2.)/(output31.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output31.beta[4])**2. + (Y - output31.beta[3])**2.)/(output31.beta[1])**2.))) + output31.beta[2]
print 'done31'

myData32 = Data([Q, W], data32)
myModel = Model(function)
guesses32 = [25000, 90, 850, -50, -50] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr32 = ODR(myData32, myModel, guesses32, maxit=1000)
odr32.set_job(fit_type=2)
output32 = odr32.run()
#output32.pprint()
Fit_out32 = (((((output32.beta[0]))*(2.*((X - output32.beta[4])**2. + (Y - output32.beta[3])**2.)/(output32.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output32.beta[4])**2. + (Y - output32.beta[3])**2.)/(output32.beta[1])**2.))) + output32.beta[2]
print 'done32'

myData33 = Data([Q, W], data33)
myModel = Model(function)
guesses33 = [25000, 90, 850, -50, -50] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr33 = ODR(myData33, myModel, guesses33, maxit=1000)
odr33.set_job(fit_type=2)
output33 = odr33.run()
#output33.pprint()
Fit_out33 = (((((output33.beta[0]))*(2.*((X - output33.beta[4])**2. + (Y - output33.beta[3])**2.)/(output33.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output33.beta[4])**2. + (Y - output33.beta[3])**2.)/(output33.beta[1])**2.))) + output33.beta[2]
print 'done33'

myData34 = Data([Q, W], data34)
myModel = Model(function)
guesses34 = [25000, 100, 850, -50, -50] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr34 = ODR(myData34, myModel, guesses34, maxit=1000)
odr34.set_job(fit_type=2)
output34 = odr34.run()
#output34.pprint()
Fit_out34 = (((((output34.beta[0]))*(2.*((X - output34.beta[4])**2. + (Y - output34.beta[3])**2.)/(output34.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output34.beta[4])**2. + (Y - output34.beta[3])**2.)/(output34.beta[1])**2.))) + output34.beta[2]
print 'done34'

myData35 = Data([Q, W], data35)
myModel = Model(function)
guesses35 = [25000, 100, 850, -50, -50] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr35 = ODR(myData35, myModel, guesses35, maxit=1000)
odr35.set_job(fit_type=2)
output35 = odr35.run()
#output35.pprint()
Fit_out35 = (((((output35.beta[0]))*(2.*((X - output35.beta[4])**2. + (Y - output35.beta[3])**2.)/(output35.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output35.beta[4])**2. + (Y - output35.beta[3])**2.)/(output35.beta[1])**2.))) + output35.beta[2]
print 'done35'

myData36 = Data([Q, W], data36)
myModel = Model(function)
guesses36 = [25000, 100, 850, -50, -50] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr36 = ODR(myData36, myModel, guesses36, maxit=1000)
odr36.set_job(fit_type=2)
output36 = odr36.run()
#output36.pprint()
Fit_out36 = (((((output36.beta[0]))*(2.*((X - output36.beta[4])**2. + (Y - output36.beta[3])**2.)/(output36.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output36.beta[4])**2. + (Y - output36.beta[3])**2.)/(output36.beta[1])**2.))) + output36.beta[2]
print 'done36'

myData37 = Data([Q, W], data37)
myModel = Model(function)
guesses37 = [25000, 100, 850, -50, -50] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr37 = ODR(myData37, myModel, guesses37, maxit=1000)
odr37.set_job(fit_type=2)
output37 = odr37.run()
#output37.pprint()
Fit_out37 = (((((output37.beta[0]))*(2.*((X - output37.beta[4])**2. + (Y - output37.beta[3])**2.)/(output37.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output37.beta[4])**2. + (Y - output37.beta[3])**2.)/(output37.beta[1])**2.))) + output37.beta[2]
print 'done37'

myData38 = Data([Q, W], data38)
myModel = Model(function)
guesses38 = [25000, 120, 850, -50, -50] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr38 = ODR(myData38, myModel, guesses38, maxit=1000)
odr38.set_job(fit_type=2)
output38 = odr38.run()
#output38.pprint()
Fit_out38 = (((((output38.beta[0]))*(2.*((X - output38.beta[4])**2. + (Y - output38.beta[3])**2.)/(output38.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output38.beta[4])**2. + (Y - output38.beta[3])**2.)/(output38.beta[1])**2.))) + output38.beta[2]
print 'done38'

myData39 = Data([Q, W], data39)
myModel = Model(function)
guesses39 = [25000, 120, 850, -50, -50] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr39 = ODR(myData39, myModel, guesses39, maxit=1000)
odr39.set_job(fit_type=2)
output39 = odr39.run()
#output39.pprint()
Fit_out39 = (((((output39.beta[0]))*(2.*((X - output39.beta[4])**2. + (Y - output39.beta[3])**2.)/(output39.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output39.beta[4])**2. + (Y - output39.beta[3])**2.)/(output39.beta[1])**2.))) + output39.beta[2]
print 'done39'

myData40 = Data([Q, W], data40)
myModel = Model(function)
guesses40 = [25000, 120, 850, -50, -50] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr40 = ODR(myData40, myModel, guesses40, maxit=1000)
odr40.set_job(fit_type=2)
output40 = odr40.run()
#output40.pprint()
Fit_out40 = (((((output40.beta[0]))*(2.*((X - output40.beta[4])**2. + (Y - output40.beta[3])**2.)/(output40.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output40.beta[4])**2. + (Y - output40.beta[3])**2.)/(output40.beta[1])**2.))) + output40.beta[2]
print 'done40'

myData41 = Data([Q, W], data41)
myModel = Model(function)
guesses41 = [25000, 120, 850, -50, -50] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr41 = ODR(myData41, myModel, guesses41, maxit=1000)
odr41.set_job(fit_type=2)
output41 = odr41.run()
#output41.pprint()
Fit_out41 = (((((output41.beta[0]))*(2.*((X - output41.beta[4])**2. + (Y - output41.beta[3])**2.)/(output41.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output41.beta[4])**2. + (Y - output41.beta[3])**2.)/(output41.beta[1])**2.))) + output41.beta[2]
print 'done41'

myData42 = Data([Q, W], data42)
myModel = Model(function)
guesses42 = [25000, 120, 850, -20, -20] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr42 = ODR(myData42, myModel, guesses42, maxit=1000)
odr42.set_job(fit_type=2)
output42 = odr42.run()
#output42.pprint()
Fit_out42 = (((((output42.beta[0]))*(2.*((X - output42.beta[4])**2. + (Y - output42.beta[3])**2.)/(output42.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output42.beta[4])**2. + (Y - output42.beta[3])**2.)/(output42.beta[1])**2.))) + output42.beta[2]
print 'done42'

myData43 = Data([Q, W], data43)
myModel = Model(function)
guesses43 = [25000, 110, 850, 0, 0] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr43 = ODR(myData43, myModel, guesses43, maxit=1000)
odr43.set_job(fit_type=2)
output43 = odr43.run()
#output43.pprint()
Fit_out43 = (((((output43.beta[0]))*(2.*((X - output43.beta[4])**2. + (Y - output43.beta[3])**2.)/(output43.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output43.beta[4])**2. + (Y - output43.beta[3])**2.)/(output43.beta[1])**2.))) + output43.beta[2]
print 'done43'

myData44 = Data([Q, W], data44)
myModel = Model(function)
guesses44 = [25000, 120, 850, -50, -50] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr44 = ODR(myData44, myModel, guesses44, maxit=1000)
odr44.set_job(fit_type=2)
output44 = odr44.run()
#output44.pprint()
Fit_out44 = (((((output44.beta[0]))*(2.*((X - output44.beta[4])**2. + (Y - output44.beta[3])**2.)/(output44.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output44.beta[4])**2. + (Y - output44.beta[3])**2.)/(output44.beta[1])**2.))) + output44.beta[2]
print 'done44'

myData45 = Data([Q, W], data45)
myModel = Model(function)
guesses45 = [25000, 120, 850, -50, -50] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr45 = ODR(myData45, myModel, guesses45, maxit=1000)
odr45.set_job(fit_type=2)
output45 = odr45.run()
#output45.pprint()
Fit_out45 = (((((output45.beta[0]))*(2.*((X - output45.beta[4])**2. + (Y - output45.beta[3])**2.)/(output45.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output45.beta[4])**2. + (Y - output45.beta[3])**2.)/(output45.beta[1])**2.))) + output45.beta[2]
print 'done45'

myData46 = Data([Q, W], data46)
myModel = Model(function)
guesses46 = [25000, 120, 850, -50, -50] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr46 = ODR(myData46, myModel, guesses46, maxit=1000)
odr46.set_job(fit_type=2)
output46 = odr46.run()
#output46.pprint()
Fit_out46 = (((((output46.beta[0]))*(2.*((X - output46.beta[4])**2. + (Y - output46.beta[3])**2.)/(output46.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output46.beta[4])**2. + (Y - output46.beta[3])**2.)/(output46.beta[1])**2.))) + output46.beta[2]
print 'done46'

myData47 = Data([Q, W], data47)
myModel = Model(function)
guesses47 = [25000, 180, 850, -50, -50] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr47 = ODR(myData47, myModel, guesses47, maxit=1000)
odr47.set_job(fit_type=2)
output47 = odr47.run()
#output47.pprint()
Fit_out47 = (((((output47.beta[0]))*(2.*((X - output47.beta[4])**2. + (Y - output47.beta[3])**2.)/(output47.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output47.beta[4])**2. + (Y - output47.beta[3])**2.)/(output47.beta[1])**2.))) + output47.beta[2]
print 'done47'

myData48 = Data([Q, W], data48)
myModel = Model(function)
guesses48 = [25000, 120, 850, 0, 0] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr48 = ODR(myData48, myModel, guesses48, maxit=1000)
odr48.set_job(fit_type=2)
output48 = odr48.run()
#output48.pprint()
Fit_out48 = (((((output48.beta[0]))*(2.*((X - output48.beta[4])**2. + (Y - output48.beta[3])**2.)/(output48.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output48.beta[4])**2. + (Y - output48.beta[3])**2.)/(output48.beta[1])**2.))) + output48.beta[2]
print 'done48'

myData49 = Data([Q, W], data49)
myModel = Model(function)
guesses49 = [25000, 120, 850, 0, 0] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr49 = ODR(myData49, myModel, guesses49, maxit=1000)
odr49.set_job(fit_type=2)
output49 = odr49.run()
#output49.pprint()
Fit_out49 = (((((output49.beta[0]))*(2.*((X - output49.beta[4])**2. + (Y - output49.beta[3])**2.)/(output49.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output49.beta[4])**2. + (Y - output49.beta[3])**2.)/(output49.beta[1])**2.))) + output49.beta[2]
print 'done49'

myData50 = Data([Q, W], data50)
myModel = Model(function)
guesses50 = [25000, 120, 850, 0, 0] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr50 = ODR(myData50, myModel, guesses50, maxit=1000)
odr50.set_job(fit_type=2)
output50 = odr50.run()
#output50.pprint()
Fit_out50 = (((((output50.beta[0]))*(2.*((X - output50.beta[4])**2. + (Y - output50.beta[3])**2.)/(output50.beta[1])**2.)))**l)*(np.exp((-2.)*(((X - output50.beta[4])**2. + (Y - output50.beta[3])**2.)/(output50.beta[1])**2.))) + output50.beta[2]
print 'done50'

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
maskedfit21 = np.where(data21 > zmin21, Fit_out21, 100)
maskedfit22 = np.where(data22 > zmin22, Fit_out22, 100)
maskedfit23 = np.where(data23 > zmin23, Fit_out23, 100)
maskedfit24 = np.where(data24 > zmin24, Fit_out24, 100)
maskedfit25 = np.where(data25 > zmin25, Fit_out25, 100)
maskedfit26 = np.where(data26 > zmin26, Fit_out26, 100)
maskedfit27 = np.where(data27 > zmin27, Fit_out27, 100)
maskedfit28 = np.where(data28 > zmin28, Fit_out28, 100)
maskedfit29 = np.where(data29 > zmin29, Fit_out29, 100)
maskedfit30 = np.where(data30 > zmin30, Fit_out30, 100)
maskedfit31 = np.where(data31 > zmin31, Fit_out31, 100)
maskedfit32 = np.where(data32 > zmin32, Fit_out32, 100)
maskedfit33 = np.where(data33 > zmin33, Fit_out33, 100)
maskedfit34 = np.where(data34 > zmin34, Fit_out34, 100)
maskedfit35 = np.where(data35 > zmin35, Fit_out35, 100)
maskedfit36 = np.where(data36 > zmin36, Fit_out36, 100)
maskedfit37 = np.where(data37 > zmin37, Fit_out37, 100)
maskedfit38 = np.where(data38 > zmin38, Fit_out38, 100)
maskedfit39 = np.where(data39 > zmin39, Fit_out39, 100)
maskedfit40 = np.where(data40 > zmin40, Fit_out40, 100)
maskedfit41 = np.where(data41 > zmin41, Fit_out41, 100)
maskedfit42 = np.where(data42 > zmin42, Fit_out42, 100)
maskedfit43 = np.where(data43 > zmin43, Fit_out43, 100)
maskedfit44 = np.where(data44 > zmin44, Fit_out44, 100)
maskedfit45 = np.where(data45 > zmin45, Fit_out45, 100)
maskedfit46 = np.where(data46 > zmin46, Fit_out46, 100)
maskedfit47 = np.where(data47 > zmin47, Fit_out47, 100)
maskedfit48 = np.where(data48 > zmin48, Fit_out48, 100)
maskedfit49 = np.where(data49 > zmin49, Fit_out49, 100)
maskedfit50 = np.where(data50 > zmin50, Fit_out50, 100)

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
Chisq21 = np.sum(np.sum((((maskeddata21 - maskedfit21))**2)/(maskedfit21)))
Chisq22 = np.sum(np.sum((((maskeddata22 - maskedfit22))**2)/(maskedfit22)))
Chisq23 = np.sum(np.sum((((maskeddata23 - maskedfit23))**2)/(maskedfit23)))
Chisq24 = np.sum(np.sum((((maskeddata24 - maskedfit24))**2)/(maskedfit24)))
Chisq25 = np.sum(np.sum((((maskeddata25 - maskedfit25))**2)/(maskedfit25)))
Chisq26 = np.sum(np.sum((((maskeddata26 - maskedfit26))**2)/(maskedfit26)))
Chisq27 = np.sum(np.sum((((maskeddata27 - maskedfit27))**2)/(maskedfit27)))
Chisq28 = np.sum(np.sum((((maskeddata28 - maskedfit28))**2)/(maskedfit28)))
Chisq29 = np.sum(np.sum((((maskeddata29 - maskedfit29))**2)/(maskedfit29)))
Chisq30 = np.sum(np.sum((((maskeddata30 - maskedfit30))**2)/(maskedfit30)))
Chisq31 = np.sum(np.sum((((maskeddata31 - maskedfit31))**2)/(maskedfit31)))
Chisq32 = np.sum(np.sum((((maskeddata32 - maskedfit32))**2)/(maskedfit32)))
Chisq33 = np.sum(np.sum((((maskeddata33 - maskedfit33))**2)/(maskedfit33)))
Chisq34 = np.sum(np.sum((((maskeddata34 - maskedfit34))**2)/(maskedfit34)))
Chisq35 = np.sum(np.sum((((maskeddata35 - maskedfit35))**2)/(maskedfit35)))
Chisq36 = np.sum(np.sum((((maskeddata36 - maskedfit36))**2)/(maskedfit36)))
Chisq37 = np.sum(np.sum((((maskeddata37 - maskedfit37))**2)/(maskedfit37)))
Chisq38 = np.sum(np.sum((((maskeddata38 - maskedfit38))**2)/(maskedfit38)))
Chisq39 = np.sum(np.sum((((maskeddata39 - maskedfit39))**2)/(maskedfit39)))
Chisq40 = np.sum(np.sum((((maskeddata40 - maskedfit40))**2)/(maskedfit40)))
Chisq41 = np.sum(np.sum((((maskeddata41 - maskedfit41))**2)/(maskedfit41)))
Chisq42 = np.sum(np.sum((((maskeddata42 - maskedfit42))**2)/(maskedfit42)))
Chisq43 = np.sum(np.sum((((maskeddata43 - maskedfit43))**2)/(maskedfit43)))
Chisq44 = np.sum(np.sum((((maskeddata44 - maskedfit44))**2)/(maskedfit44)))
Chisq45 = np.sum(np.sum((((maskeddata45 - maskedfit45))**2)/(maskedfit45)))
Chisq46 = np.sum(np.sum((((maskeddata46 - maskedfit46))**2)/(maskedfit46)))
Chisq47 = np.sum(np.sum((((maskeddata47 - maskedfit47))**2)/(maskedfit47)))
Chisq48 = np.sum(np.sum((((maskeddata48 - maskedfit48))**2)/(maskedfit48)))
Chisq49 = np.sum(np.sum((((maskeddata49 - maskedfit49))**2)/(maskedfit49)))
Chisq50 = np.sum(np.sum((((maskeddata50 - maskedfit50))**2)/(maskedfit50)))

print 'Chi Squared analysis done'

zsq1 = np.sum(np.sum(((maskeddata1 - maskedfit1)/(data1_sd))**2)/maskedfit1)
zsq2 = np.sum(np.sum(((maskeddata2 - maskedfit2)/(data2_sd))**2)/maskedfit2)
zsq3 = np.sum(np.sum(((maskeddata3 - maskedfit3)/(data3_sd))**2)/maskedfit3)
zsq4 = np.sum(np.sum(((maskeddata4 - maskedfit4)/(data4_sd))**2)/maskedfit4)
zsq5 = np.sum(np.sum(((maskeddata5 - maskedfit5)/(data5_sd))**2)/maskedfit5)
zsq6 = np.sum(np.sum(((maskeddata6 - maskedfit6)/(data6_sd))**2)/maskedfit6)
zsq7 = np.sum(np.sum(((maskeddata7 - maskedfit7)/(data7_sd))**2)/maskedfit7)
zsq8 = np.sum(np.sum(((maskeddata8 - maskedfit8)/(data8_sd))**2)/maskedfit8)
zsq9 = np.sum(np.sum(((maskeddata9 - maskedfit9)/(data9_sd))**2)/maskedfit9)
zsq10 = np.sum(np.sum(((maskeddata10 - maskedfit10)/(data10_sd))**2)/maskedfit10)
zsq11 = np.sum(np.sum(((maskeddata11 - maskedfit11)/(data11_sd))**2)/maskedfit11)
zsq12 = np.sum(np.sum(((maskeddata12 - maskedfit12)/(data12_sd))**2)/maskedfit12)
zsq13 = np.sum(np.sum(((maskeddata13 - maskedfit13)/(data13_sd))**2)/maskedfit13)
zsq14 = np.sum(np.sum(((maskeddata14 - maskedfit14)/(data14_sd))**2)/maskedfit14)
zsq15 = np.sum(np.sum(((maskeddata15 - maskedfit15)/(data15_sd))**2)/maskedfit15)
zsq16 = np.sum(np.sum(((maskeddata16 - maskedfit16)/(data16_sd))**2)/maskedfit16)
zsq17 = np.sum(np.sum(((maskeddata17 - maskedfit17)/(data17_sd))**2)/maskedfit17)
zsq18 = np.sum(np.sum(((maskeddata18 - maskedfit18)/(data18_sd))**2)/maskedfit18)
zsq19 = np.sum(np.sum(((maskeddata19 - maskedfit19)/(data19_sd))**2)/maskedfit19)
zsq20 = np.sum(np.sum(((maskeddata20 - maskedfit20)/(data20_sd))**2)/maskedfit20)
zsq21 = np.sum(np.sum(((maskeddata21 - maskedfit21)/(data21_sd))**2)/maskedfit21)
zsq22 = np.sum(np.sum(((maskeddata22 - maskedfit22)/(data22_sd))**2)/maskedfit22)
zsq23 = np.sum(np.sum(((maskeddata23 - maskedfit23)/(data23_sd))**2)/maskedfit23)
zsq24 = np.sum(np.sum(((maskeddata24 - maskedfit24)/(data24_sd))**2)/maskedfit24)
zsq25 = np.sum(np.sum(((maskeddata25 - maskedfit25)/(data25_sd))**2)/maskedfit25)
zsq26 = np.sum(np.sum(((maskeddata26 - maskedfit26)/(data26_sd))**2)/maskedfit26)
zsq27 = np.sum(np.sum(((maskeddata27 - maskedfit27)/(data27_sd))**2)/maskedfit27)
zsq28 = np.sum(np.sum(((maskeddata28 - maskedfit28)/(data28_sd))**2)/maskedfit28)
zsq29 = np.sum(np.sum(((maskeddata29 - maskedfit29)/(data29_sd))**2)/maskedfit29)
zsq30 = np.sum(np.sum(((maskeddata30 - maskedfit30)/(data30_sd))**2)/maskedfit30)
zsq31 = np.sum(np.sum(((maskeddata31 - maskedfit31)/(data31_sd))**2)/maskedfit31)
zsq32 = np.sum(np.sum(((maskeddata32 - maskedfit32)/(data32_sd))**2)/maskedfit32)
zsq33 = np.sum(np.sum(((maskeddata33 - maskedfit33)/(data33_sd))**2)/maskedfit33)
zsq34 = np.sum(np.sum(((maskeddata34 - maskedfit34)/(data34_sd))**2)/maskedfit34)
zsq35 = np.sum(np.sum(((maskeddata35 - maskedfit35)/(data35_sd))**2)/maskedfit35)
zsq36 = np.sum(np.sum(((maskeddata36 - maskedfit36)/(data36_sd))**2)/maskedfit36)
zsq37 = np.sum(np.sum(((maskeddata37 - maskedfit37)/(data37_sd))**2)/maskedfit37)
zsq38 = np.sum(np.sum(((maskeddata38 - maskedfit38)/(data38_sd))**2)/maskedfit38)
zsq39 = np.sum(np.sum(((maskeddata39 - maskedfit39)/(data39_sd))**2)/maskedfit39)
zsq40 = np.sum(np.sum(((maskeddata40 - maskedfit40)/(data40_sd))**2)/maskedfit40)
zsq41 = np.sum(np.sum(((maskeddata41 - maskedfit41)/(data41_sd))**2)/maskedfit41)
zsq42 = np.sum(np.sum(((maskeddata42 - maskedfit42)/(data42_sd))**2)/maskedfit42)
zsq43 = np.sum(np.sum(((maskeddata43 - maskedfit43)/(data43_sd))**2)/maskedfit43)
zsq44 = np.sum(np.sum(((maskeddata44 - maskedfit44)/(data44_sd))**2)/maskedfit44)
zsq45 = np.sum(np.sum(((maskeddata45 - maskedfit45)/(data45_sd))**2)/maskedfit45)
zsq46 = np.sum(np.sum(((maskeddata46 - maskedfit46)/(data46_sd))**2)/maskedfit46)
zsq47 = np.sum(np.sum(((maskeddata47 - maskedfit47)/(data47_sd))**2)/maskedfit47)
zsq48 = np.sum(np.sum(((maskeddata48 - maskedfit48)/(data48_sd))**2)/maskedfit48)
zsq49 = np.sum(np.sum(((maskeddata49 - maskedfit49)/(data49_sd))**2)/maskedfit49)
zsq50 = np.sum(np.sum(((maskeddata50 - maskedfit50)/(data50_sd))**2)/maskedfit50)

print 'standard deviation analysis done'

adj_chi1 = [Chisq1/N1, Chisq2/N2, Chisq3/N3, Chisq4/N4, Chisq5/N5, Chisq6/N6, Chisq7/N7, Chisq8/N8, Chisq9/N9, Chisq10/N10, Chisq11/N11, Chisq12/N12, Chisq13/N13, Chisq14/N14, Chisq15/N15, Chisq16/N16, Chisq17/N17, Chisq18/N18, Chisq19/N19, Chisq20/N20, Chisq21/N21, Chisq22/N22, Chisq23/N23, Chisq24/N24, Chisq25/N25, Chisq26/N26, Chisq27/N27, Chisq28/N28, Chisq29/N29, Chisq30/N30, Chisq31/N31, Chisq32/N32, Chisq33/N33, Chisq34/N34, Chisq35/N35, Chisq36/N36, Chisq37/N37, Chisq38/N38, Chisq39/N39, Chisq40/N40, Chisq41/N41, Chisq42/N42, Chisq43/N43, Chisq44/N44, Chisq45/N45, Chisq46/N46, Chisq47/N47, Chisq48/N48, Chisq49/N49, Chisq50/N50]
adj_z = [zsq1/N1, zsq2/N2, zsq3/N3, zsq4/N4, zsq5/N5, zsq6/N6, zsq7/N7, zsq8/N8, zsq9/N9, zsq10/N10, zsq11/N11, zsq12/N12, zsq13/N13, zsq14/N14, zsq15/N15, zsq16/N16, zsq17/N17, zsq18/N18, zsq19/N19, zsq20/N20, zsq21/N21, zsq22/N22, zsq23/N23, zsq24/N24, zsq25/N25, zsq26/N26, zsq27/N27, zsq28/N28, zsq29/N29, zsq30/N30, zsq31/N31, zsq32/N32, zsq33/N33, zsq34/N34, zsq35/N35, zsq36/N36, zsq37/N37, zsq38/N38, zsq39/N39, zsq40/N40, zsq41/N41, zsq42/N42, zsq43/N43, zsq44/N44, zsq45/N45, zsq46/N46, zsq47/N47, zsq48/N48, zsq49/N49, zsq50/N50]
scales = [output1.beta[0], output2.beta[0], output3.beta[0], output4.beta[0], output5.beta[0], output6.beta[0], output7.beta[0], output8.beta[0], output9.beta[0], output10.beta[0], output11.beta[0], output12.beta[0], output13.beta[0], output14.beta[0], output15.beta[0], output16.beta[0], output17.beta[0], output18.beta[0], output19.beta[0], output20.beta[0], output21.beta[0], output22.beta[0], output23.beta[0], output24.beta[0], output25.beta[0], output26.beta[0], output27.beta[0], output28.beta[0], output29.beta[0], output30.beta[0], output31.beta[0], output32.beta[0], output33.beta[0], output34.beta[0], output35.beta[0], output36.beta[0], output37.beta[0], output38.beta[0], output39.beta[0], output40.beta[0], output41.beta[0], output42.beta[0], output43.beta[0], output44.beta[0], output45.beta[0], output46.beta[0], output47.beta[0], output48.beta[0], output49.beta[0], output50.beta[0]]

#Set up the display window for the plots and the plots themselves

#fig1 = plt.figure()
#fig2 = plt.figure()
#fig3 = plt.figure()
#fig4 = plt.figure()
#fig5 = plt.figure()
#fig6 = plt.figure()
#fig7 = plt.figure()
#fig8 = plt.figure()
#fig9 = plt.figure()
#fig10 = plt.figure()
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
fig21 = plt.figure()
fig22 = plt.figure()


#plot1_1=fig1.add_subplot(111, projection='3d')
#plot1_2=fig1.add_subplot(122, projection='3d')

#plot2_1=fig2.add_subplot(111, projection='3d')
#plot2_2=fig2.add_subplot(122, projection='3d')

#plot3_1=fig3.add_subplot(111, projection='3d')
#plot3_2=fig3.add_subplot(122, projection='3d')

#plot4_1=fig4.add_subplot(121, projection='3d')
#plot4_2=fig4.add_subplot(122, projection='3d')

#plot5_1=fig5.add_subplot(121, projection='3d')
#plot5_2=fig5.add_subplot(122, projection='3d')

#plot6_1=fig6.add_subplot(121, projection='3d')
#plot6_2=fig6.add_subplot(122, projection='3d')

#plot7_1=fig7.add_subplot(121, projection='3d')
#plot7_2=fig7.add_subplot(122, projection='3d')

#plot8_1=fig8.add_subplot(121, projection='3d')
#plot8_2=fig8.add_subplot(122, projection='3d')

#plot9_1=fig9.add_subplot(121, projection='3d')
#plot9_2=fig9.add_subplot(122, projection='3d')

#plot10_1=fig10.add_subplot(121, projection='3d')
#plot10_2=fig10.add_subplot(122, projection='3d')

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

plot21=fig21.add_subplot(111)
plot22=fig22.add_subplot(111)


#plot1_1.set_title('LG 1-0 mode at 25mm')
#plot2_1.set_title('LG 1-0 mode at 150mm')
#plot3_1.set_title('LG 1-0 mode at 1100mm')

plot21.set_title('Difference Squared/N vs. Distance')
plot22.set_title('# of Standard Dev. Sq./N vs. Distance')

#plot1_1.plot_surface(Y, X, data2, rstride = 4, cstride = 4, linewidth = 0.05, cmap = 'cool')
#plot1_2.plot_surface(Y, X, maskeddata1, rstride = 8, cstride = 8, cmap = 'hot', linewidth = 0.05) 

#plot2_1.plot_surface(Y, X, data6, rstride = 4, cstride = 4, linewidth = 0.05, cmap = 'cool')
#plot2_2.plot_surface(Y, X, maskeddata2, rstride = 8, cstride = 8, cmap = 'hot', linewidth = 0.05) 

#plot3_1.plot_surface(Y, X, data44, rstride = 4, cstride = 4, linewidth = 0.05, cmap = 'cool')
#plot3_2.plot_surface(Y, X, maskeddata3, rstride = 8, cstride = 8, cmap = 'hot', linewidth = 0.05) 

#plot4_1.plot_surface(Y, X, maskedfit4, rstride = 5, cstride = 5, linewidth = 0.05, cmap = 'cool')
#plot4_2.plot_surface(Y, X, maskeddata4, rstride = 8, cstride = 8, cmap = 'hot', linewidth = 0.05) 

#plot5_1.plot_surface(Y, X, maskedfit5, rstride = 5, cstride = 5, linewidth = 0.05, cmap = 'cool')
#plot5_2.plot_surface(Y, X, maskeddata5, rstride = 8, cstride = 8, cmap = 'hot', linewidth = 0.05) 

#plot6_1.plot_surface(Y, X, maskedfit6, rstride = 5, cstride = 5, linewidth = 0.05, cmap = 'cool')
#plot6_2.plot_surface(Y, X, maskeddata6, rstride = 8, cstride = 8, cmap = 'hot', linewidth = 0.05) 

#plot7_1.plot_surface(Y, X, maskedfit7, rstride = 8, cstride = 8, linewidth = 0.05, cmap = 'cool')
#plot7_2.plot_surface(Y, X, maskeddata7, rstride = 8, cstride = 8, cmap = 'hot', linewidth = 0.05) 

#plot8_1.plot_surface(Y, X, maskedfit8, rstride = 8, cstride = 8, linewidth = 0.05, cmap = 'cool')
#plot8_2.plot_surface(Y, X, maskeddata8, rstride = 8, cstride = 8, cmap = 'hot', linewidth = 0.05)

#plot9_1.plot_surface(Y, X, maskedfit9, rstride = 8, cstride = 8, linewidth = 0.05, cmap = 'cool')
#plot9_2.plot_surface(Y, X, maskeddata9, rstride = 8, cstride = 8, cmap = 'hot', linewidth = 0.05) 

#plot10_1.plot_surface(Y, X, maskedfit10, rstride = 8, cstride = 8, linewidth = 0.05, cmap = 'cool')
#plot10_2.plot_surface(Y, X, maskeddata10, rstride = 8, cstride = 8, cmap = 'hot', linewidth = 0.05) 

#plot11_1.plot_surface(Y, X, maskedfit11, rstride = 8, cstride = 8, linewidth = 0.05, cmap = 'cool')
#plot11_2.plot_surface(Y, X, maskeddata11, rstride = 8, cstride = 8, cmap = 'hot', linewidth = 0.05) 

#plot12_1.plot_surface(Y, X, maskedfit12, rstride = 8, cstride = 8, linewidth = 0.05, cmap = 'cool')
#plot12_2.plot_surface(Y, X, maskeddata12, rstride = 8, cstride = 8, cmap = 'hot', linewidth = 0.05) 

#plot13_1.plot_surface(Y, X, maskedfit13, rstride = 8, cstride = 8, linewidth = 0.05, cmap = 'cool')
#plot13_2.plot_surface(Y, X, maskeddata13, rstride = 8, cstride = 8, cmap = 'hot', linewidth = 0.05) 

#plot14_1.plot_surface(Y, X, maskedfit14, rstride = 8, cstride = 8, linewidth = 0.05, cmap = 'cool')
#plot14_2.plot_surface(Y, X, maskeddata14, rstride = 8, cstride = 8, cmap = 'hot', linewidth = 0.05) 

#plot15_1.plot_surface(Y, X, maskedfit15, rstride = 8, cstride = 8, linewidth = 0.05, cmap = 'cool')
#plot15_2.plot_surface(Y, X, maskeddata15, rstride = 8, cstride = 8, cmap = 'hot', linewidth = 0.05) 

#plot16_1.plot_surface(Y, X, maskedfit16, rstride = 8, cstride = 8, linewidth = 0.05, cmap = 'cool')
#plot16_2.plot_surface(Y, X, maskeddata16, rstride = 8, cstride = 8, cmap = 'hot', linewidth = 0.05) 

#plot17_1.plot_surface(Y, X, maskedfit17, rstride = 8, cstride = 8, linewidth = 0.05, cmap = 'cool')
#plot17_2.plot_surface(Y, X, maskeddata17, rstride = 8, cstride = 8, cmap = 'hot', linewidth = 0.05) 

#plot18_1.plot_surface(Y, X, maskedfit18, rstride = 8, cstride = 8, linewidth = 0.05, cmap = 'cool')
#plot18_2.plot_surface(Y, X, maskeddata18, rstride = 8, cstride = 8, cmap = 'hot', linewidth = 0.05) 

#plot19_1.plot_surface(Y, X, maskedfit19, rstride = 8, cstride = 8, linewidth = 0.05, cmap = 'cool')
#plot19_2.plot_surface(Y, X, maskeddata19, rstride = 8, cstride = 8, cmap = 'hot', linewidth = 0.05) 

#plot20_1.plot_surface(Y, X, maskedfit20, rstride = 8, cstride = 8, linewidth = 0.05, cmap = 'cool')
#plot20_2.plot_surface(Y, X, maskeddata20, rstride = 8, cstride = 8, cmap = 'hot', linewidth = 0.05) 

plot21.scatter(ranges, np.array(adj_chi1))
plot22.scatter(ranges, np.array(adj_z))

xfit = pylab.arange(1, 1500, 1)

def function2(params2, data):
	const = params2[0]
	lin = params2[1]
	quad = params2[2]
		
	return (const + (lin)*data + (quad)*(data**2))

fit_y = (np.array(adj_chi1))

myData51 = Data(ranges[6:], fit_y[6:])
myModel51 = Model(function2)
guesses51 = [5., 0.5, 0.1] 
odr51 = ODR(myData51, myModel51, guesses51, maxit=1000)
odr51.set_job(fit_type=2)
output51 = odr51.run()
output51.pprint()
Fit_out51 = output51.beta[2]*(xfit**2) + output51.beta[1]*xfit + output51.beta[0]

plot21.plot(xfit, Fit_out51)

#prints and labels all five parameters in the terminal, generates the plot in a new window.

plt.show()

####Library of Laguerre Polynomials for substitution in Z2
## 1, 0       1
## 1, 1       (2 - ((2*(X**2 + Y**2))/w**2))**2
## 2, 1       (3 - ((2*(X**2 + Y**2))/w**2))**2
## 5, 0       1
## 

