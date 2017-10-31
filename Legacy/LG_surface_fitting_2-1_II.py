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
xmin = 150
xmax = 1000 
ymin = 150
ymax = 1000 
x0 = (xmin + xmax)/2
y0 = (ymin + ymax)/2
ranges_inch = np.arange(0,24,1)
ranges = (ranges_inch)*25.4 + 5 + (25.4*25)


data26_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/26_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data26_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/26_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data26_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/26_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data26_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/26_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data26_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/26_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data26_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/26_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data26_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/26_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data26_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/26_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data26_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/26_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data26_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/26_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data27_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/27_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data27_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/27_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data27_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/27_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data27_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/27_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data27_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/27_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data27_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/27_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data27_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/27_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data27_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/27_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data27_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/27_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data27_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/27_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data28_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/28_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data28_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/28_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data28_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/28_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data28_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/28_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data28_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/28_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data28_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/28_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data28_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/28_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data28_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/28_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data28_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/28_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data28_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/28_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data29_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/29_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data29_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/29_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data29_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/29_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data29_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/29_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data29_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/29_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data29_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/29_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data29_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/29_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data29_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/29_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data29_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/29_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data29_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/29_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data30_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/30_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data30_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/30_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data30_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/30_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data30_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/30_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data30_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/30_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data30_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/30_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data30_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/30_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data30_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/30_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data30_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/30_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data30_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/30_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data31_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/31_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data31_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/31_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data31_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/31_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data31_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/31_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data31_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/31_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data31_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/31_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data31_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/31_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data31_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/31_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data31_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/31_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data31_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/31_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data32_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/32_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data32_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/32_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data32_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/32_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data32_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/32_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data32_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/32_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data32_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/32_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data32_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/32_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data32_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/32_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data32_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/32_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data32_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/32_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data33_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/33_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data33_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/33_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data33_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/33_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data33_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/33_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data33_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/33_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data33_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/33_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data33_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/33_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data33_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/33_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data33_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/33_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data33_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/33_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data34_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/34_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data34_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/34_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data34_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/34_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data34_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/34_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data34_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/34_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data34_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/34_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data34_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/34_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data34_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/34_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data34_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/34_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data34_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/34_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data35_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/35_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data35_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/35_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data35_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/35_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data35_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/35_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data35_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/35_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data35_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/35_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data35_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/35_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data35_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/35_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data35_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/35_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data35_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/35_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data36_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/36_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data36_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/36_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data36_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/36_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data36_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/36_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data36_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/36_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data36_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/36_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data36_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/36_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data36_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/36_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data36_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/36_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data36_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/36_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data37_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/37_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data37_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/37_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data37_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/37_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data37_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/37_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data37_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/37_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data37_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/37_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data37_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/37_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data37_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/37_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data37_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/37_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data37_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/37_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data38_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/38_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data38_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/38_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data38_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/38_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data38_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/38_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data38_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/38_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data38_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/38_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data38_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/38_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data38_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/38_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data38_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/38_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data38_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/38_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data39_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/39_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data39_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/39_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data39_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/39_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data39_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/39_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data39_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/39_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data39_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/39_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data39_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/39_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data39_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/39_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data39_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/39_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data39_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/39_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data40_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/40_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data40_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/40_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data40_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/40_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data40_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/40_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data40_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/40_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data40_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/40_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data40_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/40_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data40_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/40_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data40_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/40_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data40_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/40_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data41_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/41_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data41_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/41_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data41_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/41_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data41_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/41_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data41_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/41_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data41_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/41_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data41_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/41_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data41_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/41_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data41_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/41_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data41_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/41_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data42_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/42_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data42_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/42_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data42_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/42_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data42_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/42_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data42_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/42_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data42_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/42_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data42_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/42_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data42_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/42_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data42_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/42_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data42_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/42_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data43_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/43_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data43_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/43_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data43_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/43_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data43_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/43_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data43_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/43_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data43_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/43_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data43_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/43_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data43_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/43_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data43_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/43_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data43_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/43_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data44_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/44_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data44_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/44_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data44_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/44_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data44_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/44_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data44_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/44_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data44_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/44_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data44_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/44_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data44_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/44_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data44_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/44_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data44_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/44_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data45_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/45_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data45_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/45_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data45_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/45_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data45_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/45_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data45_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/45_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data45_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/45_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data45_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/45_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data45_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/45_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data45_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/45_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data45_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/45_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data46_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/46_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data46_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/46_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data46_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/46_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data46_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/46_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data46_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/46_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data46_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/46_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data46_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/46_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data46_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/46_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data46_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/46_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data46_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/46_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data47_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/47_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data47_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/47_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data47_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/47_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data47_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/47_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data47_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/47_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data47_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/47_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data47_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/47_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data47_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/47_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data47_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/47_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data47_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/47_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data48_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/48_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data48_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/48_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data48_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/48_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data48_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/48_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data48_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/48_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data48_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/48_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data48_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/48_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data48_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/48_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data48_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/48_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data48_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/48_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data49_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/49_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data49_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/49_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data49_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/49_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data49_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/49_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data49_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/49_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data49_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/49_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data49_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/49_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data49_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/49_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data49_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/49_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data49_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/49_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

#data50_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/50_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data50_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/50_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data50_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/50_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data50_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/50_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data50_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/50_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data50_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/50_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data50_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/50_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data50_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/50_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data50_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/50_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
#data50_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/50_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data26 = (data26_1 + data26_2 + data26_3 + data26_4 + data26_5 + data26_6 + data26_7)/7. # + data26_8 + data26_9 + data26_10)/10.
data27 = (data27_1 + data27_2 + data27_3 + data27_4 + data27_5 + data27_6 + data27_7)/7. # + data27_8 + data27_9 + data27_10)/10.
data28 = (data28_1 + data28_2 + data28_3 + data28_4 + data28_5 + data28_6 + data28_7)/7. # + data28_8 + data28_9 + data28_10)/10.
data29 = (data29_1 + data29_2 + data29_3 + data29_4 + data29_5 + data29_6 + data29_7)/7. # + data29_8 + data29_9 + data29_10)/10.
data30 = (data30_1 + data30_2 + data30_3 + data30_4 + data30_5 + data30_6 + data30_7)/7. # + data30_8 + data30_9 + data30_10)/10.
data31 = (data31_1 + data31_2 + data31_3 + data31_4 + data31_5 + data31_6 + data31_7)/7. # + data31_8 + data31_9 + data31_10)/10.
data32 = (data32_1 + data32_2 + data32_3 + data32_4 + data32_5 + data32_6 + data32_7)/7. # + data32_8 + data32_9 + data32_10)/10.
data33 = (data33_1 + data33_2 + data33_3 + data33_4 + data33_5 + data33_6 + data33_7)/7. # + data33_8 + data33_9 + data33_10)/10.
data34 = (data34_1 + data34_2 + data34_3 + data34_4 + data34_5 + data34_6 + data34_7)/7. # + data34_8 + data34_9 + data34_10)/10.
data35 = (data35_1 + data35_2 + data35_3 + data35_4 + data35_5 + data35_6 + data35_7)/7. # + data35_8 + data35_9 + data35_10)/10.
data36 = (data36_1 + data36_2 + data36_3 + data36_4 + data36_5 + data36_6 + data36_7)/7. # + data36_8 + data36_9 + data36_10)/10.
data37 = (data37_1 + data37_2 + data37_3 + data37_4 + data37_5 + data37_6 + data37_7)/7. # + data37_8 + data37_9 + data37_10)/10.
data38 = (data38_1 + data38_2 + data38_3 + data38_4 + data38_5 + data38_6 + data38_7)/7. # + data38_8 + data38_9 + data38_10)/10.
data39 = (data39_1 + data39_2 + data39_3 + data39_4 + data39_5 + data39_6 + data39_7)/7. # + data39_8 + data39_9 + data39_10)/10.
data40 = (data40_1 + data40_2 + data40_3 + data40_4 + data40_5 + data40_6 + data40_7)/7. # + data40_8 + data40_9 + data40_10)/10.
data41 = (data41_1 + data41_2 + data41_3 + data41_4 + data41_5 + data41_6 + data41_7)/7. # + data41_8 + data41_9 + data41_10)/10.
data42 = (data42_1 + data42_2 + data42_3 + data42_4 + data42_5 + data42_6 + data42_7)/7. # + data42_8 + data42_9 + data42_10)/10.
data43 = (data43_1 + data43_2 + data43_3 + data43_4 + data43_5 + data43_6 + data43_7)/7. # + data43_8 + data43_9 + data43_10)/10.
data44 = (data44_1 + data44_2 + data44_3 + data44_4 + data44_5 + data44_6 + data44_7)/7. # + data44_8 + data44_9 + data44_10)/10.
data45 = (data45_1 + data45_2 + data45_3 + data45_4 + data45_5 + data45_6 + data45_7)/7. # + data45_8 + data45_9 + data45_10)/10.
data46 = (data46_1 + data46_2 + data46_3 + data46_4 + data46_5 + data46_6 + data46_7)/7. # + data46_8 + data46_9 + data46_10)/10.
data47 = (data47_1 + data47_2 + data47_3 + data47_4 + data47_5 + data47_6 + data47_7)/7. # + data47_8 + data47_9 + data47_10)/10.
data48 = (data48_1 + data48_2 + data48_3 + data48_4 + data48_5 + data48_6 + data48_7)/7. # + data48_8 + data48_9 + data48_10)/10.
data49 = (data49_1 + data49_2 + data49_3 + data49_4 + data49_5 + data49_6 + data49_7)/7. # + data49_8 + data49_9 + data49_10)/10.
#data50 = (data50_1 + data50_2 + data50_3 + data50_4 + data50_5 + data50_6 + data50_7 + data50_8 + data50_9 + data50_10)/10.



print 'standard deviation calculation complete'

#generate a regular x-y space as independent variables

x = np.arange(xmin, xmax, 1)
y = np.arange(ymin, ymax, 1)

#specify radial and azimuthal modes of the LG Beam

l=2.
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

#Here's the fit function, broken into three terms for the sake of debugging. Used both for generating simulated cross-sections
Z1 = np.sqrt(A)*(2.*(X**2. + Y**2.)/(w**2.))**l
Z2 = (3. - ((2.*(X**2. + Y**2.))/w**2.))**2.  ##use Mathematica LaguerreL [bottom, top, function] to generate this term
Z3 = np.exp((-2.)*((X**2. + Y**2.)/w**2.)) 
Z4 = Z1*Z2*Z3 + B

#routine to mask data to remove bottom fraction of values
Crop_range = 0.2

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
#zmin50 = np.max(np.max(data50))*Crop_range

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
#maskeddata50 = np.where(data50 > zmin50, data50, 100)

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
#N50 = np.size(np.where(maskeddata50 > 100))


#define the funciton in terms of the 5 paramters, so that the ODR can process them
def function(params, data):
	scale = params[0] #= 9000
	baseline = params[2] #= 850
	width = params[1] #= 30
	y_0 = params[3] #=0
	x_0 = params[4] #=20
		
	return ((((params[0]))*(2.*((X - params[4])**2. + (Y - params[3])**2.)/(params[1])**2.))**l)*((3. - ((2.*((X - params[4])**2. + (Y - params[3])**2.))/(params[1])**2.))**2.)*(np.exp((-2.)*(((X - params[4])**2. + (Y - params[3])**2.)/(params[1])**2.))) + params[2]

myData26 = Data([Q, W], data26)
myModel = Model(function)
guesses26 = [25000, 70, 850, -25, 120] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr26 = ODR(myData26, myModel, guesses26, maxit=500)
odr26.set_job(fit_type=2)
output26 = odr26.run()
output26.pprint()
Fit_out26 = (((((output26.beta[0]))*(2.*((X - output26.beta[4])**2. + (Y - output26.beta[3])**2.)/(output26.beta[1])**2.)))**l)*((3. - ((2.*((X - output26.beta[4])**2. + (Y - output26.beta[3])**2.))/(output26.beta[1])**2))**2)*(np.exp((-2.)*(((X - output26.beta[4])**2. + (Y - output26.beta[3])**2.)/(output26.beta[1])**2.))) + output26.beta[2]
print 'done26'

myData27 = Data([Q, W], data27)
myModel = Model(function)
guesses27 = [25000, 70, 850, -25, 120] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr27 = ODR(myData27, myModel, guesses27, maxit=500)
odr27.set_job(fit_type=2)
output27 = odr27.run()
output27.pprint()
Fit_out27 = (((((output27.beta[0]))*(2.*((X - output27.beta[4])**2. + (Y - output27.beta[3])**2.)/(output27.beta[1])**2.)))**l)*((3. - ((2.*((X - output27.beta[4])**2. + (Y - output27.beta[3])**2.))/(output27.beta[1])**2))**2)*(np.exp((-2.)*(((X - output27.beta[4])**2. + (Y - output27.beta[3])**2.)/(output27.beta[1])**2.))) + output27.beta[2]
print 'done27'

myData28 = Data([Q, W], data28)
myModel = Model(function)
guesses28 = [25000, 80, 850, -25, 120] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr28 = ODR(myData28, myModel, guesses28, maxit=500)
odr28.set_job(fit_type=2)
output28 = odr28.run()
output28.pprint()
Fit_out28 = (((((output28.beta[0]))*(2.*((X - output28.beta[4])**2. + (Y - output28.beta[3])**2.)/(output28.beta[1])**2.)))**l)*((3. - ((2.*((X - output28.beta[4])**2. + (Y - output28.beta[3])**2.))/(output28.beta[1])**2))**2)*(np.exp((-2.)*(((X - output28.beta[4])**2. + (Y - output28.beta[3])**2.)/(output28.beta[1])**2.))) + output28.beta[2]
print 'done28'

myData29 = Data([Q, W], data29)
myModel = Model(function)
guesses29 = [25000, 80, 850, -25, 120] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr29 = ODR(myData29, myModel, guesses29, maxit=500)
odr29.set_job(fit_type=2)
output29 = odr29.run()
output29.pprint()
Fit_out29 = (((((output29.beta[0]))*(2.*((X - output29.beta[4])**2. + (Y - output29.beta[3])**2.)/(output29.beta[1])**2.)))**l)*((3. - ((2.*((X - output29.beta[4])**2. + (Y - output29.beta[3])**2.))/(output29.beta[1])**2))**2)*(np.exp((-2.)*(((X - output29.beta[4])**2. + (Y - output29.beta[3])**2.)/(output29.beta[1])**2.))) + output29.beta[2]
print 'done29'

myData30 = Data([Q, W], data30)
myModel = Model(function)
guesses30 = [25000, 80, 850, -25, 120] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr30 = ODR(myData30, myModel, guesses30, maxit=500)
odr30.set_job(fit_type=2)
output30 = odr30.run()
output30.pprint()
Fit_out30 = (((((output30.beta[0]))*(2.*((X - output30.beta[4])**2. + (Y - output30.beta[3])**2.)/(output30.beta[1])**2.)))**l)*((3. - ((2.*((X - output30.beta[4])**2. + (Y - output30.beta[3])**2.))/(output30.beta[1])**2))**2)*(np.exp((-2.)*(((X - output30.beta[4])**2. + (Y - output30.beta[3])**2.)/(output30.beta[1])**2.))) + output30.beta[2]
print 'done30'

myData31 = Data([Q, W], data31)
myModel = Model(function)
guesses31 = [25000, 90, 850, -25, 120] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr31 = ODR(myData31, myModel, guesses31, maxit=500)
odr31.set_job(fit_type=2)
output31 = odr31.run()
output31.pprint()
Fit_out31 = (((((output31.beta[0]))*(2.*((X - output31.beta[4])**2. + (Y - output31.beta[3])**2.)/(output31.beta[1])**2.)))**l)*((3. - ((2.*((X - output31.beta[4])**2. + (Y - output31.beta[3])**2.))/(output31.beta[1])**2))**2)*(np.exp((-2.)*(((X - output31.beta[4])**2. + (Y - output31.beta[3])**2.)/(output31.beta[1])**2.))) + output31.beta[2]
print 'done31'

myData32 = Data([Q, W], data32)
myModel = Model(function)
guesses32 = [25000, 90, 850, -25, 120] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr32 = ODR(myData32, myModel, guesses32, maxit=500)
odr32.set_job(fit_type=2)
output32 = odr32.run()
output32.pprint()
Fit_out32 = (((((output32.beta[0]))*(2.*((X - output32.beta[4])**2. + (Y - output32.beta[3])**2.)/(output32.beta[1])**2.)))**l)*((3. - ((2.*((X - output32.beta[4])**2. + (Y - output32.beta[3])**2.))/(output32.beta[1])**2))**2)*(np.exp((-2.)*(((X - output32.beta[4])**2. + (Y - output32.beta[3])**2.)/(output32.beta[1])**2.))) + output32.beta[2]
print 'done32'

myData33 = Data([Q, W], data33)
myModel = Model(function)
guesses33 = [25000, 90, 850, -25, 120] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr33 = ODR(myData33, myModel, guesses33, maxit=500)
odr33.set_job(fit_type=2)
output33 = odr33.run()
output33.pprint()
Fit_out33 = (((((output33.beta[0]))*(2.*((X - output33.beta[4])**2. + (Y - output33.beta[3])**2.)/(output33.beta[1])**2.)))**l)*((3. - ((2.*((X - output33.beta[4])**2. + (Y - output33.beta[3])**2.))/(output33.beta[1])**2))**2)*(np.exp((-2.)*(((X - output33.beta[4])**2. + (Y - output33.beta[3])**2.)/(output33.beta[1])**2.))) + output33.beta[2]
print 'done33'

myData34 = Data([Q, W], data34)
myModel = Model(function)
guesses34 = [25000, 100, 850, -25, 120] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr34 = ODR(myData34, myModel, guesses34, maxit=500)
odr34.set_job(fit_type=2)
output34 = odr34.run()
output34.pprint()
Fit_out34 = (((((output34.beta[0]))*(2.*((X - output34.beta[4])**2. + (Y - output34.beta[3])**2.)/(output34.beta[1])**2.)))**l)*((3. - ((2.*((X - output34.beta[4])**2. + (Y - output34.beta[3])**2.))/(output34.beta[1])**2))**2)*(np.exp((-2.)*(((X - output34.beta[4])**2. + (Y - output34.beta[3])**2.)/(output34.beta[1])**2.))) + output34.beta[2]
print 'done34'

myData35 = Data([Q, W], data35)
myModel = Model(function)
guesses35 = [25000, 100, 850, -25, 120] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr35 = ODR(myData35, myModel, guesses35, maxit=500)
odr35.set_job(fit_type=2)
output35 = odr35.run()
output35.pprint()
Fit_out35 = (((((output35.beta[0]))*(2.*((X - output35.beta[4])**2. + (Y - output35.beta[3])**2.)/(output35.beta[1])**2.)))**l)*((3. - ((2.*((X - output35.beta[4])**2. + (Y - output35.beta[3])**2.))/(output35.beta[1])**2))**2)*(np.exp((-2.)*(((X - output35.beta[4])**2. + (Y - output35.beta[3])**2.)/(output35.beta[1])**2.))) + output35.beta[2]
print 'done35'

myData36 = Data([Q, W], data36)
myModel = Model(function)
guesses36 = [25000, 100, 850, -25, 120] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr36 = ODR(myData36, myModel, guesses36, maxit=500)
odr36.set_job(fit_type=2)
output36 = odr36.run()
output36.pprint()
Fit_out36 = (((((output36.beta[0]))*(2.*((X - output36.beta[4])**2. + (Y - output36.beta[3])**2.)/(output36.beta[1])**2.)))**l)*((3. - ((2.*((X - output36.beta[4])**2. + (Y - output36.beta[3])**2.))/(output36.beta[1])**2))**2)*(np.exp((-2.)*(((X - output36.beta[4])**2. + (Y - output36.beta[3])**2.)/(output36.beta[1])**2.))) + output36.beta[2]
print 'done36'

myData37 = Data([Q, W], data37)
myModel = Model(function)
guesses37 = [25000, 100, 850, -25, 120] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr37 = ODR(myData37, myModel, guesses37, maxit=500)
odr37.set_job(fit_type=2)
output37 = odr37.run()
output37.pprint()
Fit_out37 = (((((output37.beta[0]))*(2.*((X - output37.beta[4])**2. + (Y - output37.beta[3])**2.)/(output37.beta[1])**2.)))**l)*((3. - ((2.*((X - output37.beta[4])**2. + (Y - output37.beta[3])**2.))/(output37.beta[1])**2))**2)*(np.exp((-2.)*(((X - output37.beta[4])**2. + (Y - output37.beta[3])**2.)/(output37.beta[1])**2.))) + output37.beta[2]
print 'done37'

myData38 = Data([Q, W], data38)
myModel = Model(function)
guesses38 = [25000, 120, 850, -25, 120] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr38 = ODR(myData38, myModel, guesses38, maxit=500)
odr38.set_job(fit_type=2)
output38 = odr38.run()
output38.pprint()
Fit_out38 = (((((output38.beta[0]))*(2.*((X - output38.beta[4])**2. + (Y - output38.beta[3])**2.)/(output38.beta[1])**2.)))**l)*((3. - ((2.*((X - output38.beta[4])**2. + (Y - output38.beta[3])**2.))/(output38.beta[1])**2))**2)*(np.exp((-2.)*(((X - output38.beta[4])**2. + (Y - output38.beta[3])**2.)/(output38.beta[1])**2.))) + output38.beta[2]
print 'done38'

myData39 = Data([Q, W], data39)
myModel = Model(function)
guesses39 = [25000, 120, 850, -25, 120] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr39 = ODR(myData39, myModel, guesses39, maxit=500)
odr39.set_job(fit_type=2)
output39 = odr39.run()
output39.pprint()
Fit_out39 = (((((output39.beta[0]))*(2.*((X - output39.beta[4])**2. + (Y - output39.beta[3])**2.)/(output39.beta[1])**2.)))**l)*((3. - ((2.*((X - output39.beta[4])**2. + (Y - output39.beta[3])**2.))/(output39.beta[1])**2))**2)*(np.exp((-2.)*(((X - output39.beta[4])**2. + (Y - output39.beta[3])**2.)/(output39.beta[1])**2.))) + output39.beta[2]
print 'done39'

myData40 = Data([Q, W], data40)
myModel = Model(function)
guesses40 = [25000, 130, 850, -25, 120] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr40 = ODR(myData40, myModel, guesses40, maxit=500)
odr40.set_job(fit_type=2)
output40 = odr40.run()
output40.pprint()
Fit_out40 = (((((output40.beta[0]))*(2.*((X - output40.beta[4])**2. + (Y - output40.beta[3])**2.)/(output40.beta[1])**2.)))**l)*((3. - ((2.*((X - output40.beta[4])**2. + (Y - output40.beta[3])**2.))/(output40.beta[1])**2))**2)*(np.exp((-2.)*(((X - output40.beta[4])**2. + (Y - output40.beta[3])**2.)/(output40.beta[1])**2.))) + output40.beta[2]
print 'done40'

myData41 = Data([Q, W], data41)
myModel = Model(function)
guesses41 = [25000, 130, 850, -25, 120] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr41 = ODR(myData41, myModel, guesses41, maxit=500)
odr41.set_job(fit_type=2)
output41 = odr41.run()
output41.pprint()
Fit_out41 = (((((output41.beta[0]))*(2.*((X - output41.beta[4])**2. + (Y - output41.beta[3])**2.)/(output41.beta[1])**2.)))**l)*((3. - ((2.*((X - output41.beta[4])**2. + (Y - output41.beta[3])**2.))/(output41.beta[1])**2))**2)*(np.exp((-2.)*(((X - output41.beta[4])**2. + (Y - output41.beta[3])**2.)/(output41.beta[1])**2.))) + output41.beta[2]
print 'done41'

myData42 = Data([Q, W], data42)
myModel = Model(function)
guesses42 = [25000, 140, 850, -20, -20] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr42 = ODR(myData42, myModel, guesses42, maxit=500)
odr42.set_job(fit_type=2)
output42 = odr42.run()
output42.pprint()
Fit_out42 = (((((output42.beta[0]))*(2.*((X - output42.beta[4])**2. + (Y - output42.beta[3])**2.)/(output42.beta[1])**2.)))**l)*((3. - ((2.*((X - output42.beta[4])**2. + (Y - output42.beta[3])**2.))/(output42.beta[1])**2))**2)*(np.exp((-2.)*(((X - output42.beta[4])**2. + (Y - output42.beta[3])**2.)/(output42.beta[1])**2.))) + output42.beta[2]
print 'done42'

myData43 = Data([Q, W], data43)
myModel = Model(function)
guesses43 = [25000, 140, 850, 0, 0] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr43 = ODR(myData43, myModel, guesses43, maxit=500)
odr43.set_job(fit_type=2)
output43 = odr43.run()
output43.pprint()
Fit_out43 = (((((output43.beta[0]))*(2.*((X - output43.beta[4])**2. + (Y - output43.beta[3])**2.)/(output43.beta[1])**2.)))**l)*((3. - ((2.*((X - output43.beta[4])**2. + (Y - output43.beta[3])**2.))/(output43.beta[1])**2))**2)*(np.exp((-2.)*(((X - output43.beta[4])**2. + (Y - output43.beta[3])**2.)/(output43.beta[1])**2.))) + output43.beta[2]
print 'done43'

myData44 = Data([Q, W], data44)
myModel = Model(function)
guesses44 = [25000, 140, 850, -25, 120] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr44 = ODR(myData44, myModel, guesses44, maxit=500)
odr44.set_job(fit_type=2)
output44 = odr44.run()
output44.pprint()
Fit_out44 = (((((output44.beta[0]))*(2.*((X - output44.beta[4])**2. + (Y - output44.beta[3])**2.)/(output44.beta[1])**2.)))**l)*((3. - ((2.*((X - output44.beta[4])**2. + (Y - output44.beta[3])**2.))/(output44.beta[1])**2))**2)*(np.exp((-2.)*(((X - output44.beta[4])**2. + (Y - output44.beta[3])**2.)/(output44.beta[1])**2.))) + output44.beta[2]
print 'done44'

myData45 = Data([Q, W], data45)
myModel = Model(function)
guesses45 = [25000, 140, 850, -25, 120] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr45 = ODR(myData45, myModel, guesses45, maxit=500)
odr45.set_job(fit_type=2)
output45 = odr45.run()
output45.pprint()
Fit_out45 = (((((output45.beta[0]))*(2.*((X - output45.beta[4])**2. + (Y - output45.beta[3])**2.)/(output45.beta[1])**2.)))**l)*((3. - ((2.*((X - output45.beta[4])**2. + (Y - output45.beta[3])**2.))/(output45.beta[1])**2))**2)*(np.exp((-2.)*(((X - output45.beta[4])**2. + (Y - output45.beta[3])**2.)/(output45.beta[1])**2.))) + output45.beta[2]
print 'done45'

myData46 = Data([Q, W], data46)
myModel = Model(function)
guesses46 = [25000, 150, 850, -25, 120] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr46 = ODR(myData46, myModel, guesses46, maxit=500)
odr46.set_job(fit_type=2)
output46 = odr46.run()
output46.pprint()
Fit_out46 = (((((output46.beta[0]))*(2.*((X - output46.beta[4])**2. + (Y - output46.beta[3])**2.)/(output46.beta[1])**2.)))**l)*((3. - ((2.*((X - output46.beta[4])**2. + (Y - output46.beta[3])**2.))/(output46.beta[1])**2))**2)*(np.exp((-2.)*(((X - output46.beta[4])**2. + (Y - output46.beta[3])**2.)/(output46.beta[1])**2.))) + output46.beta[2]
print 'done46'

myData47 = Data([Q, W], data47)
myModel = Model(function)
guesses47 = [25000, 200, 1000, -25, 120] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr47 = ODR(myData47, myModel, guesses47, maxit=500)
odr47.set_job(fit_type=2)
output47 = odr47.run()
output47.pprint()
Fit_out47 = (((((output47.beta[0]))*(2.*((X - output47.beta[4])**2. + (Y - output47.beta[3])**2.)/(output47.beta[1])**2.)))**l)*((3. - ((2.*((X - output47.beta[4])**2. + (Y - output47.beta[3])**2.))/(output47.beta[1])**2))**2)*(np.exp((-2.)*(((X - output47.beta[4])**2. + (Y - output47.beta[3])**2.)/(output47.beta[1])**2.))) + output47.beta[2]
print 'done47'

myData48 = Data([Q, W], data48)
myModel = Model(function)
guesses48 = [25000, 200, 1000, -25, 120] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr48 = ODR(myData48, myModel, guesses48, maxit=500)
odr48.set_job(fit_type=2)
output48 = odr48.run()
output48.pprint()
Fit_out48 = (((((output48.beta[0]))*(2.*((X - output48.beta[4])**2. + (Y - output48.beta[3])**2.)/(output48.beta[1])**2.)))**l)*((3. - ((2.*((X - output48.beta[4])**2. + (Y - output48.beta[3])**2.))/(output48.beta[1])**2))**2)*(np.exp((-2.)*(((X - output48.beta[4])**2. + (Y - output48.beta[3])**2.)/(output48.beta[1])**2.))) + output48.beta[2]
print 'done48'

myData49 = Data([Q, W], data49)
myModel = Model(function)
guesses49 = [25000, 200, 1000, -25, 120] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
odr49 = ODR(myData49, myModel, guesses49, maxit=500)
odr49.set_job(fit_type=2)
output49 = odr49.run()
output49.pprint()
Fit_out49 = (((((output49.beta[0]))*(2.*((X - output49.beta[4])**2. + (Y - output49.beta[3])**2.)/(output49.beta[1])**2.)))**l)*((3. - ((2.*((X - output49.beta[4])**2. + (Y - output49.beta[3])**2.))/(output49.beta[1])**2))**2)*(np.exp((-2.)*(((X - output49.beta[4])**2. + (Y - output49.beta[3])**2.)/(output49.beta[1])**2.))) + output49.beta[2]
print 'done49'

#myData50 = Data([Q, W], data50)
#myModel = Model(function)
#guesses50 = [25000, 120, 850, 0, 0] #guesses are [Scale, width, baseline, x0, y0] for some reason, the ODR program is most sensitive to x0 and y0 guesses. The scale and width values do not need to be especially close, but inaccurate x0 and y0 lead to excessive computation times
#odr50 = ODR(myData50, myModel, guesses50, maxit=500000)
#odr50.set_job(fit_type=2)
#output50 = odr50.run()
##output50.pprint()
#Fit_out50 = (((((output50.beta[0]))*(2.*((X - output50.beta[4])**2. + (Y - output50.beta[3])**2.)/(output50.beta[1])**2.)))**l)*((3. - ((2.*((X - output50.beta[4])**2. + (Y - output50.beta[3])**2.))/(output50.beta[1])**2))**2)*(np.exp((-2.)*(((X - output50.beta[4])**2. + (Y - output50.beta[3])**2.)/(output50.beta[1])**2.))) + output50.beta[2]
#print 'done50'

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
#maskedfit50 = np.where(data50 > zmin50, Fit_out50, 100)

print 'masking data done'

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
#Chisq50 = np.sum(np.sum((((maskeddata50 - maskedfit50))**2)/(maskedfit50)))


adj_chi1 = [Chisq26/N26, Chisq27/N27, Chisq28/N28, Chisq29/N29, Chisq30/N30, Chisq31/N31, Chisq32/N32, Chisq33/N33, Chisq34/N34, Chisq35/N35, Chisq36/N36, Chisq37/N37, Chisq38/N38, Chisq39/N39, Chisq40/N40, Chisq41/N41, Chisq42/N42, Chisq43/N43, Chisq44/N44, Chisq45/N45, Chisq46/N46, Chisq47/N47, Chisq48/N48, Chisq49/N49]#, Chisq50/N50]

print adj_chi1

fig1 = plt.figure()
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


plot1_1=fig1.add_subplot(121, projection='3d')
plot1_2=fig1.add_subplot(122, projection='3d')

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

plot1_1.plot_surface(Y, X, maskedfit49, rstride = 4, cstride = 4, linewidth = 0.05, cmap = 'cool')
plot1_2.plot_surface(Y, X, maskeddata49, rstride = 8, cstride = 8, cmap = 'hot', linewidth = 0.05) 

#plot2_1.plot_surface(Y, X, maskedfit42, rstride = 4, cstride = 4, linewidth = 0.05, cmap = 'cool')
#plot2_2.plot_surface(Y, X, maskeddata42, rstride = 8, cstride = 8, cmap = 'hot', linewidth = 0.05) 

#plot3_1.plot_surface(Y, X, maskedfit43, rstride = 4, cstride = 4, linewidth = 0.05, cmap = 'cool')
#plot3_2.plot_surface(Y, X, maskeddata43, rstride = 8, cstride = 8, cmap = 'hot', linewidth = 0.05) 

#plot4_1.plot_surface(Y, X, maskedfit44, rstride = 5, cstride = 5, linewidth = 0.05, cmap = 'cool')
#plot4_2.plot_surface(Y, X, maskeddata44, rstride = 8, cstride = 8, cmap = 'hot', linewidth = 0.05) 

#plot5_1.plot_surface(Y, X, maskedfit45, rstride = 5, cstride = 5, linewidth = 0.05, cmap = 'cool')
#plot5_2.plot_surface(Y, X, maskeddata45, rstride = 8, cstride = 8, cmap = 'hot', linewidth = 0.05) 

#plot6_1.plot_surface(Y, X, maskedfit46, rstride = 5, cstride = 5, linewidth = 0.05, cmap = 'cool')
#plot6_2.plot_surface(Y, X, maskeddata46, rstride = 8, cstride = 8, cmap = 'hot', linewidth = 0.05) 

#plot7_1.plot_surface(Y, X, maskedfit47, rstride = 8, cstride = 8, linewidth = 0.05, cmap = 'cool')
#plot7_2.plot_surface(Y, X, maskeddata47, rstride = 8, cstride = 8, cmap = 'hot', linewidth = 0.05) 

#plot8_1.plot_surface(Y, X, maskedfit48, rstride = 8, cstride = 8, linewidth = 0.05, cmap = 'cool')
#plot8_2.plot_surface(Y, X, maskeddata48, rstride = 8, cstride = 8, cmap = 'hot', linewidth = 0.05)

#plot9_1.plot_surface(Y, X, maskedfit49, rstride = 8, cstride = 8, linewidth = 0.05, cmap = 'cool')
#plot9_2.plot_surface(Y, X, maskeddata49, rstride = 8, cstride = 8, cmap = 'hot', linewidth = 0.05) 

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
odr51 = ODR(myData51, myModel51, guesses51, maxit=500000)
odr51.set_job(fit_type=2)
output51 = odr51.run()
output51.pprint()
Fit_out51 = output51.beta[2]*(xfit**2) + output51.beta[1]*xfit + output51.beta[0]

plot21.plot(xfit, Fit_out51)

#prints and labels all five parameters in the terminal, generates the plot in a new window.

plt.show()

print adj_chi1
