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

xmin = 100
xmax = 950 
ymin = 100
ymax = 1100 
x0 = (xmin + xmax)/2
y0 = (ymin + ymax)/2
ranges = [15, 50, 100, 200, 400, 600, 800, 1000]

data1_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/24_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data1_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/24_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data1_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/24_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data1_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/24_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data1_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/24_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data1_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/24_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data1_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/24_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data1_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Code/July_14/24_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

x = np.arange(xmin, xmax, 1)
y = np.arange(ymin, ymax, 1)

X, Y = np.meshgrid(x - x0, y -y0)

q = 428
w = 250
z = 1000

xcross1 = data1_1[q, w:z]
xcross2 = data1_2[q, w:z]
xcross3 = data1_3[q, w:z]
xcross4 = data1_4[q, w:z]
xcross5 = data1_5[q, w:z]
xcross6 = data1_6[q, w:z]
xcross7 = data1_7[q, w:z]
xcross8 = data1_8[q, w:z]


y = np.arange(w, z, 1)

fig1 = plt.figure()
plot1 = fig1.add_subplot(111)
plot1.scatter(y, xcross1)
plot1.scatter(y, xcross2)
plot1.scatter(y, xcross3)
plot1.scatter(y, xcross4)
plot1.scatter(y, xcross5)
plot1.scatter(y, xcross6)
plot1.scatter(y, xcross7)
plot1.scatter(y, xcross8)



plt.show()
