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


xmin = 300
xmax = 850 
ymin = 500
ymax = 1050 
Num_obs = 49

x0 = (xmin + xmax)/2
y0 = (ymin + ymax)/2
ranges_inch = np.arange(0,Num_obs,1)
ranges = (ranges_inch)*25.4 + 5



for a in range(0, Num_obs):
	for b in range(0001,0010):
	data%d_%d = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/%d_%d.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64)) % (a, b, a, b)
