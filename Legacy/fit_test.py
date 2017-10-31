import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import pandas as pd
from matplotlib import cm
import scipy
from scipy.odr import ODR, Model, Data, RealData
from scipy import special
import scipy.optimize as optimize


rowstart = 125
rowend = 325
columnstart = 775
columnend = 975


data = pd.read_table('/home/flom/Desktop/REU/Data/April_26.asc', skiprows = rowstart - 1, nrows = rowend - rowstart, usecols = range(columnstart,columnend,1))
#print data 


#nx, ny = columnnum, rownum
y = range(rowstart, rowend, 1)
x = range(columnstart, columnend, 1)
hf = plt.figure()
ha = hf.add_subplot(111, projection='3d')
X, Y = np.meshgrid(x, y)  

ha.plot_surface(X,Y,data,rstride=2,cstride=2, cmap = 'cool', linewidth = 0.05, antialiased = True)
plt.title('LG 2-1 beam @ 80mm')

print X
#########################################################################


x0 = (columnstart + columnend)/2
y0 = (rowstart + rowend)/2

l = 2 ## azimuthal index
p = 1 ## radial index
P = 70 
w = 1 ## waist

r = ((X - x0)**2 + (Y-y0)**2)**0.5
theta = scipy.arctan(Y/X)


#def func(data, A, x0, y0, w, X, Y):
	#return A((-1.0)**p)*exp(-1j*l*theta)*exp((-r**2.0)/(w**2.0))*(((2.0**0.5)*r/w)**l)*(scipy.special.genlaguerre((2.0*r**2)/(w**2)))
def func(r, theta):
	return r*theta

guess = (1, 1)
data = RealData(X, Y, data)
fit = Model(func)
odr = ODR(data, fit [10])



plt.show()

