import numpy as np
import pylab
import matplotlib.pyplot as plt
import scipy
from scipy import special
from mpl_toolkits.mplot3d.axes3d import Axes3D
import cmath
import pandas as pd
import scipy.optimize as opt
import scipy.odr
from scipy.odr import ODR, Model, Data, RealData


xmin = 100 
xmax = 1000 
ymin = 100
ymax = 1000 
x0 = (xmin + xmax)/2
y0 = (ymin + ymax)/2

data1_1 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/25_0001.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data1_2 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/25_0002.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data1_3 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/25_0003.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data1_4 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/25_0004.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data1_5 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/25_0005.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data1_6 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/25_0006.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data1_7 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/25_0007.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data1_8 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/25_0008.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data1_9 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/25_0009.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))
data1_10 = np.array(pd.read_table('/home/flom/Desktop/REU/Data/August_5_2-1/25_0010.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin, ymax, 1), dtype = np.float64))

data = (data1_1 + data1_2 + data1_3 + data1_4 + data1_5 + data1_6 + data1_7 + data1_8 + data1_9 + data1_10)/10.

x = np.arange(xmin, xmax, 1)
y = np.arange(ymin, ymax, 1)
z = np.ndarray.flatten(np.array(data))

p=1.
l=2.

X, Y = np.meshgrid(x - x0, y -y0)

Q = x - x0
W = y - y0

A = 100.
w = 30.
B = 0.

r = (X**2 + Y**2)**0.5
theta = np.arctan(Y/X)



Z1 = ((2.*(r**2)/w**2))**l
Z2 = (3. - ((2.*r**2)/w**2.))**2.  ##use Mathematica LaguerreL [bottom, top, function] to generate this term
Z3 = np.exp((-(2.))*(r**2/w**2.))
Z4 = A*Z1*Z2*Z3 + B

def function(params, data):
	Scale = params[0] 
	Baseline = params[2] 
	width = params[1]
	x_0 = params[3]
	y_0 = params[4]
		
	return ((((params[0]))*(2.*((X - params[3])**2. + (Y - params[4])**2.)/(params[1])**2.))**l)*((3. - ((2.*((X - params[3])**2. + (Y - params[4])**2.))/(params[1])**2.))**2.)*(np.exp((-2.)*(((X - params[3])**2. + (Y - params[4])**2.)/(params[1])**2.))) + params[2]

myData = Data([Q, W], data)
myModel = Model(function)
guesses = [150, 80, 800, 200, 200]	

odr = scipy.odr.ODR(myData, myModel, guesses, maxit=1000)

odr.set_job(fit_type=2)

output = odr.run()

output.pprint()

Fit_out = (output.beta[0])*(2.*((X - output.beta[3])**2. + (Y - output.beta[4])**2.)/((output.beta[1])**2.))**l*((3. - ((2.*((X - output.beta[3])**2. + (Y - output.beta[4])**2.))/(output.beta[1])**2.))**2.)*(np.exp((-2.)*(((X - output.beta[3])**2. + (Y - output.beta[4])**2.)/(output.beta[1])**2.))) + (output.beta[2])

fig1 = plt.figure()

plot1=fig1.add_subplot(121, projection='3d')
plot2=fig1.add_subplot(122, projection='3d')
#plot3=fig1.add_subplot(223, projection='3d')
#plot4=fig1.add_subplot(224, projection='3d')

#plot1.plot_surface(X, Y, Z1, rstride = 5, cstride = 5, cmap = 'cool', linewidth = 0.05)
#plot2.plot_surface(X, Y, Z2, rstride = 5, cstride = 5, cmap = 'cool', linewidth = 0.05)
#plot3.plot_surface(X, Y, Z3, rstride = 5, cstride = 5, cmap = 'cool', linewidth = 0.05)
#plot4.plot_surface(X, Y, Z4, rstride = 5, cstride = 5, cmap = 'cool', linewidth = 0.05)

Crop_range = 0.2


zmin = np.max(np.max(data))*Crop_range
maskeddata = np.where(data > zmin, data, 100)
maskedfit = np.where(data > zmin, Fit_out, 100)


plot1.plot_surface(X, Y, maskedfit, rstride = 8, cstride = 8, linewidth = 0.05, cmap = 'cool')
plot2.plot_surface(X, Y, maskeddata,rstride=3,cstride=3, cmap = 'cool', linewidth = 0.1)
#plot3.plot_surface(X + xerr, Y + yerr, Z4, rstride = 1, cstride = 1, cmap = 'cool', linewidth = 0.05)

##calculate Chi Squared values




Chisq = (np.sum(np.sum(((maskeddata - maskedfit)**2)/maskedfit)))

print 'Chi Sq / N'
print Chisq

plt.show()

####Library of Laguerre Polynomials for substitution in Z2

## 1, 0       1

## 1, 1       (2 - ((2*(X**2 + Y**2))/w**2))**2

## 2, 1       (3 - ((2*(X**2 + Y**2))/w**2))**2

## 5, 0       1

## 





