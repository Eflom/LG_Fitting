import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import Axes3D
import scipy
import scipy.optimize as optimize
import pylab


q = np.arange(1,101,1)
w = np.arange(1,101,1)

err = pylab.random(100)*5

x= q + err
y= w + err
z= 3*q + 3*w

AA = np.vstack((x,y))
BB = np.transpose(AA)

A = zip(x,y)
B = np.vstack(A)

fig1=plt.figure()
plot1 = fig1.add_subplot(121, projection='3d')
plot2 = fig1.add_subplot(122, projection='3d')
X, Y = np.meshgrid(x, y, sparse = True)

Z = 2*X + 3*Y

plot1.plot_surface(X+err ,Y+err, Z,rstride=2,cstride=2,cmap = 'cool', linewidth = 0.05)

R = [x,y,z]
RR = np.transpose(R)
RRR = np.transpose([q,w,z])

def func(data, a, b):
	return data[:,0]*a + data[:,1]*b
			
guess = (2.00, 3.00)
params, pcov = optimize.curve_fit(func, RRR[:,:2], RRR[:,2], guess)

print(params)

plot2.plot_surface(X, Y, (params[0]*X + params[1]*Y), rstride=2, cstride=2, cmap = 'hot', linewidth = 0.05)


plt.show()
