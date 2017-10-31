import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import pandas as pd
from matplotlib import cm
import scipy
import numpy.ma as ma


xmin = 200 #xmin
xmax = 800 #xmax
ymin = 400 #ymin
ymax = 1000 #ymax


data1 = pd.read_table('/home/flom/Desktop/REU/Data/June_28/1_0/Post_2.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin,ymax,1))
data2 = pd.read_table('/home/flom/Desktop/REU/Data/June_28/1_0/Post_8.asc', skiprows = xmin - 1, nrows = xmax - xmin, usecols = range(ymin,ymax,1))
#data[1:40] = np.nan

#Z = np.array(data)

#zmin = np.max(np.max(data))*0.10
#maskz = np.where(data>2000, data, np.nan)

y = range(xmin, xmax, 1)
x = range(ymin, ymax, 1)
hf = plt.figure()
ha = hf.add_subplot(111, projection='3d')
xx = np.ma.array(x)
yy = np.ma.array(y)
X, Y = np.meshgrid(xx, yy)  


#ha.plot_surface(Y,X,data1,rstride=3,cstride=3, cmap = 'cool', linewidth = 0.05)
ha.plot_surface(Y,X,data2, rstride=2, cstride=2, cmap = 'cool', linewidth = 0.05)
plt.title('LG 1-0 beam @ 800mm')

#ha.scatter(X, Y, maskz, s=10)

plt.show()
