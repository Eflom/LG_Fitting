import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import pandas as pd
from matplotlib import cm
import scipy

rowstart1 = 150
rowend1 = 450
columnstart1 = 200
columnend1 = 400

rowstart2 = 200
rowend2 = 450
columnstart2 = 750
columnend2 = 1000

rowstart3 = 200
rowend3 = 450
columnstart3 = 150
columnend3 = 400

data1 = pd.read_table('/home/flom/Desktop/REU/Data/May_5/20mm.asc', skiprows = rowstart1 - 1, nrows = rowend1 - rowstart1, usecols = range(columnstart1,columnend1,1))
data2 = pd.read_table('/home/flom/Desktop/REU/Data/May_5/75mm.asc', skiprows = rowstart2 - 1, nrows = rowend2 - rowstart2, usecols = range(columnstart2,columnend2,1))
data3 = pd.read_table('/home/flom/Desktop/REU/Data/May_5/120mm.asc', skiprows = rowstart3 - 1, nrows = rowend3 - rowstart3, usecols = range(columnstart3,columnend3,1))




y1 = range(rowstart1, rowend1, 1)
x1 = range(columnstart1, columnend1, 1)

y2 = range(rowstart2, rowend2, 1)
x2 = range(columnstart2, columnend2, 1)

y3 = range(rowstart3, rowend3, 1)
x3 = range(columnstart3, columnend3, 1)


plot = plt.figure()
#fig1 = plot.add_subplot(131, projection='3d')
fig2 = plot.add_subplot(111, projection='3d')
#fig3 = plot.add_subplot(133, projection='3d')

#X1, Y1 = np.meshgrid(x1,y1)  
X2, Y2 = np.meshgrid(x2,y2)
#X3, Y3 = np.meshgrid(x3,y3)


#fig1.plot_surface(X1,Y1,data1,rstride=1,cstride=1, cmap = 'cool', linewidth = 0.05, antialiased = True)
fig2.plot_surface(X2,Y2,data2,rstride=2,cstride=2, cmap = 'cool', linewidth = 0.05, antialiased = True)
#fig3.plot_surface(X3,Y3,data3,rstride=1,cstride=1, cmap = 'cool', linewidth = 0.05, antialiased = True)

#fig1.set_title('20mm')
fig2.set_title('LG2-1 @ 75mm')
#fig3.set_title('120mm')

plt.show()
