import scipy.optimize as optimize
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

A = np.array([(1,2,5.1), (10,40,89.4), (4,5,14.1)])

def func(data, a, b):
    return data[:,0]*a + data[:,1]*b

guess = (1,2)
params, pcov = optimize.curve_fit(func, A[:,:2], A[:,2], guess)
print(params)


