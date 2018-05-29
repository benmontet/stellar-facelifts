import numpy as np

a = np.loadtxt('cumulative.csv', skiprows=58, usecols=[-3, -2, -1], delimiter=',')

np.savetxt('ra_dec.txt', a, fmt='%.5f %.6f %.3f', delimiter=',')