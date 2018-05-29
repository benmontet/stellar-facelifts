import numpy as np

a = np.loadtxt('k2candidates.csv', skiprows=34, usecols=[11, 13, -2], delimiter=',')

np.savetxt('k2_ra_dec.txt', a, fmt='%.5f %.6f %.3f', delimiter=',')