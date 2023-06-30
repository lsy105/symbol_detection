import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
x = np.asarray([0, 11, 25, 39, 53, 63])
y = x**2
p = np.arange(4)
p_avg = np.sum(p)
p_6 = np.insert(p,[0,4],p_avg)


f = interpolate.interp1d(x, p_6)

xnew = np.arange(0, 64, 1)
ynew = f(xnew)   # use interpolation function returned by `interp1d`
plt.plot(x, y, 'o', xnew, ynew, '-')
plt.show()
