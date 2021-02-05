import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random

def F(x, y):
  return -(2*x**2 + y**2 + 3*y)+50



fig = plt.figure(figsize=(8, 6))
ax = fig.gca(projection = '3d')
jet = plt.get_cmap('jet')

x = np.linspace(-4, 4, 100)
y = np.linspace(-4, 4, 100)
X, Y = np.meshgrid(x, y)
Z = F(X, Y)
surf = ax.plot_surface(X, Y, Z, cmap = jet, linewidth = 0)
ax.set_zlim3d(0, Z.max())

ax.set_ylim([-5, 5])
ax.set_xlim([-5, 5])

# ax.set_xlabel('Parameter 1')
# ax.set_ylabel('Parameter 2')
# ax.set_zlabel('Objective')

plt.show()