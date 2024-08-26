import numpy as np
from matplotlib import pyplot as plt

f = 1

a = 1.1
b = 0.1

t = np.arange(0, 2, 0.01)

x = np.cos(2 * np.pi * f * t)
y = a * np.sin(2 * np.pi * f * t + b)

plt.subplot(1, 2, 1)
plt.axis("equal")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x, y)

plt.subplot(1, 2, 2)
plt.plot(x + y, x - y)
plt.show()
