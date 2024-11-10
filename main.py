import numpy as np

p = np.array([[49.1, 50.9], [26.1, 73.9]]) / 100

p_inv = np.linalg.inv(p)

# print(p_inv)

v = np.array([0.51, 0.49])

print(p_inv@v)

print(v.T@p_inv)
