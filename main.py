import numpy as np
from qutip import *


pnt = [1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)]
b = Bloch()

b.add_points(pnt)

b.show()
