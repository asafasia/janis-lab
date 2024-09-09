import numpy as np
from qutip import *
from configuration import *
import experiments_objects.qubit_spectroscopy as qubit_spectroscopy
from datetime import datetime, timedelta

n_avg = 10000
sweep_points_1 = 200
sweep_points_2 = 100

time_ns = n_avg * sweep_points_1 * sweep_points_2 * thermalization_time * 1.1
tim_sec = time_ns * 1e-9
time_min = tim_sec / 60
time_hr = time_min / 60
time_days = time_hr / 24

print(f"time in nano seconds ~ {time_ns:.1e}")
print(f"time in seconds ~ {tim_sec:.0f} s")
print(f"time in minutes ~ {time_min:.0f} min")
print(f"time in hours ~ {time_hr:.2f} hr")
print(f"time in days ~ {time_days:.1f} days")


current_time = datetime.now()
time_interval = timedelta(hours=time_hr)
future_time = current_time + time_interval

print("##############################################")
print("##############################################")
print("Current time:", current_time.strftime("%Y-%m-%d %H:%M:%S"))
print("Time of finish:", future_time.strftime("%Y-%m-%d %H:%M:%S"))