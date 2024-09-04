from configuration import *
import sys
import time
from tkinter import ttk
import tkinter as tk

from qualang_tools.addons.calibration.calibrations import QUA_calibrations

def print_progress_bar(iteration, total, length=40):
    progress = (iteration / total)
    arrow = '=' * int(round(progress * length))
    spaces = ' ' * (length - len(arrow))
    sys.stdout.write(f'\r[{arrow}{spaces}] {int(progress * 100)}%')
    sys.stdout.flush()


def run_progress_bar(total_time):
    steps = 100
    interval = total_time / steps
    for i in range(steps + 1):
        print_progress_bar(i, steps)
        time.sleep(interval)

    sys.stdout.write('\n')  # Move to the next line after completion


def estimate_time(n_avg, pi_pulses, saturation_pulses, measurements, with_progress_bar=False):
    pi_length = pi_pulses * pi_pulse_length/1e9
    saturation_length = saturation_pulses * saturation_len/1e9
    measurement_length = measurements * (thermalization_time // 4 + readout_len)/1e9
    total_time = n_avg * (pi_length + saturation_length + measurement_length)

    total_time = total_time

    if with_progress_bar:
        run_progress_bar(total_time)
    else:
        print(f"Total time: {total_time / 1e9:0.3f} s")
    return total_time


if __name__ == "__main__":
    n_avg = 1000
    pi_pulses = 2
    saturation_pulse = 1
    measurements = 2

    time_total = estimate_time(n_avg, pi_pulses, saturation_pulse, measurements, True)
