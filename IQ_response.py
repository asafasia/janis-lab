from matplotlib import pyplot as plt
from time import sleep

from change_args import modify_json
from configuration import *
from instruments_py27.spectrum_analyzer import N9010A_SA
from qm import QuantumMachinesManager
from qm.qua import *
import numpy as np

lo_freq = 6e3
I_port = 1
Q_port = 2
element = "resonator"

print(config["controllers"]["con1"])
I0 = config["controllers"]["con1"]["analog_outputs"]["1"]["offset"]
Q0 = config["controllers"]["con1"]["analog_outputs"]["2"]["offset"]


def open_qm():
    qm = QuantumMachinesManager(qm_host, 9510)
    config = {
        "version": 1,
        "controllers": {
            "con1": {
                "type": "opx1",
                "analog_outputs": {
                    I_port: {"offset": I0},
                    Q_port: {"offset": Q0}
                }
            }
        },
        "elements": {
            "RR1": {
                "singleInput": {
                    "port": ("con1", I_port)
                },
                "intermediate_frequency": 0.0,
                "operations": {
                    "pulse": "my_pulse"
                }
            },
            "RR2": {
                "singleInput": {
                    "port": ("con1", Q_port)
                },
                "intermediate_frequency": 0.0,
                "operations": {
                    "pulse": "my_pulse"
                }
            }
        },
        "pulses": {
            "my_pulse": {
                "operation": "control",
                "length": 2000,
                "waveforms": {
                    "single": "zero_wave"
                }
            }
        },
        "waveforms": {
            "zero_wave": {
                "type": "constant",
                "sample": 0.0
            }
        }
    }
    return qm.open_qm(config)


def plot_ellipse(plt, theta, volt, title, figs=[None, None]):
    plt.figure(figs[0])
    # plt.plot(volt * np.cos(theta), volt * np.sin(theta))
    plt.polar(theta, volt)
    plt.polar(theta + np.pi, volt)

    # plt.xlabel('I')
    # plt.ylabel('Q')
    # plt.axis('square')
    plt.title(title)

    plt.figure(figs[1])
    plt.plot(theta / np.pi / 2, volt)
    plt.xlabel("$\Theta/2\pi$")
    plt.ylabel("Voltage ($\sqrt{10^{P/10}\cdot 50}$)")
    plt.title(title)


def getWithIQ(IQ, qm, sa, averaging=False, verbose=False):
    """Sets DAC output to I=IQ[0] and Q=IQ[1] and measures with spectrum analyzer"""
    if verbose:
        print("Setting I=%f, Q=%f" % (IQ[0], IQ[1]))
    qm.set_output_dc_offset_by_element("RR1", "single", float(IQ[0]))
    qm.set_output_dc_offset_by_element("RR2", "single", float(IQ[1]))
    if averaging:
        sa.restart_averaging()
        sleep(1.0)
    else:
        sleep(0.2)
    sa.set_marker_max()
    t = sa.get_marker()

    if verbose:
        print("Transmitted power is %f dBm" % t)
    return t


averaging = False

num_points = 51  # angular points to test response
amp = 0.1  # I,Q amplitude

sa = N9010A_SA(sa_address, False)
sa.setup_spectrum_analyzer(center_freq=6e3, span=0.5e6, BW=0.02e6, points=125)
sa.set_marker_max()
if averaging:
    sa.setup_averaging(True, 4)
else:
    sa.setup_averaging(False)

qm = open_qm()

with program() as prog:
    with infinite_loop_():
        play("pulse", "RR1")
        play("pulse", "RR2")

job = qm.execute(prog)

theta = np.linspace(0, 2 * np.pi, num_points)
power = np.zeros(theta.shape)
I = amp * np.cos(theta)
Q = amp * np.sin(theta)

print("Getting response...")
getWithIQ([I0, Q0], qm, sa)  # to prevent problems
for idx in range(len(theta)):
    print("idx = %d" % idx + " of %d" % len(theta))
    iq = [I[idx] + I0, Q[idx] + Q0]
    # iq = [I[idx], Q[idx]]
    power[idx] = getWithIQ(iq, qm, sa, averaging=averaging)
    # plt.figure(figs[0])
    # plt.plot(volt * np.cos(theta), volt * np.sin(theta))

volt = np.sqrt(10 ** (power / 10.0) * 50)
plot_ellipse(plt, theta, volt, "Uncalibrated", [1, 2])
plt.show()

# %% calibrate angle - set maximal voltage to theta=0

theta0 = theta[volt.argmax()]
c = np.cos(theta0)
s = np.sin(theta0)
rot = np.array([[c, -s], [s, c]])  # rotation matrix

print("Getting response with angular correction...")
getWithIQ([I0, Q0], qm, sa)  # to prevent problems
for idx in range(len(theta)):
    print("idx = %d" % idx + " of %d" % len(theta))
    iq = rot @ [I[idx], Q[idx]]
    power[idx] = getWithIQ(iq + [I0, Q0], qm, sa, averaging=averaging)

volt = np.sqrt(10 ** (power / 10.0) * 50)

# plot
plot_ellipse(plt, theta, volt, "Angular correction", [3, 4])

# find long radius - volt at 0,pi (actually should be equal if symmetric around origin)
idx_pi = np.abs(theta - np.pi).argmin()
r_long = volt[idx_pi] + volt[0]

# find short radius - volt at +/- pi/2 (actually should be equal if symmetric around origin)
idx_pi_2 = np.abs(theta - np.pi / 2).argmin()
idx_3_pi_2 = np.abs(theta - 3 * np.pi / 2).argmin()
r_short = volt[idx_3_pi_2] + volt[idx_pi_2]
plt.figure(3)
plt.polar(0, r_long / 2, 'ro')
plt.polar(np.pi / 2, r_short / 2, 'ro')
plt.polar(np.pi, r_long / 2, 'ro')
plt.polar(3 * np.pi / 2, r_short / 2, 'ro')
plt.show()

# %% calibrate scaling

scaling_m = np.array([[1, 0.0], [0.0, r_long / r_short]])

print("Getting response with all corrections...")
getWithIQ([I0, Q0], qm, sa)  # to prevent problems
for idx in range(len(theta)):
    print("idx = %d" % idx + " of %d" % len(theta))
    iq = rot @ scaling_m @ [I[idx], Q[idx]]
    power[idx] = getWithIQ(iq + [I0, Q0], qm, sa, averaging=averaging)

volt = np.sqrt(10 ** (power / 10.0) * 50)
# plot
plot_ellipse(plt, theta, volt, "Corrected", [5, 6])

print(np.array(rot @ scaling_m).flatten())
plt.show()

# %% test model


theta0 = theta[volt.argmax()]
c = np.cos(theta0)
s = np.sin(theta0)
rot2 = np.array([[c, -s], [s, c]])  # rotation matrix

print("Getting response with angular correction...")
getWithIQ([I0, Q0], qm, sa)  # to prevent problems
for idx in range(len(theta)):
    print("idx = %d" % idx + " of %d" % len(theta))
    iq = rot @ scaling_m @ rot2 @ [I[idx], Q[idx]]
    power[idx] = getWithIQ(iq + [I0, Q0], qm, sa, averaging=averaging)

volt = np.sqrt(10 ** (power / 10.0) * 50)

# plot
plot_ellipse(plt, theta, volt, "Angular correction", [7, 8])

# find long radius - volt at 0,pi (actually should be equal if symmetric around origin)
idx_pi = np.abs(theta - np.pi).argmin()
r_long = volt[idx_pi] + volt[0]

# find short radius - volt at +/- pi/2 (actually should be equal if symmetric around origin)
idx_pi_2 = np.abs(theta - np.pi / 2).argmin()
idx_3_pi_2 = np.abs(theta - 3 * np.pi / 2).argmin()
r_short = volt[idx_3_pi_2] + volt[idx_pi_2]
plt.figure(7)
plt.polar(0, r_long / 2, 'ro')
plt.polar(np.pi / 2, r_short / 2, 'ro')
plt.polar(np.pi, r_long / 2, 'ro')
plt.polar(3 * np.pi / 2, r_short / 2, 'ro')
plt.show()

# %% calibrate scaling

scaling_m2 = np.array([[1, 0.0], [0.0, r_long / r_short]])

print("Getting response with all corrections...")
getWithIQ([I0, Q0], qm, sa)  # to prevent problems
for idx in range(len(theta)):
    print("idx = %d" % idx + " of %d" % len(theta))
    iq = rot @ scaling_m @ rot2 @ scaling_m2 @ [I[idx], Q[idx]]
    power[idx] = getWithIQ(iq + [I0, Q0], qm, sa, averaging=averaging)

volt = np.sqrt(10 ** (power / 10.0) * 50)
# plot
plot_ellipse(plt, theta, volt, "Corrected", [9, 10])

correction_matrix = np.array(rot @ scaling_m @ rot2 @ scaling_m2).flatten()
correction_matrix = list(correction_matrix)

modify_json(qubit, element, "resonator_correction_matrix", correction_matrix)

plt.show()
