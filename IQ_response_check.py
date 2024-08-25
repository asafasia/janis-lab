import numpy as np
from matplotlib import pyplot as plt
from qm import QuantumMachinesManager
from qm.qua import *
from configuration import *
from time import sleep
from instruments_py27.spectrum_analyzer import N9010A_SA

I0 = config["controllers"]["con1"]["analog_outputs"][1]["offset"]
Q0 = config["controllers"]["con1"]["analog_outputs"][2]["offset"]

sa = N9010A_SA(sa_address, False)
sa.setup_spectrum_analyzer(center_freq=6e3, span=5e6, BW=0.2e6, points=35)
sa.set_marker_max()

qop_ip = None
qmm = QuantumMachinesManager("192.168.43.137", 9510)

qm = qmm.open_qm(config)

with program() as prog:
    with infinite_loop_():
        play("readout", "resonator")

job = qm.execute(prog)


def getWithIQ(IQ, qm, sa, averaging=False, verbose=False):
    """Sets DAC output to I=IQ[0] and Q=IQ[1] and measures with spectrum analyzer"""

    qm.set_output_dc_offset_by_element("resonator", "I", float(IQ[0]))
    qm.set_output_dc_offset_by_element("resonator", "Q", float(IQ[1]))
    sleep(0.2)

    sa.set_marker_max()
    return sa.get_marker()


num_points = 51

amp = 0.2  # I,Q amplitude
thetas = np.linspace(0, 2 * np.pi, num_points)
power_vec = []
for theta in thetas:
    I = np.cos(theta) * amp + I0
    Q = np.sin(theta) * amp + Q0
    power = getWithIQ([I, Q], qm, sa)
    power_vec.append(power)
    print(f"Theta: {theta / np.pi / 2 * 360}, Power: {power}")
    sleep(0.1)

volts = np.sqrt(10 ** (np.array(power_vec) / 10.0) * 50)

plt.polar(thetas, volts)
plt.show()

