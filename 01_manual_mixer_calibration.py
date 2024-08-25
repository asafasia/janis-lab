"""
        MIXER CALIBRATION
The program is designed to play a continuous single tone to calibrate an IQ mixer. To do this, connect the mixer's
output to a spectrum analyzer. Adjustments for the DC offsets, gain, and phase must be made manually.

If you have access to the API for retrieving data from the spectrum analyzer, you can utilize the commented lines below
to semi-automate the process.

Before proceeding to the next node, take the following steps:
    - Update the DC offsets in the configuration at: config/controllers/"con1"/analog_outputs.
    - Modify the DC gain and phase for the IQ signals in the configuration, under either:
      mixer_qubit_g & mixer_qubit_g or mixer_resonator_g & mixer_resonator_g.
"""

from qm import QuantumMachinesManager
from qm.qua import *
from configuration import *
import numpy as np
import matplotlib.pyplot as plt
from time import sleep

from instruments_py27.spectrum_analyzer import N9010A_SA

###################
# The QUA program #
###################
element = "resonator"

with program() as cw_output:
    with infinite_loop_():
        # It is best to calibrate LO leakage first and without any power played (cf. note below)
        play("cw" * amp(0), element)

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qm_host, port=qm_port)
qm = qmm.open_qm(config)

job = qm.execute(cw_output)

# When done, the halt command can be called and the offsets can be written directly into the config file.

# job.halt()

# These are the 2 commands used to correct for mixer imperfections. The first is used to set the DC of the `I` and `Q`
# channels to compensate for the LO leakage. The 2nd command is used to correct for the phase and amplitude mismatches
# between the channels.
# The output of the IQ Mixer should be connected to a spectrum analyzer and values should be chosen as to minimize the
# unwanted peaks.
# If python can read the output of the spectrum analyzer, then this process can be automated and the correct values can
# be found using an optimization method such as Nelder-Mead:
# https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html

# qm.set_output_dc_offset_by_element('qubit', ('I', 'Q'), (-0.001, 0.003))
# qm.set_mixer_correction('mixer_qubit', int(qubit_IF), int(qubit_LO), IQ_imbalance(0.015, 0.01))

# Note that the LO leakage (DC Offset) depends on the 'I' & 'Q' powers, it is advised to run this step with no input power.
# This will ensure that there is no LO leakage while the pulses are not played in the case where the is no switch.
# This can be achieved by changing the line above to `play("cw" * amp(0), "qubit")`

# Automatic LO leakage correction

sa = N9010A_SA(sa_address, False)
sa.setup_spectrum_analyzer(center_freq=6e9 / 1e6, span=0.5e6, BW=0.1e6, points=15)
sa.set_marker_max()

centers = [0, 0]
span = 0.1
#
fig1 = plt.figure()
for n in range(3):
    offset_i = np.linspace(centers[0] - span, centers[0] + span, 11)
    offset_q = np.linspace(centers[1] - span, centers[1] + span, 11)
    lo_leakage = np.zeros((len(offset_q), len(offset_i)))
    for i in range(len(offset_i)):
        for q in range(len(offset_q)):
            print(i,q)
            qm.set_output_dc_offset_by_element(element, ("I", "Q"), (offset_i[i], offset_q[q]))
            sleep(0.01)
            # Write functions to extract the lo leakage from the spectrum analyzer
            lo_leakage[q][i] = sa.get_marker()

#
    minimum = np.argwhere(lo_leakage == np.min(lo_leakage))[0]

    print(minimum)
    centers = [offset_i[minimum[0]], offset_q[minimum[1]]]
    print(centers)
    span = span / 10
    plt.subplot(131)
    plt.pcolor(offset_i, offset_q, lo_leakage.transpose())
    plt.xlabel("I offset [V]")
    plt.ylabel("Q offset [V]")
    plt.title(f"Minimum at (I={centers[0]:.3f}, Q={centers[1]:.3f}) = {lo_leakage[minimum[0]][minimum[1]]:.1f} dBm")
plt.suptitle(f"LO leakage correction for {element}")



print(f"For {element}, I offset is {centers[0]} and Q offset is {centers[1]}")

# Automatic image cancellation
centers = [0, 0]
span = [0.2, 0.5]

fig2 = plt.figure()
for n in range(3):
    gain = np.linspace(centers[0] - span[0], centers[0] + span[0], 21)
    phase = np.linspace(centers[1] - span[1], centers[1] + span[1], 21)
    image = np.zeros((len(phase), len(gain)))
    for g in range(len(gain)):
        for p in range(len(phase)):
            qm.set_mixer_correction(
                config["elements"][element]["mixInputs"]["mixer"],
                int(config["elements"][element]["intermediate_frequency"]),
                int(config["elements"][element]["mixInputs"]["lo_frequency"]),
                IQ_imbalance(gain[g], phase[p]),
            )
            sleep(0.01)
            # Write functions to extract the image from the spectrum analyzer
            # image[q][i] =
    minimum = np.argwhere(image == np.min(image))[0]
    centers = [gain[minimum[0]], phase[minimum[1]]]
    span = (np.array(span) / 10).tolist()
    plt.subplot(132)
    plt.pcolor(gain, phase, image.transpose())
    plt.xlabel("Gain")
    plt.ylabel("Phase imbalance [rad]")
    plt.title(f"Minimum at (I={centers[0]:.3f}, Q={centers[1]:.3f}) = {image[minimum[0]][minimum[1]]:.1f} dBm")
plt.suptitle(f"Image cancellation for {element}")

print(f"For {element}, gain is {centers[0]} and phase is {centers[1]}")
plt.show()