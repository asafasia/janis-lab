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

from change_args import modify_json
from configuration import *
import numpy as np
import matplotlib.pyplot as plt
from time import sleep

from instruments_py27.spectrum_analyzer import N9010A_SA

###################
# The QUA program #
###################
element = "resonator"

if element != "resonator" and element != "qubit":
    raise ValueError("Element must be either 'resonator' or 'qubit'")

with program() as cw_output:
    with infinite_loop_():
        # It is best to calibrate LO leakage first and without any power played (cf. note below)
        play("cw" * amp(1), element)

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qm_host, port=qm_port)
qm = qmm.open_qm(config)

job = qm.execute(cw_output)

sa = N9010A_SA(sa_address, False)

f_LO = args['qubit1'][element][f'{element}_LO']
f_IF = args['qubit1'][element][f'{element}_IF']

sa.setup_spectrum_analyzer(center_freq=f_LO / 1e6 + f_IF / 1e6, span=0.5e6, BW=0.1e6, points=15)
sa.set_marker_max()
sa.setup_averaging(False, 1)

# I0 = resonator_args["IQ_input"]["I"]
# Q0 = resonator_args["IQ_input"]["Q"]
#
# centers = [0, 0]
# span = 0.1
# #
# fig1 = plt.figure()
# for n in range(3):
#     print('n = ', n)
#     offset_i = np.linspace(centers[0] - span, centers[0] + span, 11)
#     offset_q = np.linspace(centers[1] - span, centers[1] + span, 11)
#     lo_leakage = np.zeros((len(offset_q), len(offset_i)))
#     for i in range(len(offset_i)):
#         print('i = ', i)
#         for q in range(len(offset_q)):
#             qm.set_output_dc_offset_by_element(element, ("I", "Q"), (offset_i[i], offset_q[q]))
#             sleep(0.05)
#             # Write functions to extract the lo leakage from the spectrum analyzer
#             lo_leakage[q][i] = sa.get_marker()
#
#     #
#     minimum = np.argwhere(lo_leakage == np.min(lo_leakage))[0]
#
#     print(minimum)
#     centers = [offset_i[minimum[0]], offset_q[minimum[1]]]
#     print(centers)
#     span = span / 5
#     plt.pcolor(offset_i, offset_q, lo_leakage.transpose())
#     plt.xlabel("I offset [V]")
#     plt.ylabel("Q offset [V]")
#     plt.title(f"Minimum at (I={centers[0]:.3f}, Q={centers[1]:.3f}) = {lo_leakage[minimum[0]][minimum[1]]:.1f} dBm")
#     plt.colorbar()
#     plt.show()
# # plt.suptitle(f"LO leakage correction for {element}")
#
# print(f"For {element}, I offset is {centers[0]} and Q offset is {centers[1]}")

# Automatic image cancellation
centers = [0, 0]

span = [0.1, 0.1]
num = 11
fig2 = plt.figure()
for n in range(3):
    gain = np.linspace(centers[0] - span[0], centers[0] + span[0], num)
    phase = np.linspace(centers[1] - span[1], centers[1] + span[1], num)
    image = np.zeros((len(phase), len(gain)))
    for g in range(len(gain)):
        print('g = ', g)
        for p in range(len(phase)):
            qm.set_mixer_correction(
                config["elements"][element]["mixInputs"]["mixer"],
                int(config["elements"][element]["intermediate_frequency"]),
                int(config["elements"][element]["mixInputs"]["lo_frequency"]),
                IQ_imbalance(gain[g], phase[p]),
            )
            sleep(0.1)
            # Write functions to extract the image from the spectrum analyzer
            image[g][p] = sa.get_marker()
    minimum = np.argwhere(image == np.min(image))[0]
    centers = [gain[minimum[0]], phase[minimum[1]]]
    span = (np.array(span) / 5).tolist()
    # plt.subplot(132)
    # plt.pcolor(gain, phase, image.transpose())
    plt.xlabel("Gain")
    plt.ylabel("Phase imbalance [rad]")
    plt.title(f"Minimum at (gain={centers[0]:.3f}, phase={centers[1]:.3f}) = {image[minimum[0]][minimum[1]]:.1f} dBm")
    plt.show()
plt.colorbar()
plt.suptitle(f"Image cancellation for {element}")

q = centers[0]
p = centers[1]

correction_matrix = IQ_imbalance(q, p).tolist()

modify_json(qubit, element, "resonator_correction_matrix", correction_matrix)

print(f"For {element}, gain is {centers[0]} and phase is {centers[1]}")
plt.show()
