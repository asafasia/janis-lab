import numpy as np
from matplotlib import pyplot as plt
from time import sleep
import time
from qm import QuantumMachinesManager
from instruments_py27.spectrum_analyzer import N9010A_SA
from configuration import *
from qm.qua import *
from change_args import modify_json
from get_traces_spectrum import plot_traces

I_FIG_NUM = 1
Q_FIG_NUM = 2
F = plt.figure(I_FIG_NUM)
plotFigs = True
element = 'qubit'

LOFreq = args[qubit][element][f"{element}_LO"]

I0 = 0
Q0 = 0


def open_qm():
    qm = QuantumMachinesManager(qm_host, 9510)

    return qm.open_qm({
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
    })


def getWithIQ(IQ, qm, sa, verbose=True):
    """Sets DAC output to I=IQ[0] and Q=IQ[1] and measures with spectrum analyzer"""
    qm.set_output_dc_offset_by_element("RR1", "single", float(IQ[0]))
    qm.set_output_dc_offset_by_element("RR2", "single", float(IQ[1]))
    sa.setup_averaging(average, 4)

    if average:
        sleep(0.5)
    else:
        sleep(0.1)

    t = sa.get_marker()

    if verbose:
        print("Transmitted power is %f dBm" % t)
    return t


def findMinI(I0, Q0, currRange, numPoints, qm, SA, plotRes=False):
    """scans numPoints +/-currRange/2 around I0 with a constant Q0
    returns the I which gave the minimal transmission
    plot scan if plotRes = True
    """
    scanVec = np.linspace(max([I0 - currRange / 2, -0.5]), min([I0 + currRange / 2, 0.5 - 2 ** -16]), numPoints)
    tRes = []

    for val in scanVec:
        tRes.append(getWithIQ([val, Q0], qm, SA))
        # print(tRes)

    if plotRes:
        plt.figure(I_FIG_NUM)
        plt.plot(scanVec, tRes, label=str(Q0))

    minVal = min(tRes)
    return minVal, scanVec[tRes.index(minVal)]


def findMinQ(I0, Q0, currRange, numPoints, qm, SA, plotRes=False):
    """scans numPoints +/-currRange/2 around Q0 with a constant I0
    returns the Q which gave the minimal transmission
    plot scan if plotRes = True
    """
    scanVec = np.linspace(max([Q0 - currRange / 2, -0.5]), min([Q0 + currRange / 2, 0.5 - 2 ** -16]), numPoints)
    tRes = []

    for val in scanVec:
        tRes.append(getWithIQ([I0, val], qm, SA))

    if plotRes:
        plt.figure(Q_FIG_NUM)
        plt.plot(scanVec, tRes, label=str(I0))

    minVal = min(tRes)
    return minVal, scanVec[tRes.index(minVal)]


with program() as prog:
    with infinite_loop_():
        play("pulse", "RR1")
        play("pulse", "RR2")

if __name__ == "__main__":

    qubit = 'qubit1'
    average = False
    I0 = 0
    Q0 = 0

    I_port = args[qubit][element]["IQ_input"]["I"]
    Q_port = args[qubit][element]["IQ_input"]["Q"]

    sa = N9010A_SA(sa_address, False)
    sa.setup_spectrum_analyzer(center_freq=LOFreq / 1e6, span=0.5e6, BW=0.1e6, points=15)
    sa.set_marker_max()
    qm = open_qm()

    job = qm.execute(prog)
    getWithIQ([0.0, 0.0], qm, sa)  # send a sequence in order to have a trigger before setting SA marker to max

    currMin = 100.0  # minimal transmission, start with a high value
    currRange = 0.2  # 0.98#0.80 # 2**10 #range to scan around the minima
    minimum = -90  # Stop at this value
    numPoints = 20  # number of points
    I0 = 0.0
    Q0 = 0.0

    start = time.time()
    while currMin > minimum and currRange >= 16. / 2 ** 16:
        minTI, I0 = findMinI(I0, Q0, currRange, numPoints, qm, sa, plotFigs)  # scan I
        currMin, Q0 = findMinQ(I0, Q0, currRange, numPoints, qm, sa, plotFigs)  # Scan Q
        print(f"Range = {currRange}, I0 = {I0}, Q0 = {Q0}, currMin = {currMin} ")
        currRange = currRange / 2
    end = time.time()

    print("Elapsed time is %f seconds" % (end - start))

    plt.figure(I_FIG_NUM)

    plt.xlabel("I [pixels]")
    plt.ylabel("Transmitted power [dBm]")
    plt.legend()
    plt.draw()
    plt.figure(Q_FIG_NUM)
    plt.xlabel("Q [pixels]")
    plt.ylabel("Transmitted power [dBm]")
    plt.legend()
    plt.draw()
    plt.show()

    print(I0, Q0)
    qm.set_output_dc_offset_by_element("RR1", "single", I0)
    qm.set_output_dc_offset_by_element("RR2", "single", Q0)

    plot_traces(LOFreq / 1e6, 500e6, 0.1e6, 5005, True)
    plt.show()

    response = input("Do you want to updata IQ bias? (yes/no): ").strip().lower()

    if response == 'y':
        print("Okay, let's update the IQ bias.")
        modify_json(qubit, element, "IQ_bias", {"I": I0, "Q": Q0})
    elif response == 'n':
        print("Okay, maybe next time.")
    else:
        print("Invalid response. Please enter 'y' or 'n'.")
