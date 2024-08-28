from matplotlib import pyplot as plt
import numpy as np
from qm import QuantumMachinesManager
from instruments_py27.spectrum_analyzer import N9010A_SA
from configuration import *
from pprint import pprint
from qm.qua import *


def create_vectors_from_string(input_string):
    """
    Creates two vectors from a comma-separated string based on alternating indices.

    Args:
        input_string (str): The input string to be split.

    Returns:
        tuple: A tuple containing two lists of numbers.
    """
    # Split the input string by commas and convert it to a list of floats
    numbers = list(map(float, input_string.split(',')))

    # Create two vectors based on alternating indices
    vector1 = numbers[0::2]  # Elements at even indices (0, 2, 4, ...)
    vector2 = numbers[1::2]  # Elements at odd indices (1, 3, 5, ...)

    return np.array(vector1), np.array(vector2)


def plot_traces(center_freq, span, BW, points, average=False):
    sa = N9010A_SA(sa_address, False)
    sa.setup_averaging(average, 10)
    sa.setup_spectrum_analyzer(center_freq=center_freq, span=span, BW=BW,
                               points=points)
    y = sa.get_data()

    # Example usage
    x, y = create_vectors_from_string(y)

    plt.plot(x / 1e6, y, label='')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Power (dBm)')
    plt.title('Spectrum Analyzer Data')
    plt.ylim([-100, 0])


if __name__ == "__main__":
    qmm = QuantumMachinesManager(host=qm_host, port=qm_port)
    qm = qmm.open_qm(config)

    element = "qubit"

    with program() as prog:
        with infinite_loop_():
            play("readout", "resonator")
            play("saturation", "qubit")

    pending_job = qm.queue.add_to_start(prog)

    center_freq = args['qubit1'][element][f"{element}_LO"] / 1e6,
    span = 500e6,
    BW = 0.2e6,
    points = 5000
    average = True

    plot_traces(center_freq, span, BW, points, average)
    # plt.axvline(x=qubit_freq / u.MHz, color='r', linestyle='--', label='Qubit frequency')
    plt.legend()
    plt.show()
