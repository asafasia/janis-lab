from qm.qua import *
from qm import QuantumMachinesManager
from qualang_tools.bakery import baking
from configuration import *

with baking(config, padding_method="symmetric_r") as b:
    # Create arbitrary waveforms

    singleInput_sample = [0.4, 0.3, 0.2, 0.3, 0.3]
    mixInput_sample_I = [0.2, 0.3, 0.4]
    mixInput_sample_Q = [0.1, 0.2, 0.4]

    # Assign waveforms to quantum element operation

    # b.add_op("single_Input_Op", "qubit", singleInput_sample, digital_marker=None)
    b.add_op("mix_Input_Op", "qubit", [mixInput_sample_I, mixInput_sample_Q], digital_marker=None)

    # Play the operations

    # b.play("single_Input_Op", "qubit")
    b.play("mix_Input_Op", "qubit")

    # print(b.)

