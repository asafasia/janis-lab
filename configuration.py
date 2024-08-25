import json
from qualang_tools.units import unit
import numpy as np


def IQ_imbalance(g, phi):
    """
    Creates the correction matrix for the mixer imbalance caused by the gain and phase imbalances, more information can
    be seen here:
    https://docs.qualang.io/libs/examples/mixer-calibration/#non-ideal-mixer
    :param g: relative gain imbalance between the 'I' & 'Q' ports. (unit-less), set to 0 for no gain imbalance.
    :param phi: relative phase imbalance between the 'I' & 'Q' ports (radians), set to 0 for no phase imbalance.
    """
    c = np.cos(phi)
    s = np.sin(phi)
    N = 1 / ((1 - g ** 2) * (2 * c ** 2 - 1))
    return [float(N * x) for x in [(1 - g) * c, (1 + g) * s, (1 - g) * s, (1 + g) * c]]


u = unit(coerce_to_integer=True)
depletion_time = 0.5 * u.us

sa_address = "TCPIP0::192.168.43.100::inst0::INSTR"
qm_host = "192.168.43.137"
qm_port = 9510

with open('args.json', 'r') as file:
    args = json.load(file)

con = 'con1'
qubit = 'qubit1'
#############################################
#                  Qubits                   #
#############################################
qubit_args = args[qubit]["qubit"]

qubit_LO = qubit_args['qubit_LO']
qubit_IF = qubit_args['qubit_IF']
qubit_correction_matrix = qubit_args['qubit_correction_matrix']

#############################################
#                Resonators                 #
#############################################
resonator_args = args[qubit]["resonator"]
resonator_LO = resonator_args['resonator_LO']
resonator_IF = resonator_args['resonator_IF']
readout_pulse_length = resonator_args['readout_pulse_length']
readout_pulse_amplitude = resonator_args['readout_pulse_amplitude']
resonator_correction_matrix = resonator_args['resonator_correction_matrix']
time_of_flight = resonator_args['time_of_flight']

#############################################
#                   else                    #
#############################################

const_len = 100
const_amp = 0.01

#############################################
#                  Config                   #
#############################################
config = {
    "version": 1,
    "controllers": {
        con: {
            "analog_outputs": {
                "1": {"offset": resonator_args['IQ_bias']['I']},  # I resonator
                "2": {"offset": resonator_args['IQ_bias']['Q']},  # Q resonator
                "3": {"offset": qubit_args['IQ_bias']['I']},  # I qubit
                "4": {"offset": qubit_args['IQ_bias']['Q']},  # Q qubit
            },
            "digital_outputs": {},
            "analog_inputs": {
                1: {"offset": 0.0, "gain_db": 0},  # I from down-conversion
                2: {"offset": 0.0, "gain_db": 0},  # Q from down-conversion
            },
        }
    },
    "elements": {
        "qubit": {
            "mixInputs": {
                "I": (con, qubit_args['IQ_input']['I']),
                "Q": (con, qubit_args['IQ_input']['Q']),
                "lo_frequency": qubit_LO,
                "mixer": "mixer_qubit",
            },
            "intermediate_frequency": qubit_IF,
            "operations": {
                "cw": "const_pulse",
            },
        },

        "resonator": {
            "mixInputs": {
                "I": (con, resonator_args['IQ_input']['I']),
                "Q": (con, resonator_args['IQ_input']['Q']),
                "lo_frequency": resonator_LO,
                "mixer": "mixer_resonator",
            },
            "intermediate_frequency": resonator_IF,
            "operations": {
                "cw": "const_pulse",
                "readout": "readout_pulse",
            },
            "outputs": {
                "out1": (con, 1),
                "out2": (con, 2),
            },
            "time_of_flight": time_of_flight,
            "smearing": 0,

        },
    },

    "pulses": {
        "const_pulse": {
            "operation": "control",
            "length": const_len,
            "waveforms": {
                "I": "const_wf",
                "Q": "zero_wf",
            },
        },

        "readout_pulse": {
            "operation": "measurement",
            "length": readout_pulse_length,
            "waveforms": {
                "I": "readout_wf",
                "Q": "zero_wf"
            },
            "integration_weights": {
                "cos": "cosine_weights",
                "sin": "cosine_weights",
                "minus_sin": "minus_sine_weights"
            },
            "digital_marker": "ON"

        },
    },

    "waveforms": {
        "const_wf": {"type": "constant", "sample": const_amp},
        "zero_wf": {"type": "constant", "sample": 0.0},
        "readout_wf": {"type": "constant", "sample": readout_pulse_amplitude}
    },

    "mixers": {
        "mixer_qubit": [
            {
                "intermediate_frequency": qubit_IF,
                "lo_frequency": qubit_LO,
                "correction": qubit_correction_matrix
            }

        ],
        "mixer_resonator": [
            {
                "intermediate_frequency": resonator_IF,
                "lo_frequency": resonator_LO,
                "correction": resonator_correction_matrix

            }

        ],
    },
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},
    },
    "integration_weights": {
        "cosine_weights": {
            "cosine": [(1.0, readout_pulse_length)],
            "sine": [(0.0, readout_pulse_length)]
        },
        "sine_weights": {
            "cosine": [(0.0, readout_pulse_length)],
            "sine": [(1.0, readout_pulse_length)]
        },
        "minus_sine_weights": {
            "cosine": [(0.0, readout_pulse_length)],
            "sine": [(-1.0, readout_pulse_length)],
        },
    }

}
