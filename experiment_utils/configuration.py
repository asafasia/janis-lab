import json
from qualang_tools.units import unit
from experiment_utils.pulses import *

args_path = 'C:/Users/owner/Documents/GitHub/janis-lab/experiment_utils/args.json'
optimal_weights_path = 'C:/Users/owner/Documents/GitHub/janis-lab/experiment_utils/optimal_weights.npz'


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


def state_measurement_stretch(fid_matrix, states):
    y = np.array([1 - states, states])
    bias = (fid_matrix[0][0] + fid_matrix[1][1]) / 2 - 0.5
    # print(f"Bias = {bias + 0.5}")

    inverse_fid_matrix = np.linalg.inv(fid_matrix)
    new_y = inverse_fid_matrix @ y - bias

    return new_y[1]


u = unit(coerce_to_integer=True)

sa_address = "TCPIP0::192.168.43.100::inst0::INSTR"
qm_host = "192.168.43.137"
qm_port = 9510

with open(args_path, 'r') as file:
    args = json.load(file)

con = 'con1'
qubit = 'qubit4'
#############################################
#                  Qubits                   #
#############################################
qubit_args = args[qubit]["qubit"]
qubit_LO = qubit_args['qubit_LO']
qubit_freq = qubit_args['qubit_freq']
qubit_IF = qubit_LO - qubit_freq
qubit_correction_matrix = qubit_args['qubit_correction_matrix']
thermalization_time = qubit_args['thermalization_time']
drag_coef = 0
anharmonicity = -200 * u.MHz
AC_stark_detuning = 0 * u.MHz
saturation_len = qubit_args['saturation_length']
saturation_amp = qubit_args['saturation_amplitude']
res_pulse_len = qubit_args['resonator_spec_pulse_length']
res_pulse_amp = qubit_args['resonator_spec_pulse_amplitude']
pi_pulse_length = qubit_args['pi_pulse_length']
pi_pulse_amplitude = qubit_args['pi_pulse_amplitude']
qubit_T1 = qubit_args['T1']

#############################################
#                Resonators                 #
#############################################
resonator_args = args[qubit]["resonator"]
resonator_LO = resonator_args['resonator_LO']
resonator_freq = resonator_args['resonator_freq']
resonator_IF = resonator_LO - resonator_freq
readout_len = resonator_args['readout_pulse_length']
readout_amp = resonator_args['readout_pulse_amplitude']
resonator_correction_matrix = resonator_args['resonator_correction_matrix']
time_of_flight = resonator_args['time_of_flight']
smearing = resonator_args['smearing']
fid_matrix = resonator_args['fidelity_matrix']
depletion_time = 2000
ringdown_length = 0


def amp_V_to_Hz(amp):
    return amp / pi_pulse_amplitude / (2 * pi_pulse_length * 1e-9) / 1e6



opt_weights = False
if opt_weights:
    from qualang_tools.config.integration_weights_tools import convert_integration_weights

    weights = np.load(optimal_weights_path)
    opt_weights_real = convert_integration_weights(weights["weights_real"])
    opt_weights_minus_imag = convert_integration_weights(weights["weights_minus_imag"])
    opt_weights_imag = convert_integration_weights(weights["weights_imag"])
    opt_weights_minus_real = convert_integration_weights(weights["weights_minus_real"])
else:
    opt_weights_real = [(1.0, readout_len)]
    opt_weights_minus_imag = [(0.0, readout_len)]
    opt_weights_imag = [(0.0, readout_len)]
    opt_weights_minus_real = [(-1.0, readout_len)]

#############################################
#                   else                    #
#############################################
const_pulse_len = 1000
const_pulse_amp = 0.1

##########################################
#               Flux line                #
##########################################
max_frequency_point = 0.0
flux_settle_time = 100 * u.ns

# Resonator frequency versus flux fit parameters according to resonator_spec_vs_flux
# amplitude * np.cos(2 * np.pi * frequency * x + phase) + offset
amplitude_fit, frequency_fit, phase_fit, offset_fit = [0, 0, 0, 0]

# FLux pulse parameters
const_flux_len = 200
const_flux_amp = 0.45

# IQ Plane Angle
rotation_angle = resonator_args["rotation_angle"]
# Threshold for single shot g-e discrimination
ge_threshold = resonator_args['threshold']

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
                1: {"offset": 0, "gain_db": 0},  # I from down-conversion
                2: {"offset": 0, "gain_db": 0},  # Q from down-conversion
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
                "saturation": "saturation_pulse",
                "res_spec": "res_spec_pulse",
                "x180": "x180_pulse",
                "x90": "x90_pulse",
                "-x90": "-x90_pulse",
                "y90": "y90_pulse",
                "-y90": "-y90_pulse",

            },
        },
        "qubit2": {
            "mixInputs": {
                "I": (con, qubit_args['IQ_input']['I']),
                "Q": (con, qubit_args['IQ_input']['Q']),
                "lo_frequency": qubit_LO,
                "mixer": "mixer_qubit2",
            },
            "intermediate_frequency": qubit_IF,
            "operations": {
                "x180": "x180_pulse",
                "x90": "x90_pulse",
                "-x90": "-x90_pulse",
                "y90": "y90_pulse",
                "-y90": "-y90_pulse",
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
            "smearing": smearing,

        },
    },

    "pulses": {
        "const_pulse": {
            "operation": "control",
            "length": 1000,
            "waveforms": {
                "I": "const_wf",
                "Q": "zero_wf",
            },
        },
        "saturation_pulse": {
            "operation": "control",
            "length": saturation_len,
            "waveforms":
                {
                    "I": "saturation_drive_wf",
                    "Q": "zero_wf"
                },
        },
        "res_spec_pulse": {
            "operation": "control",
            "length": res_pulse_len,
            "waveforms":
                {
                    "I": "resonator_spec_drive_wf",
                    "Q": "zero_wf"
                },
        },

        "x180_pulse": {
            "operation": "control",
            "length": pi_pulse_length,
            "waveforms": {
                "I": "x180_I_wf",
                "Q": "x180_Q_wf",
            },
        },
        "x90_pulse": {
            "operation": "control",
            "length": pi_pulse_length,
            "waveforms": {
                "I": "x90_I_wf",
                "Q": "x90_Q_wf",
            },
        },
        "-x90_pulse": {
            "operation": "control",
            "length": pi_pulse_length,
            "waveforms": {
                "I": "minus_x90_I_wf",
                "Q": "minus_x90_Q_wf",
            },
        },
        "y90_pulse": {
            "operation": "control",
            "length": pi_pulse_length,
            "waveforms": {
                "I": "y90_I_wf",
                "Q": "y90_Q_wf",
            },
        },
        "-y90_pulse": {
            "operation": "control",
            "length": pi_pulse_length,
            "waveforms": {
                "I": "minus_y90_I_wf",
                "Q": "minus_y90_Q_wf",
            },
        },

        "readout_pulse": {
            "operation": "measurement",
            "length": readout_len,
            "waveforms": {
                "I": "readout_wf",
                "Q": "zero_wf"
            },
            "integration_weights": {
                "cos": "rotated_cosine_weights",
                "sin": "rotated_sine_weights",
                "minus_sin": "rotated_minus_sine_weights",
                "opt_cos": "opt_cosine_weights",
                "opt_sin": "opt_sine_weights",
                "opt_minus_sin": "opt_minus_sine_weights",
            },
            "digital_marker": "ON"

        },

    },

    "waveforms": {
        "const_wf": {"type": "constant", "sample": 0.1},
        "saturation_drive_wf": {"type": "constant", "sample": saturation_amp},
        "resonator_spec_drive_wf": {"type": "constant", "sample": res_pulse_amp},
        "zero_wf": {"type": "constant", "sample": 0.0},
        "x180_I_wf": {"type": "constant", "sample": 0},
        "x180_Q_wf": {"type": "constant", "sample": qubit_args['pi_pulse_amplitude']},
        "x90_I_wf": {"type": "constant", "sample": 0},
        "x90_Q_wf": {"type": "constant", "sample": pi_pulse_amplitude / 2},
        "minus_x90_I_wf": {"type": "constant", "sample": 0},
        "minus_x90_Q_wf": {"type": "constant", "sample": -pi_pulse_amplitude / 2},
        "y90_I_wf": {"type": "constant", "sample": -pi_pulse_amplitude / 2},
        "y90_Q_wf": {"type": "constant", "sample": 0},
        "minus_y90_I_wf": {"type": "constant", "sample": qubit_args['pi_pulse_amplitude'] / 2},
        "minus_y90_Q_wf": {"type": "constant", "sample": 0},
        "readout_wf": {"type": "constant", "sample": readout_amp},
    },

    "mixers": {
        "mixer_qubit": [
            {
                "intermediate_frequency": qubit_IF,
                "lo_frequency": qubit_LO,
                "correction": IQ_imbalance(-0.0048, 0.0816)
            }

        ],
        "mixer_qubit2": [
            {
                "intermediate_frequency": qubit_IF,
                "lo_frequency": qubit_LO,
                "correction": IQ_imbalance(-0.0048, 0.0816)
            }

        ],
        "mixer_resonator": [
            {
                "intermediate_frequency": resonator_IF,
                "lo_frequency": resonator_LO,
                "correction": IQ_imbalance(-0.0552, 0.104)

            }

        ],
    },
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},
    },
    "integration_weights": {
        "cosine_weights": {
            "cosine": [(1.0, readout_len)],
            "sine": [(0.0, readout_len)],
        },
        "sine_weights": {
            "cosine": [(0.0, readout_len)],
            "sine": [(1.0, readout_len)],
        },
        "minus_sine_weights": {
            "cosine": [(0.0, readout_len)],
            "sine": [(-1.0, readout_len)],
        },
        "opt_cosine_weights": {
            "cosine": opt_weights_real,
            "sine": opt_weights_minus_imag,
        },
        "opt_sine_weights": {
            "cosine": opt_weights_imag,
            "sine": opt_weights_real,
        },
        "opt_minus_sine_weights": {
            "cosine": opt_weights_minus_imag,
            "sine": opt_weights_minus_real,
        },
        "rotated_cosine_weights": {
            "cosine": [(np.cos(rotation_angle), readout_len - ringdown_length)],
            "sine": [(np.sin(rotation_angle), readout_len - ringdown_length)],
        },
        "rotated_sine_weights": {
            "cosine": [(-np.sin(rotation_angle), readout_len - ringdown_length)],
            "sine": [(np.cos(rotation_angle), readout_len - ringdown_length)],
        },
        "rotated_minus_sine_weights": {
            "cosine": [(np.sin(rotation_angle), readout_len - ringdown_length)],
            "sine": [(-np.cos(rotation_angle), readout_len - ringdown_length)],
        },
    }
}
