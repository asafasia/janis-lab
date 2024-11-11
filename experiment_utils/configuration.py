import json
from qualang_tools.units import unit
from experiment_utils.pulses import *
from qualang_tools.config.waveform_tools import drag_gaussian_pulse_waveforms

args_path = 'C:/Users/owner/Documents/GitHub/janis-lab/experiment_utils/'
optimal_weights_path = 'C:/Users/owner/Documents/GitHub/janis-lab/experiment_utils/optimal_weights.npz'

user = 'Asaf'
if user == 'Asaf':
    args_path += 'args_asaf.json'
elif user == 'Ariel':
    args_path += 'args_ariel.json'
elif user == 'Guy':
    args_path += 'args_guy.json'


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
    bias = (fid_matrix[0][0] + fid_matrix[1][1]) / 2 - 0.5
    inverse_fid_matrix = np.linalg.inv(fid_matrix)
    p = 0.95
    bias = 0
    if isinstance(states, (int, float)):
        vec = np.array([1 - states, states])
        new_vec = vec.T @ inverse_fid_matrix - bias
        return new_vec[1]
    else:
        new_vec = []
        for state in states:
            vec = np.array([1 - state, state])
            new_vec.append(vec.T @ inverse_fid_matrix - bias)

        new_vec = np.array(new_vec)

        return new_vec.T[1]


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
mixer_qubit_g = 0.2054400000000001
mixer_qubit_phi = 0.19518545454545452
qubit_correction_matrix = IQ_imbalance(mixer_qubit_g, mixer_qubit_phi)

qubit_T1 = qubit_args['T1']
thermalization_time = qubit_args['thermalization_time']
# Saturation_pulse
saturation_len = qubit_args['saturation_length']
saturation_amp = qubit_args['saturation_amplitude']
# Square pi pulse
square_pi_len = qubit_args['pi_pulse_length']
square_pi_amp = qubit_args['pi_pulse_amplitude']
# Drag pulses
drag_coef = 0
anharmonicity = -200 * u.MHz
AC_stark_detuning = 0 * u.MHz

x180_len = 40
x180_sigma = x180_len / 5
x180_amp = 0.35
x180_wf, x180_der_wf = np.array(
    drag_gaussian_pulse_waveforms(x180_amp, x180_len, x180_sigma, drag_coef, anharmonicity, AC_stark_detuning)
)
x180_I_wf = x180_wf
x180_Q_wf = x180_der_wf
# No DRAG when alpha=0, it's just a gaussian.

x90_len = x180_len
x90_sigma = x90_len / 5
x90_amp = x180_amp / 2
x90_wf, x90_der_wf = np.array(
    drag_gaussian_pulse_waveforms(x90_amp, x90_len, x90_sigma, drag_coef, anharmonicity, AC_stark_detuning)
)
x90_I_wf = x90_wf
x90_Q_wf = x90_der_wf
# No DRAG when alpha=0, it's just a gaussian.

minus_x90_len = x180_len
minus_x90_sigma = minus_x90_len / 5
minus_x90_amp = -x90_amp
minus_x90_wf, minus_x90_der_wf = np.array(
    drag_gaussian_pulse_waveforms(
        minus_x90_amp,
        minus_x90_len,
        minus_x90_sigma,
        drag_coef,
        anharmonicity,
        AC_stark_detuning,
    )
)
minus_x90_I_wf = minus_x90_wf
minus_x90_Q_wf = minus_x90_der_wf
# No DRAG when alpha=0, it's just a gaussian.

y180_len = x180_len
y180_sigma = y180_len / 5
y180_amp = x180_amp
y180_wf, y180_der_wf = np.array(
    drag_gaussian_pulse_waveforms(y180_amp, y180_len, y180_sigma, drag_coef, anharmonicity, AC_stark_detuning)
)
y180_I_wf = (-1) * y180_der_wf
y180_Q_wf = y180_wf
# No DRAG when alpha=0, it's just a gaussian.

y90_len = x180_len
y90_sigma = y90_len / 5
y90_amp = y180_amp / 2
y90_wf, y90_der_wf = np.array(
    drag_gaussian_pulse_waveforms(y90_amp, y90_len, y90_sigma, drag_coef, anharmonicity, AC_stark_detuning)
)
y90_I_wf = (-1) * y90_der_wf
y90_Q_wf = y90_wf
# No DRAG when alpha=0, it's just a gaussian.

minus_y90_len = y180_len
minus_y90_sigma = minus_y90_len / 5
minus_y90_amp = -y90_amp
minus_y90_wf, minus_y90_der_wf = np.array(
    drag_gaussian_pulse_waveforms(
        minus_y90_amp,
        minus_y90_len,
        minus_y90_sigma,
        drag_coef,
        anharmonicity,
        AC_stark_detuning,
    )
)
minus_y90_I_wf = (-1) * minus_y90_der_wf
minus_y90_Q_wf = minus_y90_wf


# No DRAG when alpha=0, it's just a gaussian.

def amp_V_to_Hz(amp):
    return amp / square_pi_len / (2 * square_pi_amp * 1e-9) / 1e6


#############################################
#                Resonators                 #
#############################################
resonator_args = args[qubit]["resonator"]

resonator_LO = resonator_args['resonator_LO']
resonator_freq = resonator_args['resonator_freq']
resonator_IF = resonator_LO - resonator_freq
mixer_resonator_g = 0.1976436363636364
mixer_resonator_phi = 0.11934545454545453
resonator_correction_matrix = IQ_imbalance(mixer_resonator_g, mixer_resonator_phi)

readout_len = resonator_args['readout_pulse_length']
readout_amp = resonator_args['readout_pulse_amplitude']

time_of_flight = resonator_args['time_of_flight']
smearing = resonator_args['smearing']
depletion_time = 0 * u.us

fid_matrix = resonator_args['fidelity_matrix']
ringdown_length = 0

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

# IQ Plane
rotation_angle = resonator_args["rotation_angle"]
ge_threshold = resonator_args['threshold']

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

#############################################
#                  Config                   #
#############################################
config = {
    "version": 1,
    "controllers": {
        con: {
            "analog_outputs": {
                1: {"offset": resonator_args['IQ_bias']['I']},  # I resonator
                2: {"offset": resonator_args['IQ_bias']['Q']},  # Q resonator
                3: {"offset": qubit_args['IQ_bias']['I']},  # I qubit
                4: {"offset": qubit_args['IQ_bias']['Q']},  # Q qubit
            },
            "digital_outputs": {},
            "analog_inputs": {
                1: {"offset": 0.257, "gain_db": 3},  # I from down-conversion
                2: {"offset": 0.1913, "gain_db": 0},  # Q from down-conversion
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
                "pi": "square_pi_pulse",
                "pi_half": "square_pi_half_pulse",
                "x180": "x180_pulse",
                "y180": "y180_pulse",
                "x90": "x90_pulse",
                "-x90": "-x90_pulse",
                "y90": "y90_pulse",
                "-y90": "-y90_pulse",
                "y360": "y360_pulse",

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
                "I": (con, 1),
                "Q": (con, 2),
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
        "square_pi_pulse": {
            "operation": "control",
            "length": square_pi_len,
            "waveforms": {
                "I": "square_pi_wf",
                "Q": "zero_wf",
            },
        },
        "square_pi_half_pulse": {
            "operation": "control",
            "length": square_pi_len,
            "waveforms": {
                "I": "square_pi_half_wf",
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

        "x180_pulse": {
            "operation": "control",
            "length": square_pi_len,
            "waveforms": {
                "I": "x180_I_wf",
                "Q": "x180_Q_wf",
            },
        },
        "y180_pulse": {
            "operation": "control",
            "length": square_pi_len,
            "waveforms": {
                "I": "y180_I_wf",
                "Q": "y180_Q_wf",
            },
        },
        "x90_pulse": {
            "operation": "control",
            "length": square_pi_len,
            "waveforms": {
                "I": "x90_I_wf",
                "Q": "x90_Q_wf",
            },
        },
        "-x90_pulse": {
            "operation": "control",
            "length": square_pi_len,
            "waveforms": {
                "I": "minus_x90_I_wf",
                "Q": "minus_x90_Q_wf",
            },
        },
        "y90_pulse": {
            "operation": "control",
            "length": square_pi_len,
            "waveforms": {
                "I": "y90_I_wf",
                "Q": "y90_Q_wf",
            },
        },
        "-y90_pulse": {
            "operation": "control",
            "length": square_pi_len,
            "waveforms": {
                "I": "minus_y90_I_wf",
                "Q": "minus_y90_Q_wf",
            },
        },
        "y360_pulse": {
            "operation": "control",
            "length": square_pi_len,
            "waveforms": {
                "I": "y360_I_wf",
                "Q": "y360_Q_wf",
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
        "const_wf": {"type": "constant", "sample": 0.01},
        "saturation_drive_wf": {"type": "constant", "sample": saturation_amp},
        "square_pi_wf": {"type": "constant", "sample": square_pi_amp},
        "square_pi_half_wf": {"type": "constant", "sample": square_pi_amp / 2},
        "zero_wf": {"type": "constant", "sample": 0.0},
        "x90_I_wf": {"type": "constant", "sample": 0},
        "x90_Q_wf": {"type": "constant", "sample": square_pi_amp / 2},
        "x180_I_wf": {"type": "constant", "sample": 0},
        "x180_Q_wf": {"type": "constant", "sample": qubit_args['pi_pulse_amplitude']},
        "minus_x90_I_wf": {"type": "constant", "sample": 0},
        "minus_x90_Q_wf": {"type": "constant", "sample": -square_pi_amp / 2},
        "y90_I_wf": {"type": "constant", "sample": -square_pi_amp / 2},
        "y90_Q_wf": {"type": "constant", "sample": 0},
        "y180_I_wf": {"type": "constant", "sample": qubit_args['pi_pulse_amplitude']},
        "y180_Q_wf": {"type": "constant", "sample": 0},
        "minus_y90_I_wf": {"type": "constant", "sample": qubit_args['pi_pulse_amplitude'] / 2},
        "minus_y90_Q_wf": {"type": "constant", "sample": 0},
        "y360_I_wf": {"type": "constant", "sample": square_pi_amp * 2},
        "y360_Q_wf": {"type": "constant", "sample": 0},
        "readout_wf": {"type": "constant", "sample": readout_amp},
    },

    "mixers": {
        "mixer_qubit": [
            {
                "intermediate_frequency": qubit_IF,
                "lo_frequency": qubit_LO,
                "correction": qubit_correction_matrix
            }

        ],
        "mixer_qubit2": [
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
