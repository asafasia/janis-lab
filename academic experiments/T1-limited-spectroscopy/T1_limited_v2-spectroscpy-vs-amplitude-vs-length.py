from experiments_objects.qubit_spectroscopy import T1_spectropcpy
from experiment_utils.configuration import *
import matplotlib.pyplot as plt
from experiment_utils.time_estimation import calculate_time
import experiment_utils.labber_util as lu
import numpy as np

if __name__ == "__main__":
    exp_args = {
        'qubit': 'qubit4',
        'n_avg': 5000,
        'N': 200,
        'span': 5 * u.MHz,
        'state_discrimination': True,
        'pulse_type': 'lorentzian',
        'cutoff': 0.0005,
        'eco': False,
        'n': 1 / 2,
        'pulse_length': 20 * u.us,
        'pulse_amplitude': 0.2
    }


    def run_experiment_for_amplitudes(amplitudes, pulse_length):
        states = []
        for i, a in enumerate(amplitudes):
            print(
                f"################################ Experiment {i + 1}/{len(amplitudes)}, amplitude = {a:0.2f} V ################################")
            qubit_spec = T1_spectropcpy(**exp_args)
            qubit_spec.pulse_amplitude = a
            qubit_spec.pulse_length = pulse_length
            qubit_spec.generate_experiment()
            state = qubit_spec.execute()[2]

            states.append(state)

        amplitudes = amp_V_to_Hz(amplitudes)

        return amplitudes, qubit_spec.detunings, states


    na = 100
    nl = 20
    calculate_time(exp_args['n_avg'], exp_args['N'], na, nl, pulse_len=20e3)
    amplitudes = np.linspace(0.00, exp_args['pulse_amplitude'], na)
    lengths = np.linspace(10, 50, nl) * u.us
    print(lengths)
    for l in lengths:
        print('pulse length = ', l / 1e3, 'us')
        exp_args['pulse_length'] = int(l)  # for save
        # run
        amplitudes_MHz, detunings, states = run_experiment_for_amplitudes(amplitudes, int(l))
        # save
        meta_data = {}
        meta_data["tags"] = ["Nadav-Lab", "T1-limited-spectroscopy-2d", "overnight"]
        meta_data["user"] = "Asaf"
        meta_data["exp_args"] = exp_args
        meta_data["args"] = args
        measured_data = dict(states=states)
        sweep_parameters = dict(detuning=detunings, amplitudes=amplitudes_MHz)
        units = dict(detuning="Hz", amplitudes="MHz")
        exp_result = dict(measured_data=measured_data, sweep_parameters=sweep_parameters, units=units,
                          meta_data=meta_data)
        lu.create_logfile("T1-qubit-spectroscopy", **exp_result, loop_type="2d")
