from experiments_objects.qubit_spectroscopy import T1_spectropcpy
from experiment_utils.configuration import *
import matplotlib.pyplot as plt
from experiment_utils.time_estimation import calculate_time
import experiment_utils.labber_util as lu
from experiment_utils.MHz_to_Volt import amp_MHz_to_Volt
import numpy as np

if __name__ == "__main__":
    exp_args = {
        'qubit': 'qubit4',
        'n_avg': 8000,
        'N': 200,
        'span': 0.1 * u.MHz,
        'state_discrimination': True,
        'pulse_type': 'square',
        'cutoff': 0.005,
        'eco': False,
        'n': 1 / 2,
        'pulse_length': 60 * u.us,
        'pulse_amplitude': amp_MHz_to_Volt(0.01)  # ~ MHz
    }
    exp_name = 'T1-qubit-spectroscopy'
    exp_name += f'-{exp_args["pulse_type"]}'
    if exp_args['eco']:
        exp_name += '-eco'


    def run_experiment_for_amplitudes(amplitudes):
        states = []
        for i, a in enumerate(amplitudes):
            print(
                f"################################ Experiment {i + 1}/{len(amplitudes)}, amplitude = {a:0.2f} V ################################")
            exp_args['pulse_amplitude'] = a
            qubit_spec = T1_spectropcpy(**exp_args)
            qubit_spec.generate_experiment()
            state = qubit_spec.execute()[2]

            states.append(state)

        amplitudes = amp_V_to_Hz(amplitudes)

        return amplitudes, qubit_spec.detunings, states


    # %%
    na = 10
    amplitudes = np.linspace(0.00, exp_args['pulse_amplitude'], na)
    calculate_time(exp_args['n_avg'], exp_args['N'], na)
    amplitudes, detunings, states = run_experiment_for_amplitudes(amplitudes)

    t1 = qubit_args['T1']
    t2 = qubit_args['T2']

    states = np.array(states)

    plt.title(
        f'{exp_args["pulse_type"]} , eco = {exp_args["eco"]} \n pulse length = {exp_args["pulse_length"] / 1e3:.3f} us ,'
        f' pulse amplitude = {exp_args["pulse_amplitude"] * 1e3:3f} mV ({amplitudes[-1]:.3f} MHz)'
        f'\n n = {exp_args["n"]} , cutoff = {exp_args["cutoff"]}')

    if exp_args['span'] < 0.2 * u.MHz:
        plt.axvline(1 / t2 * 1e3 / 2 / np.pi, color='b', linestyle='--', label='T2 limit')
        plt.axvline(-1 / t2 * 1e3 / 2 / np.pi, color='b', linestyle='--')
        plt.axvline(1 / t1 * 1e3 / 2 / np.pi, color='k', linestyle='--', label='T1 limit')
        plt.axvline(-1 / t1 * 1e3 / 2 / np.pi, color='k', linestyle='--')

    plt.pcolor(detunings / 1e6, amplitudes, states)
    plt.xlabel('Detuning (MHz)')
    plt.ylabel('Amplitude (MHz)')
    plt.colorbar()
    plt.show()

    plt.plot(detunings / 1e6, states[-1])
    plt.show()

    meta_data = {}
    meta_data["tags"] = ["Nadav-Lab", "T1-limited-spectroscopy-2d", "overnight"]
    meta_data["user"] = "Asaf"
    meta_data["exp_args"] = exp_args
    meta_data["args"] = args
    measured_data = dict(states=states)
    sweep_parameters = dict(detuning=detunings, amplitudes=amplitudes)
    units = dict(detuning="Hz", amplitudes="MHz")
    exp_result = dict(measured_data=measured_data, sweep_parameters=sweep_parameters, units=units, meta_data=meta_data)
    lu.create_logfile(exp_name, **exp_result, loop_type="2d")
