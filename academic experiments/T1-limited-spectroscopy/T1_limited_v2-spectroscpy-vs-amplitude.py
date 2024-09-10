from experiments_objects.qubit_spectroscopy import T1_spectropcpy
import labber_util as lu
from configuration import *
import matplotlib.pyplot as plt
from time_estimation import calculate_time

if __name__ == "__main__":
    exp_args = {
        'qubit': 'qubit4',
        'n_avg': 500,
        'N': 100,
        'span': 5 * u.MHz,
        'state_discrimination': True,
        'pulse_type': 'lorentzian',
        'cutoff': 0.001,
        'eco': False,
        'n': 1 / 2,
        'pulse_length': 40 * u.us,
        'pulse_amplitude': 0.1
    }

    na = 10
    amplitudes = np.linspace(0.00, exp_args['pulse_amplitude'], na)

    calculate_time(exp_args['n_avg'], exp_args['N'], na)

    states = []
    for i, a in enumerate(amplitudes):
        print(
            f"################################ Experiment {i + 1}/{len(amplitudes)}, amplitude = {a} ################################")
        exp_args['pulse_amplitude'] = a
        qubit_spec = T1_spectropcpy(**exp_args)
        qubit_spec.generate_experiment()
        state = qubit_spec.execute()[2]

        states.append(state)

    amplitudes = amp_V_to_Hz(amplitudes)
    # %%
    import numpy as np

    t1 = qubit_args['T1']
    t2 = qubit_args['T2']

    states = np.array(states)

    plt.title(
        f'{exp_args["pulse_type"]} , eco = {exp_args["eco"]} \n pulse length = {exp_args["pulse_length"] / 1e3:.3f} us ,'
        f' pulse amplitude = {exp_args["pulse_amplitude"]} V ({qubit_spec.pulse_amp_Hz:.3f} MHz)'
        f'\n n = {exp_args["n"]} , cutoff = {exp_args["cutoff"]}')

    if exp_args['span'] < 3 * u.MHz:
        plt.axvline(1 / t2 * 1e3 / 2 / np.pi, color='b', linestyle='--', label='T2 limit')
        plt.axvline(-1 / t2 * 1e3 / 2 / np.pi, color='b', linestyle='--')
        plt.axvline(1 / t1 * 1e3 / 2 / np.pi, color='k', linestyle='--', label='T1 limit')
        plt.axvline(-1 / t1 * 1e3 / 2 / np.pi, color='k', linestyle='--')

    plt.pcolor(qubit_spec.detunings / 1e6, amplitudes, states)
    plt.xlabel('Detuning (MHz)')
    plt.ylabel('Amplitude (MHz)')
    plt.colorbar()
    plt.show()

    plt.plot(qubit_spec.detunings / 1e6, states[-1])
    plt.show()

    # data = {
    #     'states': states.tolist()
    # }
    # sweep = {
    #     'detunings': qubit_spec.detunings.tolist(),
    #     'amplitudes': amplitudes.tolist(),
    # }
    #
    # saver = Saver()
    #
    # saver.save('T1-limited-spectroscopy-vs-amplitude', data, sweep, exp_args, args)
    # print('Data saved')

    meta_data = {}

    # add tags and user
    meta_data["tags"] = ["Nadav-Lab", "spin-locking", "overnight"]
    meta_data["user"] = "Asaf"
    meta_data["exp_args"] = exp_args
    meta_data["args"] = args

    # arrange data in a form that is more suitable for labber (separate sweep parameters from measured ones, include units
    # etc.)
    measured_data = dict(states=states)
    sweep_parameters = dict(detuning=qubit_spec.detunings, amplitudes=amplitudes)
    units = dict(detuning="Hz", amplitudes="MHz")

    exp_result = dict(measured_data=measured_data, sweep_parameters=sweep_parameters, units=units, meta_data=meta_data)

    # create logfile
    lu.create_logfile("T1-qubit-spectroscopy", **exp_result, loop_type="2d")
