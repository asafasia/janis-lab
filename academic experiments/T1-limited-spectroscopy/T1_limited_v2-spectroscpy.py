from scipy.ndimage import gaussian_filter1d
from experiments_objects.qubit_spectroscopy import T1_spectropcpy
from experiment_utils.configuration import *
import matplotlib.pyplot as plt
import experiment_utils.labber_util as lu

if __name__ == "__main__":
    exp_args = {
        'qubit': 'qubit4',
        'n_avg': 1500,
        'N': 100,
        'span': 0.20 * u.MHz,
        'state_discrimination': True,
        'pulse_type': 'lorentzian',
        'cutoff': 0.0005,
        'eco': False,
        'n': 1 / 4,
        'pulse_length': 20 * u.us,
        'pulse_amplitude': 0.051
    }

    qubit_spec = T1_spectropcpy(**exp_args)
    qubit_spec.generate_experiment()
    state = qubit_spec.execute()[2]

    # %%
    t1 = qubit_args['T1']
    t2 = qubit_args['T2']

    plt.plot(qubit_spec.detunings / 1e6, state, label='measured data')

    sigma = 10  # Standard deviation for Gaussian kernel
    y_smooth = gaussian_filter1d(state, sigma=sigma)

    plt.plot(qubit_spec.detunings / 1e6, y_smooth)
    if exp_args['span'] < 3 * u.MHz:
        plt.axvline(1 / t2 * 1e3 / 2 / np.pi, color='b', linestyle='--',
                    label=f'T2 limit ~ {1 / t2 / 2 / np.pi * 1e3:0.2f} MHz')
        plt.axvline(-1 / t2 * 1e3 / 2 / np.pi, color='b', linestyle='--')
        plt.axvline(1 / t1 * 1e3 / 2 / np.pi, color='k', linestyle='--', label='T1 limit')
        plt.axvline(-1 / t1 * 1e3 / 2 / np.pi, color='k', linestyle='--')

    plt.xlabel("Detuning (MHz)")
    plt.ylabel("State")
    plt.ylim([0, 1])
    plt.legend()
    plt.show()

    meta_data = {
        "tags": ["Nadav-Lab", "T1-limited-spectroscopy", "overnight"],
        "user": "Asaf",
        "exp_args": exp_args,
        "args": args
    }
    measured_data = dict(states=state)
    sweep_parameters = dict(detuning=qubit_spec.detunings)
    units = dict(detuning="Hz")
    exp_result = dict(measured_data=measured_data, sweep_parameters=sweep_parameters, units=units, meta_data=meta_data)
    lu.create_logfile("T1-qubit-spectroscopy", **exp_result, loop_type="1d")