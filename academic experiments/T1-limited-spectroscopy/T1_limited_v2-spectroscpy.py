from scipy.ndimage import gaussian_filter1d
from experiments_objects.qubit_spectroscopy import T1_spectropcpy
from configuration import *
import matplotlib.pyplot as plt

if __name__ == "__main__":
    exp_args = {
        'qubit': 'qubit4',
        'n_avg': 1000,
        'N': 100,
        'span': 1 * u.MHz,
        'state_discrimination': True,
        'pulse_type': 'square',
        'cutoff': 0.001,
        'eco': True,
        'n': 1 / 4,
        'pulse_length': 20 * u.us,
        'pulse_amplitude': 0.031
    }

    qubit_spec = T1_spectropcpy(**exp_args)
    qubit_spec.generate_experiment()
    # qubit_spec.simulate()
    # plt.show()
    state = qubit_spec.execute()[2]
    # qubit_spec.save(state)

    # %%
    t1 = qubit_args['T1']
    t2 = qubit_args['T2']

    plt.plot(qubit_spec.detunings / 1e6, state, label='measured data')

    sigma = 10  # Standard deviation for Gaussian kernel
    y_smooth = gaussian_filter1d(state, sigma=sigma)

    plt.plot(qubit_spec.detunings / 1e6, y_smooth)
    if exp_args['span'] < 3 * u.MHz:
        plt.axvline(1 / t2 * 1e3 / 2 / np.pi, color='b', linestyle='--', label='T2 limit')
        plt.axvline(-1 / t2 * 1e3 / 2 / np.pi, color='b', linestyle='--')
        plt.axvline(1 / t1 * 1e3 / 2 / np.pi, color='k', linestyle='--', label='T1 limit')
        plt.axvline(-1 / t1 * 1e3 / 2 / np.pi, color='k', linestyle='--')

    plt.xlabel("Detuning (MHz)")
    plt.ylabel("State")
    # plt.ylim([-0.1, 1.1])
    # plt.title(
    #     f'{exp_args["pulse_type"]} , eco = {exp_args["eco"]} \n pulse length = {exp_args["pulse_length"] / 1e3:.3f} us ,'
    #     f' pulse amplitude = {exp_args["pulse_amplitude"]} V ({qubit_spec.pulse_amp_Hz:.3f} MHz)'
    #     f'\n n = {exp_args["n"]} , cutoff = {exp_args["cutoff"]}')
    plt.legend()

    plt.show()
#
