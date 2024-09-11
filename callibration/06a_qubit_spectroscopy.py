from experiment_utils.configuration import *
import matplotlib.pyplot as plt
from experiments_objects.qubit_spectroscopy import Qubit_Spec

if __name__ == "__main__":
    args = {
        'qubit': 'qubit4',
        'n_avg': 2000,
        'N': 200,
        'span': 0.2 * u.MHz,
        'state_discrimination': True,
        'pulse_amplitude': 0.03,
        'pulse_length': 60 * u.ns,
    }

    qubit_spec = Qubit_Spec(**args)
    qubit_spec.generate_experiment()
    qubit_spec.execute()
    # %%
    qubit_spec.plot(with_fit=True)
    qubit_spec.save()
    plt.show()

    response = input("Do you want to update qubit freq? (yes/no): ").strip().lower()
    if response == 'y':
        qubit_spec.update_max_freq()
        print("Qubit frequency updated.")
    else:
        print("Qubit frequency not updated.")
