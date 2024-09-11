from experiment_utils.configuration import *


def amp_Volt_to_MHz(amp_Volt):
    return amp_Volt / pi_pulse_amplitude / (2 * pi_pulse_length * 1e-9) / 1e6


def amp_MHz_to_Volt(amp_MHz):
    return amp_MHz * 2 * pi_pulse_length * 1e-9 * 1e6 * pi_pulse_amplitude


if __name__ == "__main__":


    amp_Volt = 0.08

    amp_MHz = amp_Volt_to_MHz(amp_Volt)

    print(f'amp_Volt = {amp_MHz}')


    amp_MHz = 22.3462

    amp_Volt = amp_MHz_to_Volt(amp_MHz)

    print(f'amp_Volt = {amp_Volt}')

