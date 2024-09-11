import numpy as np
from matplotlib import pyplot as plt


def generate_eco_pulse(amplitude=0.01, length=1000):
    ts = np.linspace(-1, 1, length)
    vec = (2 * np.heaviside(ts, 0) - 1) * amplitude

    # vec = -np.sin(10 * np.pi * ts) * amplitude

    return vec.tolist()


def generate_lorentzian_pulse(amplitude=0.01, length=1000, cutoff=0.1, n=1 / 2):
    sigma = ((1) ** 2 / ((1 / cutoff ** (1 / n)) - 1)) ** (1 / 2)
    # print(f'sigma = {sigma * length} ns')
    # plt.axvline(x=sigma * length + length/2, color='r', linestyle='--')
    # plt.axvline(x=-sigma * length + length/2, color='r', linestyle='--')

    ts = np.linspace(-1, 1, length)
    vec = amplitude / (1 + (ts / sigma) ** 2) ** n
    return vec.tolist()


def generate_half_lorentzian_pulse(amplitude=0.01, length=1000, cutoff=0.1, n=1 / 2):
    vec = np.array(generate_lorentzian_pulse(amplitude=amplitude, length=length, cutoff=cutoff, n=n))
    half = generate_eco_pulse(amplitude=1, length=length)
    return (half * vec).tolist()


if __name__ == "__main__":
    amplitude = 1
    cutoff = 0.01
    length = 10000
    n = 1 / 4

    eco_pulse_samples = generate_eco_pulse(amplitude=amplitude, length=length)
    lorentzian_pulse_samples = generate_lorentzian_pulse(
        amplitude=amplitude,
        length=length,
        cutoff=cutoff,
        n=n

    )
    lorentzian_half_pulse_samples = generate_half_lorentzian_pulse(
        amplitude=amplitude,
        length=length,
        cutoff=cutoff,
        n=n
    )
    plt.plot(eco_pulse_samples)
    plt.plot(lorentzian_pulse_samples)
    plt.plot(lorentzian_half_pulse_samples)
    plt.axhline(cutoff, color='k', linestyle='--')
    plt.axhline(-cutoff, color='k', linestyle='--')

    plt.show()
