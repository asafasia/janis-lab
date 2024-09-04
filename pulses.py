import numpy as np
from matplotlib import pyplot as plt


def generate_eco_pulse(amplitude=0.01, length=1000):
    ts = np.linspace(-1, 1, length)
    vec = (2 * np.heaviside(ts, 0) - 1) * amplitude
    return vec.tolist()


def generate_lorentzian_pulse(amplitude=0.01, length=1000, cutoff=0.1, n=1 / 2):
    sigma = ((1) ** 2 / ((1 / cutoff ** (1 / n)) - 1)) ** (1 / 2)
    print(f'sigma = {sigma * length} ns')
    ts = np.linspace(-1, 1, length)
    vec = amplitude / (1 + (ts / sigma) ** 2) ** n
    return vec.tolist()


def generate_half_lorentzian_pulse(amplitude=0.01, length=1000, cutoff=0.1, n=1 / 2):
    vec = np.array(generate_lorentzian_pulse(amplitude=amplitude, length=length, cutoff=cutoff, n=n))
    half = generate_eco_pulse(amplitude=1, length=length)
    # half = -np.sin(np.pi * ts / 1)
    return (half * vec).tolist()


if __name__ == "__main__":
    amplitude = 1
    cutoff = 0.1
    length = 10000
    eco_pulse_samples = generate_eco_pulse(amplitude=amplitude, length=length)
    lorentzian_pulse_samples = generate_lorentzian_pulse(amplitude=amplitude, length=length, cutoff=cutoff)
    generate_half_lorentzian_pulse(amplitude=amplitude, length=length, cutoff=cutoff)
    plt.plot(eco_pulse_samples)
    plt.plot(lorentzian_pulse_samples)
    plt.plot(generate_half_lorentzian_pulse(amplitude=amplitude, length=length, cutoff=cutoff))
    plt.axhline(cutoff)
    plt.show()
