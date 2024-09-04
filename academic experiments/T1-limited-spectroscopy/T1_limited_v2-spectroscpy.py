from importlib import reload
from scipy.ndimage import gaussian_filter1d

import numpy as np
from qualang_tools.bakery import baking

import configuration
from saver import Saver

reload(configuration)
from change_args import modify_json
from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from qualang_tools.plot import interrupt_on_close
from change_args import modify_json
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
import matplotlib.pyplot as plt
from macros import readout_macro


class Qubit_Spec:
    def __init__(
            self,
            qubit,
            n_avg,
            N,
            span,
            state_discrimination,
            pulse_type,
            eco,
            n,
            cutoff,
            pulse_length,
            pulse_amplitude,
    ):
        self.qubit = qubit
        self.qmm = QuantumMachinesManager(host=qm_host, port=qm_port)
        self.span = span
        self.N = N
        self.n_avg = n_avg
        self.frequencies = qubit_LO - np.arange(qubit_freq - self.span / 2, qubit_freq + self.span / 2,
                                                self.span // self.N)
        self.detunings = qubit_freq - qubit_LO + self.frequencies
        self.state_discrimination = state_discrimination
        self.pulse_type = pulse_type
        self.eco = eco
        self.n = n
        self.cutoff = cutoff
        self.experiment = None
        self.pulse_length = pulse_length
        self.pulse_amplitude = pulse_amplitude
        self.pulse_amp_Hz = pulse_amplitude / pi_pulse_amplitude / (2 * pi_pulse_length * 1e-9) / 1e6

        print("rabi_freq = ", self.pulse_amp_Hz, "MHz")

    def generate_experiment(self, pi_pulse=None):
        with baking(config, "symmetric_r") as b:
            if self.pulse_type == 'lorentzian':
                if self.eco:
                    vec = generate_half_lorentzian_pulse(self.pulse_amplitude, self.pulse_length, self.cutoff, self.n)
                else:
                    vec = generate_lorentzian_pulse(self.pulse_amplitude, self.pulse_length, self.cutoff, self.n)
            elif self.pulse_type == 'square':
                if self.eco:
                    vec = generate_eco_pulse(self.pulse_amplitude, self.pulse_length)
                else:
                    vec = self.pulse_amplitude * np.ones(int(self.pulse_length))
            else:
                raise ValueError("pulse type not recognized")

            b.add_op("lorentzian", "qubit", [np.zeros_like(vec), vec])
            b.play("lorentzian", "qubit")

        with program() as qubit_spec:
            n = declare(int)  # QUA variable for the averaging loop
            df = declare(int)  # QUA variable for the qubit frequency
            I = declare(fixed)  # QUA variable for the measured 'I' quadrature
            Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature
            I_st = declare_stream()  # Stream for the 'I' quadrature
            Q_st = declare_stream()  # Stream for the 'Q' quadrature
            n_st = declare_stream()  # Stream for the averaging iteration 'n'
            state = declare(bool)  # QUA variable for state discrimination
            state_st = declare_stream()  # Stream for the qubit state

            with for_(n, 0, n < self.n_avg, n + 1):
                with for_(*from_array(df, self.frequencies)):
                    update_frequency("qubit", df)
                    # play("eco", "qubit")
                    b.run()
                    wait(100, "qubit")
                    align("qubit", "resonator")
                    measure(
                        "readout",
                        "resonator",
                        None,
                        dual_demod.full('cos', 'out1', 'sin', 'out2', I),
                        dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q)
                    )
                    align()

                    wait(thermalization_time // 4, "resonator")

                    assign(state, I > ge_threshold)
                    save(state, state_st)
                    save(I, I_st)
                    save(Q, Q_st)
                save(n, n_st)

            with stream_processing():
                I_st.buffer(len(self.frequencies)).average().save("I")
                Q_st.buffer(len(self.frequencies)).average().save("Q")
                state_st.boolean_to_int().buffer(len(self.frequencies)).average().save("state")
                n_st.save("iteration")

        self.experiment = qubit_spec

    def simulate(self):
        simulation_config = SimulationConfig(duration=10000)  # In clock cycles = 4ns
        job = self.qmm.simulate(config, self.experiment, simulation_config)
        job.get_simulated_samples().con1.plot()

    def execute(self):
        qm = self.qmm.open_qm(config)
        job = qm.execute(self.experiment)
        results = fetching_tool(job, data_list=["I", "Q", "state", "iteration"], mode="live")
        while results.is_processing():
            I, Q, state, iteration = results.fetch_all()
            progress_counter(iteration, self.n_avg, start_time=results.get_start_time())

        # if state_measurement_stretch:
        state = state_measurement_stretch(resonator_args['fidelity_matrix'], state)

        self.I, self.Q, self.state = I, Q, state

        return I, Q, state

    def plot(self):
        t1 = qubit_args['T1']
        t2 = qubit_args['T2']
        plt.plot(self.detunings / 1e6, self.state)
        max_freq = self.detunings[np.argmax(self.state)]
        plt.axvline(max_freq / 1e6, color='r', linestyle='--')
        plt.axvline(1 / t2 * 1e3 / 2, color='b', linestyle='--')
        plt.axvline(-1 / t2 * 1e3 / 2, color='b', linestyle='--')
        plt.xlabel("Detuning (MHz)")
        plt.ylabel("State")
        plt.show()

    def update_max_freq(self):
        max_freq = self.frequencies[np.argmax(self.state)]
        modify_json(self.qubit, 'qubit', 'qubit_freq', qubit_LO - max_freq)

    def save(self, state):
        saver = Saver()
        measured_data = {
            'I': None,
            'Q': None,
            'state': state.tolist(),
        }
        sweep = {
            'detunings': self.detunings.tolist(),
            'frequencies': self.frequencies.tolist(),
        }
        metadata = {
            'n_avg': self.n_avg,
        }
        saver.save('T1_limit_spectroscopy', measured_data, sweep, metadata, args)


if __name__ == "__main__":
    exp_args = {
        'qubit': 'qubit4',
        'n_avg': 200,
        'N': 1350,
        'span': 0.3 * u.MHz,
        'state_discrimination': True,
        'pulse_type': 'lorentzian',
        'cutoff': 0.001,
        'eco': True,
        'n': 1 / 2,
        'pulse_length': 20 * u.us,
        'pulse_amplitude': 0.2
    }

    qubit_spec = Qubit_Spec(**exp_args)
    qubit_spec.generate_experiment('x90')
    qubit_spec.simulate()
    plt.show()
    _, _, state = qubit_spec.execute()
    qubit_spec.save(state)

    # %%
    t1 = qubit_args['T1']
    t2 = qubit_args['T2']

    plt.plot(qubit_spec.detunings / 1e6, state, label='measured data')

    sigma = 30  # Standard deviation for Gaussian kernel
    y_smooth = gaussian_filter1d(state, sigma=sigma)

    plt.plot(qubit_spec.detunings / 1e6, y_smooth)
    if exp_args['span'] < 1 * u.MHz:
        plt.axvline(1 / t2 * 1e3 / 2, color='b', linestyle='--', label='T2 limit')
        plt.axvline(-1 / t2 * 1e3 / 2, color='b', linestyle='--')
        plt.axvline(1 / t1 * 1e3 / 2, color='k', linestyle='--', label='T1 limit')
        plt.axvline(-1 / t1 * 1e3 / 2, color='k', linestyle='--')

    plt.xlabel("Detuning (MHz)")
    plt.ylabel("State")
    # plt.ylim([-0.1, 1.1])
    plt.title(
        f'{exp_args["pulse_type"]} , eco = {exp_args["eco"]} \n pulse length = {exp_args["pulse_length"] / 1e3:.3f} us ,'
        f' pulse amplitude = {exp_args["pulse_amplitude"]} V ({qubit_spec.pulse_amp_Hz:.3f} MHz)'
        f'\n n = {exp_args["n"]} , cutoff = {exp_args["cutoff"]}')
    plt.legend()

    plt.show()
