from importlib import reload
import configuration
from saver import Saver

reload(configuration)
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
    def __init__(self, qubit, n_avg, N, span, state_discrimination):
        self.qubit = qubit
        self.qmm = QuantumMachinesManager(host=qm_host, port=qm_port)
        self.span = span
        self.N = N
        self.n_avg = n_avg
        self.frequencies = qubit_LO - np.arange(qubit_freq - self.span / 2, qubit_freq + self.span / 2,
                                                self.span // self.N)
        self.detunings = qubit_freq - qubit_LO + self.frequencies
        self.state_discrimination = state_discrimination
        print("rabi_freq = ", saturation_amp / pi_pulse_amplitude / (2 * pi_pulse_length * 1e-9) / 1e6, "MHz")
        self.experiment = None

    def generate_experiment(self, pi_pulse=None):
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
                    if pi_pulse:
                        play(pi_pulse, "qubit")
                    update_frequency("qubit", df)
                    play("saturation", "qubit")
                    wait(100, "qubit")
                    align("qubit", "resonator")
                    measure(
                        "readout",
                        "resonator",
                        None,
                        dual_demod.full('cos', 'out1', 'sin', 'out2', I),
                        dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q)
                    )
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

    def execute(self):
        qm = self.qmm.open_qm(config)
        job = qm.execute(self.experiment)
        results = fetching_tool(job, data_list=["I", "Q", "state", "iteration"], mode="live")
        while results.is_processing():
            I, Q, state, iteration = results.fetch_all()
            progress_counter(iteration, self.n_avg, start_time=results.get_start_time())

        if state_measurement_stretch:
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

    def save(self):
        saver = Saver()
        measured_data = {
            'I': self.I.tolist(),
            'Q': self.Q.tolist(),
            'state': self.state.tolist(),
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
    args = {
        'qubit': 'qubit4',
        'n_avg': 1000,
        'N': 200,
        'span': 1 * u.MHz,
        'state_discrimination': True,
    }

    qubit_spec = Qubit_Spec(**args)
    qubit_spec.generate_experiment()
    qubit_spec.execute()
    qubit_spec.plot()
    qubit_spec.save()
    plt.show()

    response = input("Do you want to update qubit freq? (yes/no): ").strip().lower()
    if response == 'y':
        qubit_spec.update_max_freq()
        print("Qubit frequency updated.")
    else:
        print("Qubit frequency not updated.")
