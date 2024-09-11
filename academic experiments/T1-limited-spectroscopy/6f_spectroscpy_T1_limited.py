from importlib import reload
import configuration
from saver import Saver

reload(configuration)
from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
import matplotlib.pyplot as plt

###################
# The QUA program #
###################
state_discrimination = True
saturation_amp = qubit_args['saturation_amplitude']
pi_amp = qubit_args['pi_pulse_amplitude']
rabi_freq = saturation_amp / pi_amp / (2 * pi_pulse_length * 1e-9)

print(f"saturation amp = {rabi_freq / 1e6:.4f} MHz")
n_avg = 2000
N = 500
span = 3 * u.MHz
df = span // N
frequencies = qubit_LO - np.arange(qubit_freq - span / 2, qubit_freq + span / 2, df)
detunings = qubit_freq - qubit_LO + frequencies

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

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(df, frequencies)):
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
        I_st.buffer(len(frequencies)).average().save("I")
        Q_st.buffer(len(frequencies)).average().save("Q")
        state_st.boolean_to_int().buffer(len(frequencies)).average().save("state")
        n_st.save("iteration")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qm_host, port=qm_port)

###########################
# Run or Simulate Program #
###########################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, qubit_spec, simulation_config)
    job.get_simulated_samples().con1.plot()
    plt.show()

else:
    qm = qmm.open_qm(config)
    job = qm.execute(qubit_spec)
    results = fetching_tool(job, data_list=["I", "Q", "state", "iteration"], mode="live")
    while results.is_processing():
        I, Q, state, iteration = results.fetch_all()
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        S = u.demod2volts(I + 1j * Q, readout_len)
        R = np.abs(S)  # Amplitude
        phase = np.angle(S)  # Phase
        max_freq = frequencies[np.argmax(I)]
        plt.pause(0.1)

    if state_discrimination:
        # y = state
        y = state_measurement_stretch(resonator_args['fidelity_matrix'], state)

    else:
        y = I

    # %%

    plt.suptitle(f"Qubit spectroscopy")
    plt.title(
        f"drive amplitude = {rabi_freq / 1e6:.4f} MHz ({saturation_amp * 1e3} mV), drive length = {saturation_len / 1e3} us")
    plt.plot(detunings / u.MHz, y)
    plt.ylabel(r"state")
    plt.ylim([-0.1, 1.1])
    plt.legend()
    t1 = qubit_args['T1'] / 1e9
    t2 = qubit_args['T2'] / 1e9


    def Torrey_solution(detuning, t1, t2, rabi_freq):
        return 0.5 * (-(1 + (detuning * t2) ** 2)) / (1 + (detuning * t2) ** 2 + t1 * t2 * rabi_freq ** 2) + 1 / 2 + 0.1


    plt.plot(detunings / 1e6, Torrey_solution(detunings, t1, t2, rabi_freq),
             'r-', label="Torrey's solution")
    plt.legend()
    plt.show()

saver = Saver()
measured_data = {
    'I': I,
    'Q': Q,
    'state': state,
}
sweep = {
    'detunings': detunings,
    'frequencies': frequencies,
}
metadata = {
    'n_avg': n_avg,
    'rabi_freq': rabi_freq,
    'span': span
}
saver.save('T1_limit_spectroscopy', measured_data, sweep, metadata, args)

# class Qubit_Spec:
#     def __init__(self):
#         self.qmm = QuantumMachinesManager(host=qm_host, port=qm_port)
#         self.qm = self.qmm.open_qm(config)
#         self.job = self.qm.execute(qubit_spec)
#         self.results = fetching_tool(self.job, data_list=["I", "Q", "state", "iteration"], mode="live")
#
#     def
