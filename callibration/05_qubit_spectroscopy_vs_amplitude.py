from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from experiment_utils.configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
import matplotlib.pyplot as plt

###################
# The QUA program #
###################
n_avg = 1000
pulse_length = 3e3

N = 201
span = 10 * u.MHz
df = span // N
a_min = 0.1
a_max = 1.1
frequencies = qubit_LO - np.arange(qubit_freq - span / 2, qubit_freq + span / 2, df)
amplitudes = np.linspace(a_min, a_max, 2)

with program() as resonator_spec_2D:
    n = declare(int)  # QUA variable for the averaging loop
    df = declare(int)  # QUA variable for the readout frequency
    a = declare(fixed)  # QUA variable for the readout amplitude pre-factor
    I = declare(fixed)  # QUA variable for the measured 'I' quadrature
    Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature
    I_st = declare_stream()  # Stream for the 'I' quadrature
    Q_st = declare_stream()  # Stream for the 'Q' quadrature
    n_st = declare_stream()  # Stream for the averaging iteration 'n'
    state = declare(bool)  # QUA variable for state discrimination
    state_st = declare_stream()  # Stream for the qubit state

    with for_(n, 0, n < n_avg, n + 1):  # QUA for_ loop for averaging
        with for_(*from_array(df, frequencies)):  # QUA for_ loop for sweeping the frequency

            update_frequency("qubit", df)
            with for_each_(a, amplitudes):
                play("x90", "qubit2")
                wait(10, "qubit2")
                align()
                play("saturation" * amp(a), "qubit")
                wait(100, "qubit")
                align("qubit", "resonator")  # Save the 'I' & 'Q' quadratures to their respective streams
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
        I_st.buffer(len(amplitudes)).buffer(len(frequencies)).average().save("I")
        Q_st.buffer(len(amplitudes)).buffer(len(frequencies)).average().save("Q")
        state_st.boolean_to_int().buffer(len(amplitudes)).buffer(len(frequencies)).average().save("state")
        n_st.save("iteration")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qm_host, port=qm_port)

#######################
# Simulate or execute #
#######################
simulate = False

if simulate:
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, resonator_spec_2D, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    qm = qmm.open_qm(config)
    job = qm.execute(resonator_spec_2D)
    results = fetching_tool(job, data_list=["I", "Q", "state", "iteration"], mode="live")
    fig = plt.figure()
    interrupt_on_close(fig, job)
    while results.is_processing():
        I, Q, state, iteration = results.fetch_all()
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        S = u.demod2volts(I + 1j * Q, readout_len)
        R = np.abs(S)  # Amplitude
        phase = np.angle(S)  # Phase
        row_sums = R.sum(axis=0)
        R /= row_sums[np.newaxis, :]
        plt.pause(0.1)

    y = state

    plt.suptitle(f"Qubit Spectroscopy vs Amplitude")
    plt.title(r"$R=\sqrt{I^2 + Q^2}$ (normalized)")
    plt.pcolor((qubit_LO - frequencies) / u.MHz, amplitudes * saturation_amp, y.T)
    plt.ylabel("drive amplitude [V]")
    plt.xlabel("drive freq [MHz]")
    plt.colorbar()
    plt.tight_layout()
    plt.axvline(x=qubit_freq / u.MHz, color='r', linestyle='--', label='Qubit frequency')
    plt.axvline(x=qubit_LO / u.MHz, color='r', linestyle='--', label='Qubit frequency')
    plt.xlim([(qubit_LO - frequencies)[0] / u.MHz, (qubit_LO - frequencies)[-1] / u.MHz])
    plt.legend()
    plt.show()
    for y0 in y.T:
        plt.plot((qubit_LO - frequencies) / u.MHz, y0)
    plt.xlabel("drive freq [MHz]")
    plt.ylabel("amplitude [V]")
    plt.axvline(x=qubit_freq / u.MHz, color='r', linestyle='--', label='Qubit frequency')
    plt.axvline(x=qubit_LO / u.MHz, color='r', linestyle='--', label='Qubit frequency')
    plt.xlim([(qubit_LO - frequencies)[0] / u.MHz, (qubit_LO - frequencies)[-1] / u.MHz])

    plt.show()

# saver = Saver()
# data = {'freqs': 1, 'amps': 2, 'I': I, 'Q': Q, 'state': state}
# metadata = {'qubit': 'q1', 'version': '1.0'}
