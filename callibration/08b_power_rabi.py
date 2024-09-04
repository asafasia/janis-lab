"""
        POWER RABI
The sequence consists in playing the qubit pulse (x180 or square_pi or else) and measuring the state of the resonator
for different qubit pulse amplitudes.
The experimental results are then analyzed to find the qubit pulse amplitude for the chosen duration.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated the IQ mixer connected to the qubit drive line (external mixer or Octave port)
    - Having found the rough qubit frequency and pi pulse duration (rabi_chevron_duration or time_rabi).
    - Set the qubit frequency and desired pi pulse duration (x180_len) in the configuration.

Next steps before going to the next node:
    - Update the qubit pulse amplitude (x180_amp) in the configuration.
"""
from importlib import reload

from scipy.optimize import curve_fit

import configuration

reload(configuration)
from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
import matplotlib.pyplot as plt

###################
# The QUA program #
###################

n_avg = 1000  # The number of averages
n_a = 50
amplitudes = np.linspace(0, 1, n_a)
state_discrimination = True
num_pis = 4  # Number of pi pulses in the sequence

with program() as power_rabi:
    n = declare(int)  # QUA variable for the averaging loop
    a = declare(fixed)  # QUA variable for the qubit drive amplitude pre-factor
    I = declare(fixed)  # QUA variable for the measured 'I' quadrature
    Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature
    I_st = declare_stream()  # Stream for the 'I' quadrature
    Q_st = declare_stream()  # Stream for the 'Q' quadrature
    n_st = declare_stream()  # Stream for the averaging iteration 'n'
    state = declare(bool)  # QUA variable for the qubit state
    state_st = declare_stream()  # Stream for the qubit state

    with for_(n, 0, n < n_avg, n + 1):  # QUA for_ loop for averaging
        with for_(*from_array(a, amplitudes)):  # QUA for_ loop for sweeping the pulse amplitude pre-factor
            for _ in range(num_pis):
                play("eco" * amp(a), "qubit")
            wait(200, "qubit")
            align("qubit", "resonator")
            measure(
                "readout",
                "resonator",
                None,
                dual_demod.full('cos', 'out1', 'sin', 'out2', I),
                dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q)
            )
            wait(thermalization_time // 4, "resonator")
            save(I, I_st)
            save(Q, Q_st)
            assign(state, I > ge_threshold)
            save(state, state_st)

        save(n, n_st)

    with stream_processing():
        I_st.buffer(len(amplitudes)).average().save("I")
        Q_st.buffer(len(amplitudes)).average().save("Q")
        state_st.boolean_to_int().buffer(len(amplitudes)).average().save("state")

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
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, power_rabi, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    qm = qmm.open_qm(config)
    job = qm.execute(power_rabi)
    results = fetching_tool(job, data_list=["I", "Q", "state", "iteration"], mode="live")
    while results.is_processing():
        I, Q, state, iteration = results.fetch_all()
        I, Q = u.demod2volts(I, readout_len), u.demod2volts(Q, readout_len)
        S = u.demod2volts(I + 1j * Q, readout_len)
        R = np.abs(S)  # Amplitude
        phase = np.angle(S)  # Phase
        progress_counter(iteration, n_avg, start_time=results.get_start_time())

    # %%
    x = amplitudes * pi_pulse_amplitude * num_pis
    if state_discrimination:
        y = state

    else:
        y = I

    fid_matrix = resonator_args['fidelity_matrix']
    y = state_measurement_stretch(fid_matrix, y)
    plt.plot(x * 1e3, y, '.')


    def cos_fit(x, a, b, c, d):
        return a * np.cos(2 * np.pi * 1 / b * x + np.pi) + d


    rabi_amp = qubit_args['pi_pulse_amplitude']
    args = curve_fit(cos_fit, x, y, p0=[max(y) / 2 - min(y) / 2, rabi_amp * 2, np.pi, np.mean(y)])[0]
    plt.plot(x * 1e3, cos_fit(x, *args), 'r-', label=f'fit rabi amp = {args[1] / 2:.5f} V')

    plt.suptitle("Power Rabi")
    plt.xlabel("Rabi amplitude (mV)")
    plt.ylabel("amplitude (V)")
    plt.legend()
    plt.ylim([-0.2, 1.2])

    plt.show()
    plt.ylim([0, 1.2])
    from saver import Saver

    data = {'x': x.tolist(), 'I': I.tolist(), 'Q': Q.tolist()}
    # metadata = {'n_avg': n_avg, 'num_pis': num_pis, 'args': args}

    metadata = {}
    # Save the data and metadata
    s1 = Saver()
    # s1.save("rabi_experiment", data, metadata)
