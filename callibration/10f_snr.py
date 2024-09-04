"""
        READOUT OPTIMISATION: FREQUENCY
This sequence involves measuring the state of the resonator in two scenarios: first, after thermalization
(with the qubit in the |g> state) and then after applying a pi pulse to the qubit (transitioning the qubit to the
|e> state). This is done while varying the readout frequency.
The average I & Q quadratures for the qubit states |g> and |e>, along with their variances, are extracted to
determine the Signal-to-Noise Ratio (SNR). The readout frequency that yields the highest SNR is selected as the
optimal choice.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the config.

Next steps before going to the next node:
    - Update the readout frequency (resonator_IF) in the configuration.
"""
import numpy as np
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
n_avg = 200  # The number of averages
df = 500 * u.kHz
span = 20 * u.MHz
f_min = resonator_freq - span / 2
f_max = resonator_freq + span / 2

frequencies = resonator_LO - np.arange(f_min, f_max + 0.1, df)
amplitudes = np.linspace(0, 1.5, 20)

with program() as ro_freq_opt:
    n = declare(int)  # QUA variable for the averaging loop
    a = declare(fixed)  # QUA variable for the qubit drive amplitude pre-factor
    df = declare(int)  # QUA variable for the readout frequency
    I_g = declare(fixed)  # QUA variable for the 'I' quadrature when the qubit is in |g>
    Q_g = declare(fixed)  # QUA variable for the 'Q' quadrature when the qubit is in |g>
    Ig_st = declare_stream()
    Qg_st = declare_stream()
    I_e = declare(fixed)  # QUA variable for the 'I' quadrature when the qubit is in |e>
    Q_e = declare(fixed)  # QUA variable for the 'Q' quadrature when the qubit is in |e>
    Ie_st = declare_stream()
    Qe_st = declare_stream()
    n_st = declare_stream()

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(a, amplitudes)):  # QUA for_ loop for sweeping the pulse amplitude
            with for_(*from_array(df, frequencies)):
                # Update the frequency of the digital oscillator linked to the resonator element
                update_frequency("resonator", df)
                # Measure the state of the resonator
                measure(
                    "readout" * amp(a),
                    "resonator",
                    None,
                    dual_demod.full('cos', 'out1', 'sin', 'out2', I_g),
                    dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q_g)
                )
                # Wait for the qubit to decay to the ground state
                wait(thermalization_time // 4, "resonator")
                # Save the 'I_e' & 'Q_e' quadratures to their respective streams
                save(I_g, Ig_st)
                save(Q_g, Qg_st)

                align()  # global align
                # Play the x180 gate to put the qubit in the excited state
                play("x180", "qubit")
                # Align the two elements to measure after playing the qubit pulse.
                align("qubit", "resonator")
                # Measure the state of the resonator
                measure(
                    "readout" * amp(a),
                    "resonator",
                    None,
                    dual_demod.full('cos', 'out1', 'sin', 'out2', I_e),
                    dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q_e)
                )
                # Wait for the qubit to decay to the ground state
                wait(thermalization_time // 4, "resonator")
                # Save the 'I_e' & 'Q_e' quadratures to their respective streams
                save(I_e, Ie_st)
                save(Q_e, Qe_st)
        # Save the averaging iteration to get the progress bar
        save(n, n_st)

    with stream_processing():
        n_st.save("iteration")
        # mean values
        Ig_st.buffer(len(frequencies)).buffer(len(amplitudes)).buffer(n_avg).save("Ig")
        Qg_st.buffer(len(frequencies)).buffer(len(amplitudes)).buffer(n_avg).save("Qg")
        Ie_st.buffer(len(frequencies)).buffer(len(amplitudes)).buffer(n_avg).save("Ie")
        Qe_st.buffer(len(frequencies)).buffer(len(amplitudes)).buffer(n_avg).save("Qe")
        # variances to get the SNR

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
    job = qmm.simulate(config, ro_freq_opt, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(ro_freq_opt)  # execute QUA program
    # Get results from QUA program
    results = fetching_tool(
        job,
        data_list=["Ig", "Qg", "Ie", "Qe", "iteration"],
        mode="live",
    )
    # Live plotting
    while results.is_processing():
        # Fetch results
        Ig, Qg, Ie, Qe, iteration = results.fetch_all()
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())

        Ze = Ie + 1j * Qe
        Zg = Ig + 1j * Qg

        Ig_avg = np.mean(Ig, axis=0)
        Qg_avg = np.mean(Qg, axis=0)
        Ie_avg = np.mean(Ie, axis=0)
        Qe_avg = np.mean(Qe, axis=0)

        Ig_var = np.var(Ig, axis=0)
        Qg_var = np.var(Qg, axis=0)
        Ie_var = np.var(Ie, axis=0)
        Qe_var = np.var(Qe, axis=0)

        Z = (Ie_avg - Ig_avg) + 1j * (Qe_avg - Qg_avg)
        var = (Ig_var + Qg_var + Ie_var + Qe_var) / 4
        SNR = ((np.abs(Z)) ** 2) / (2 * var)

    # print(Ig_avg)
    for snr in SNR:
        plt.plot(resonator_LO / u.MHz - frequencies / u.MHz, snr, ".-")

    plt.show()
    plt.pcolor((resonator_LO - frequencies) / u.MHz, amplitudes * readout_amp, SNR, shading='auto')
    plt.colorbar()
    plt.title(f"readout pulse length = {readout_len / 1e3} us")
    print(
        f"The optimal readout frequency is {frequencies[np.argmax(SNR)] / u.MHz} MHz\n and the optimal amplitude is {amplitudes[np.argmax(SNR)]}")
    plt.show()
