"""
        ECHO MEASUREMENT
The program consists in playing a Ramsey sequence with an echo pulse in the middle to compensate for dephasing and
enhance the coherence time (x90 - idle_time - x180 - idle_time - x90 - measurement) for different idle times.
Here the gates are on resonance so no oscillation is expected.

From the results, one can fit the exponential decay and extract T2.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the config.
    - Having the qubit frequency perfectly calibrated (ramsey).
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR.
    - Set the desired flux bias.
"""

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array, get_equivalent_log_array
import matplotlib.pyplot as plt

###################
# The QUA program #
###################
n_avg = 1000
tau_min = 4
tau_max = 20_000
N = 200
d_tau = (tau_max - tau_min) // N // 4 * 4
taus = np.arange(tau_min, tau_max + 0.1, d_tau)  # Linear sweep

with program() as echo:
    n = declare(int)
    n_st = declare_stream()
    I = declare(fixed)
    I_st = declare_stream()
    Q = declare(fixed)
    Q_st = declare_stream()
    tau = declare(int)
    state = declare(bool)  # QUA variable for state discrimination
    state_st = declare_stream()  # Stream for the qubit state

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(tau, taus // 4)):
            # 1st x90 pulse
            play("x90", "qubit")
            # Wait the varying idle time
            wait(tau, "qubit")
            # Echo pulse
            play("x180", "qubit")
            # Wait the varying idle time
            wait(tau, "qubit")
            # 2nd x90 pulse
            play("x90", "qubit")
            # Align the two elements to measure after playing the qubit pulse.
            align("qubit", "resonator")
            # Measure the state of the resonator
            measure(
                "readout",
                "resonator",
                None,
                dual_demod.full('cos', 'out1', 'sin', 'out2', I),
                dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q)
            )
            # Wait for the qubit to decay to the ground state
            wait(thermalization_time * u.ns, "resonator")
            assign(state, I > ge_threshold)
            save(state, state_st)
            save(I, I_st)
            save(Q, Q_st)
        save(n, n_st)

    with stream_processing():
        if np.isclose(np.std(taus[1:] / taus[:-1]), 0, atol=1e-3):
            taus = get_equivalent_log_array(taus)
            I_st.buffer(len(taus)).average().save("I")
            Q_st.buffer(len(taus)).average().save("Q")
        else:
            I_st.buffer(len(taus)).average().save("I")
            Q_st.buffer(len(taus)).average().save("Q")
            state_st.boolean_to_int().buffer(len(taus)).average().save("state")
            n_st.save("iteration")

######################################
#  Open Communication with the QOP  #
######################################
qmm = QuantumMachinesManager(host=qm_host, port=qm_port)

###########################
# Run or Simulate Program #
###########################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, echo, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(echo)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["I", "Q", "state", "iteration"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
    while results.is_processing():
        # Fetch results
        I, Q, state, iteration = results.fetch_all()
        # Convert the results into Volts
        I, Q = u.demod2volts(I, readout_len), u.demod2volts(Q, readout_len)
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        # Plot results
        if iteration % 100 == 0:
            print(f"iteration {iteration}/{n_avg}")
    plt.suptitle(f"Echo measurement")
    plt.plot(2 * taus, state, ".")
    plt.ylabel("state [V]")

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()

    # Fit the results to extract the qubit coherence time T2
    try:
        from qualang_tools.plot.fitting import Fit

        fit = Fit()
        plt.figure()
        T2_fit = fit.T1(2 * taus, state, plot=True)
        qubit_T2 = np.abs(T2_fit["T1"][0])
        plt.xlabel("Delay [ns]")
        plt.ylabel("I quadrature [V]")
        print(f"Qubit coherence time T2 = {qubit_T2:.0f} ns")
        plt.legend((f"Coherence time T2 = {qubit_T2:.0f} ns",))
        plt.title("Echo measurement")
    except (Exception,):
        pass

plt.show()