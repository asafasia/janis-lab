"""
        TIME RABI
The sequence consists in playing the qubit pulse (x180 or square_pi or else) and measuring the state of the resonator
for different qubit pulse durations.
The results are then post-processed to find the qubit pulse duration for the chosen amplitude.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated the IQ mixer connected to the qubit drive line (external mixer or Octave port)
    - Having found the rough qubit frequency and pi pulse amplitude (rabi_chevron_amplitude or power_rabi).
    - Set the qubit frequency and desired pi pulse amplitude (x180_amp) in the configuration.

Next steps before going to the next node:
    - Update the qubit pulse duration (x180_len) in the configuration.
"""
from importlib import reload

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from experiment_utils.configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
import matplotlib.pyplot as plt
import experiment_utils.labber_util as lu

###################
# The QUA program #
###################

n_avg = 1000  # The number of averages
N = 100
# Pulse duration sweep (in clock cycles = 4ns) - must be larger than 4 clock cycles
t_min = 16 // 4
t_max = 30000 // 4
dt = (t_max - t_min) // N // 4
durations = np.arange(t_min, t_max, dt)
amplitude = 0.03

with program() as time_rabi:
    n = declare(int)  # QUA variable for the averaging loop
    t = declare(int)  # QUA variable for the qubit pulse duration
    I = declare(fixed)  # QUA variable for the measured 'I' quadrature
    Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature
    I_st = declare_stream()  # Stream for the 'I' quadrature
    Q_st = declare_stream()  # Stream for the 'Q' quadrature
    n_st = declare_stream()  # Stream for the averaging iteration 'n'
    state = declare(bool)  # QUA variable for the qubit state
    state_st = declare_stream()  # Stream for the qubit state

    with for_(n, 0, n < n_avg, n + 1):  # QUA for_ loop for averaging
        with for_(*from_array(t, durations)):  # QUA for_ loop for sweeping the pulse duration
            # Play the qubit pulse with a variable duration (in clock cycles = 4ns)
            play("x180" * amp(amplitude), "qubit", duration=t)

            # Align the two elements to measure after playing the qubit pulse.
            align("qubit", "resonator")
            # Measure the state of the resonator
            # The integration weights have changed to maximize the SNR after having calibrated the IQ blobs.
            measure(
                "readout",
                "resonator",
                None,
                dual_demod.full('cos', 'out1', 'sin', 'out2', I),
                dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q)
            )
            # Wait for the qubit to decay to the ground state
            wait(thermalization_time // 4, "resonator")
            save(I, I_st)
            save(Q, Q_st)
            assign(state, I > ge_threshold)
            save(state, state_st)

        # Save the averaging iteration to get the progress bar
        save(n, n_st)

    with stream_processing():
        I_st.buffer(len(durations)).average().save("I")
        Q_st.buffer(len(durations)).average().save("Q")
        state_st.boolean_to_int().buffer(len(durations)).average().save("state")
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
    job = qmm.simulate(config, time_rabi, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    qm = qmm.open_qm(config)
    job = qm.execute(time_rabi)
    results = fetching_tool(job, data_list=["I", "Q", "state", "iteration"], mode="live")
    while results.is_processing():
        I, Q, state, iteration = results.fetch_all()
        I, Q = u.demod2volts(I, readout_len), u.demod2volts(Q, readout_len)
        S = u.demod2volts(I + 1j * Q, readout_len)
        R = np.abs(S)  # Amplitude
        phase = np.angle(S)  # Phase
        progress_counter(iteration, n_avg, start_time=results.get_start_time())

    y = state
    fid_matrix = resonator_args['fidelity_matrix']
    y = state_measurement_stretch(fid_matrix, y)
    # Plot results
    plt.suptitle("Time Rabi")
    plt.cla()
    plt.plot(4 * durations, y)
    plt.ylabel("I quadrature [V]")
    plt.tight_layout()
    plt.show()
    # Fit the results to extract the x180 length
    try:
        from qualang_tools.plot.fitting import Fit

        fit = Fit()
        plt.figure()
        rabi_fit = fit.rabi(4 * durations, I, plot=True)
        plt.title(f"Time Rabi")
        plt.xlabel("Rabi pulse duration [ns]")
        plt.ylabel("I quadrature [V]")
        print(f"Optimal x180_len = {round(1 / rabi_fit['f'][0] / 2 / 4) * 4} ns for {square_pi_amp:} V")
    except (Exception,):
        pass

    # %%

    # add tags and user
    meta_data = {}

    meta_data["user"] = "Asaf"
    meta_data["n_avg"] = {
        "n_avg": n_avg,
        "amplitude": amplitude * square_pi_amp,
    }
    meta_data["args"] = args

    measured_data = dict(states=y)
    sweep_parameters = dict(delay=durations * 4 / 1e9)
    units = dict(delay="s")

    exp_result = dict(measured_data=measured_data, sweep_parameters=sweep_parameters, units=units, meta_data=meta_data)
    lu.create_logfile("time-rabi", **exp_result, loop_type="1d")
