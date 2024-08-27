"""
        RESONATOR SPECTROSCOPY
This sequence involves measuring the resonator by sending a readout pulse and demodulating the signals to extract the
'I' and 'Q' quadratures across varying readout intermediate frequencies.
The data is then post-processed to determine the resonator resonance frequency.
This frequency can be used to update the readout intermediate frequency in the configuration under "resonator_IF".

Prerequisites:
    - Ensure calibration of the time of flight, offsets, and gains (referenced as "time_of_flight").
    - Calibrate the IQ mixer connected to the readout line (whether it's an external mixer or an Octave port).
    - Define the readout pulse amplitude and duration in the configuration.
    - Specify the expected resonator depletion time in the configuration.

Before proceeding to the next node:
    - Update the readout frequency, labeled as "resonator_IF", in the configuration.
"""

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig

from change_args import modify_json
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
import matplotlib.pyplot as plt
from scipy import signal

###################
# The QUA program #
###################
center = resonator_freq
n_avg = 500  # The number of averages
# The frequency sweep parameters
span = 10 * u.MHz
f_min = center - span / 2
f_max = center + span / 2
df = 50 * u.kHz

frequencies = resonator_LO - np.arange(f_min, f_max + 0.1, df)

with program() as resonator_spec_without_drive:
    n = declare(int)  # QUA variable for the averaging loop
    f = declare(int)  # QUA variable for the readout frequency
    I = declare(fixed)  # QUA variable for the measured 'I' quadrature
    Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature
    I_st = declare_stream()  # Stream for the 'I' quadrature
    Q_st = declare_stream()  # Stream for the 'Q' quadrature

    with for_(n, 0, n < n_avg, n + 1):  # QUA for_ loop for averaging
        with for_(*from_array(f, frequencies)):  # QUA for_ loop for sweeping the frequency
            reset_phase('resonator')  # Reset the phase of the resonator element
            update_frequency("resonator", f)
            measure(
                "readout",
                "resonator",
                None,
                dual_demod.full('cos', 'out1', 'sin', 'out2', I),
                dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q)
            )
            save(I, I_st)
            save(Q, Q_st)
            wait(depletion_time, "resonator")

    with stream_processing():
        I_st.buffer(len(frequencies)).average().save("I")
        Q_st.buffer(len(frequencies)).average().save("Q")

with program() as resonator_spec_with_drive:
    n = declare(int)  # QUA variable for the averaging loop
    f = declare(int)  # QUA variable for the readout frequency
    I = declare(fixed)  # QUA variable for the measured 'I' quadrature
    Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature
    I_st = declare_stream()  # Stream for the 'I' quadrature
    Q_st = declare_stream()  # Stream for the 'Q' quadrature

    with for_(n, 0, n < n_avg, n + 1):  # QUA for_ loop for averaging
        with for_(*from_array(f, frequencies)):  # QUA for_ loop for sweeping the frequency
            update_frequency("resonator", f)
            play("saturation", "qubit")
            align("qubit", "resonator")
            measure(
                "readout",
                "resonator",
                None,
                dual_demod.full('cos', 'out1', 'sin', 'out2', I),
                dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q)
            )
            save(I, I_st)
            save(Q, Q_st)
            wait(thermalization_time, "resonator")
    with stream_processing():
        I_st.buffer(len(frequencies)).average().save("I")
        Q_st.buffer(len(frequencies)).average().save("Q")

#####################################
#  Open Communication with the QOP  #
#####################################

qmm = QuantumMachinesManager(host=qm_host, port=qm_port)

#######################
# Simulate or execute #
#######################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, resonator_spec_with_drive, simulation_config)
    # Plot the simulated samples
    job.get_simulated_samples().con1.plot()

else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(resonator_spec_without_drive)
    # Get results from QUA program
    # results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")

    results = job.result_handles
    results.wait_for_all_values()

    I_m = results.I.fetch_all()
    Q_m = results.Q.fetch_all()

    s = I_m + 1j * Q_m

    S = u.demod2volts(s, resonator_args['readout_pulse_length'])

    R1 = np.abs(S)  # Amplitude
    phase1 = np.angle(S)  # Phase
    phase1 = signal.detrend(np.unwrap(phase1))

    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(resonator_spec_with_drive)
    # Get results from QUA program
    # results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")

    # results.wait_for_all_values()
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
    # while results.is_processing():
    # Fetch results
    results = job.result_handles
    results.wait_for_all_values()

    I_m = results.I.fetch_all()
    Q_m = results.Q.fetch_all()

    s = I_m + 1j * Q_m

    S = u.demod2volts(s, resonator_args['readout_pulse_length'])
    phase2 = np.angle(S)  # Phase
    phase2 = signal.detrend(np.unwrap(phase2))
    R2 = np.abs(S)  # Amplitude

    diff = np.abs(R1 - R2)
    # diff = np.abs(phase1 - phase2)

    res_freq = frequencies[np.argmax(diff)]
    print("Resonator  freq is: ", (resonator_LO - res_freq) / 1e6, "MHz")

    ground = R1[np.argmax(diff)]
    excited = R2[np.argmax(diff)]
    print('ground state amplitude: ', ground)
    print('excited state amplitude: ', excited)

    plt.suptitle(f"Resonator spectroscopy - LO = {resonator_LO / u.GHz} GHz")
    plt.subplot(311)
    plt.title(f'resonator amp = {readout_pulse_amplitude}')
    plt.axvline(x=(resonator_LO - res_freq) / u.MHz, color='r', linestyle='--')
    plt.axvline(x=center / u.MHz, color='g', linestyle='--')

    # plt.cla()
    plt.plot((resonator_LO - frequencies) / u.MHz, R1, label='Without Drive')
    plt.plot((resonator_LO - frequencies) / u.MHz, R2, label='With Drive')
    plt.legend()

    plt.ylim([0, max(R1) * 1.2])

    plt.ylabel(r"$R=\sqrt{I^2 + Q^2}$ [V]")
    plt.subplot(312)
    plt.plot((resonator_LO - frequencies) / u.MHz, phase1, label='Without Drive')
    plt.plot((resonator_LO - frequencies) / u.MHz, phase2, label='With Drive')

    # plt.ylim([-np.pi / 2, np.pi / 2])
    plt.xlabel("Intermediate frequency [MHz]")
    plt.ylabel("Phase [rad]")
    plt.legend()
    plt.subplot(313)
    plt.ylabel("Diff [V]")
    plt.plot((resonator_LO - frequencies) / u.MHz, diff)
    plt.axvline(x=(resonator_LO - res_freq) / u.MHz, color='r', linestyle='--',
                label=f'new freq = {(resonator_LO - res_freq) / u.MHz} MHz')
    plt.axvline(x=center / u.MHz, color='g', linestyle='--',
                label=f'old freq = {(center) / u.MHz} MHz')
    plt.legend()
    plt.tight_layout()
    plt.show()

    response = input("Do you want to update resonator frequency? (yes/no): ").strip().lower()

    if response == 'y':
        print("Updated the resonator frequency in the configuration file.")
        modify_json(qubit, 'resonator', "resonator_freq", resonator_LO - res_freq)
    elif response == 'n':
        print("Okay, maybe next time.")
    else:
        print("Invalid response. Please enter 'y' or 'n'.")
