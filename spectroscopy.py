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
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
import matplotlib.pyplot as plt
from scipy import signal

###################
# The QUA program #
###################
f_LO = resonator_args["resonator_LO"]
n_avg = 1000  # The number of averages
# The frequency sweep parameters
f_min = 6870 * u.MHz
f_max = 6880 * u.MHz
df = 110 * u.kHz
frequencies = f_LO - np.arange(f_min, f_max + 0.1, df)  # The frequency vector (+ 0.1 to add f_max to frequencies)

with program() as resonator_spec_without_drive:
    n = declare(int)  # QUA variable for the averaging loop
    f = declare(int)  # QUA variable for the readout frequency
    I = declare(fixed)  # QUA variable for the measured 'I' quadrature
    Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature
    I_st = declare_stream()  # Stream for the 'I' quadrature
    Q_st = declare_stream()  # Stream for the 'Q' quadrature
    # n_st = declare_stream()  # Stream for the averaging iteration 'n'

    stream_II = declare_stream()
    stream_IQ = declare_stream()
    stream_QI = declare_stream()
    stream_QQ = declare_stream()

    II = declare(fixed)
    IQ = declare(fixed)
    QI = declare(fixed)
    QQ = declare(fixed)

    with for_(n, 0, n < n_avg, n + 1):  # QUA for_ loop for averaging
        with for_(*from_array(f, frequencies)):  # QUA for_ loop for sweeping the frequency

            reset_phase('resonator')  # Reset the phase of the resonator element

            update_frequency("resonator", f)
            measure("readout", "resonator", None,
                    ("cos", "out1", II), ("sin", "out1", IQ),
                    ("cos", "out2", QI), ("sin", "out2", QQ))

            wait(depletion_time * u.ns, "resonator")

            save(II, stream_II)
            save(IQ, stream_IQ)
            save(QI, stream_QI)
            save(QQ, stream_QQ)
        # Save the averaging iteration to get the progress bar
        # save(n, n_st)

    with stream_processing():
        stream_II.buffer(len(frequencies)).average().save("II")
        stream_IQ.buffer(len(frequencies)).average().save("IQ")
        stream_QI.buffer(len(frequencies)).average().save("QI")
        stream_QQ.buffer(len(frequencies)).average().save("QQ")

        #####################################
        #  Open Communication with the QOP  #
        #####################################

with program() as resonator_spec_with_drive:
    n = declare(int)  # QUA variable for the averaging loop
    f = declare(int)  # QUA variable for the readout frequency
    I = declare(fixed)  # QUA variable for the measured 'I' quadrature
    Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature
    I_st = declare_stream()  # Stream for the 'I' quadrature
    Q_st = declare_stream()  # Stream for the 'Q' quadrature
    # n_st = declare_stream()  # Stream for the averaging iteration 'n'

    stream_II = declare_stream()
    stream_IQ = declare_stream()
    stream_QI = declare_stream()
    stream_QQ = declare_stream()

    II = declare(fixed)
    IQ = declare(fixed)
    QI = declare(fixed)
    QQ = declare(fixed)

    with for_(n, 0, n < n_avg, n + 1):  # QUA for_ loop for averaging
        with for_(*from_array(f, frequencies)):  # QUA for_ loop for sweeping the frequency

            reset_phase('resonator')  # Reset the phase of the resonator element

            update_frequency("resonator", f)
            play("saturation", "qubit")
            measure("readout", "resonator", None,
                    ("cos", "out1", II), ("sin", "out1", IQ),
                    ("cos", "out2", QI), ("sin", "out2", QQ))

            wait(depletion_time * u.ns, "resonator")

            save(II, stream_II)
            save(IQ, stream_IQ)
            save(QI, stream_QI)
            save(QQ, stream_QQ)
        # Save the averaging iteration to get the progress bar
        # save(n, n_st)

    with stream_processing():
        stream_II.buffer(len(frequencies)).average().save("II")
        stream_IQ.buffer(len(frequencies)).average().save("IQ")
        stream_QI.buffer(len(frequencies)).average().save("QI")
        stream_QQ.buffer(len(frequencies)).average().save("QQ")

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
    job = qmm.simulate(config, resonator_spec_without_drive, simulation_config)
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

    II = results.II.fetch_all()
    QI = results.QI.fetch_all()
    IQ = results.IQ.fetch_all()
    QQ = results.QQ.fetch_all()

    I_m = II + QQ
    Q_m = IQ - QI

    s = I_m + 1j * Q_m

    S = u.demod2volts(s, 1000)

    R1 = np.abs(S)  # Amplitude
    phase1 = np.angle(S)  # Phase

    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(resonator_spec_with_drive)
    # Get results from QUA program
    # results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")

    results = job.result_handles
    results.wait_for_all_values()
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
    # while results.is_processing():
    # Fetch results
    # I, Q, iteration = results.fetch_all()

    II = results.II.fetch_all()
    QI = results.QI.fetch_all()
    IQ = results.IQ.fetch_all()
    QQ = results.QQ.fetch_all()

    I_m = II + QQ
    Q_m = IQ - QI

    s = I_m + 1j * Q_m

    S = u.demod2volts(s, 1000)

    R2 = np.abs(S)  # Amplitude

    res_freq = frequencies[np.argmax(abs(R1 - R2))]

    max_freq1 = f_LO - frequencies[np.argmin(R1)]
    max_freq2 = f_LO - frequencies[np.argmin(R2)]


    phase2 = np.angle(S)  # Phase

    plt.suptitle(f"Resonator spectroscopy - LO = {resonator_LO / u.GHz} GHz")
    plt.subplot(311)
    plt.axvline(x=max_freq1 / u.MHz, color='r', linestyle='--',
                label=f"Resonator frequency: {max_freq1 / u.MHz} MHz")
    plt.axvline(x=max_freq2 / u.MHz, color='r', linestyle='--',
                label=f"Resonator frequency: {max_freq1 / u.MHz} MHz")

    # plt.cla()
    plt.plot((f_LO - frequencies) / u.MHz, R1)
    plt.plot((f_LO - frequencies) / u.MHz, R2)

    plt.ylim([0, max(R1) * 1.2])

    plt.ylabel(r"$R=\sqrt{I^2 + Q^2}$ [V]")
    plt.subplot(312)
    plt.plot((f_LO - frequencies) / u.MHz, signal.detrend(np.unwrap(phase1)), )
    plt.plot((f_LO - frequencies) / u.MHz, signal.detrend(np.unwrap(phase2)), )

    plt.ylim([-np.pi / 2, np.pi / 2])
    plt.xlabel("Intermediate frequency [MHz]")
    plt.ylabel("Phase [rad]")
    plt.subplot(313)
    plt.plot((f_LO - frequencies) / u.MHz, np.abs(R1 - R2))
    plt.axvline(x=(f_LO - res_freq) / u.MHz, color='r', linestyle='--')

    plt.show()
    plt.tight_layout()

    print("Resonator freq is: ", (f_LO - res_freq)/1e6, "MHz")
