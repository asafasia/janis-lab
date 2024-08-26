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
n_avg = 1000  # The number of averages
# The frequency sweep parameters
f_min = 60 * u.MHz
f_max = 85 * u.MHz
df = 110 * u.kHz
frequencies = np.arange(f_min, f_max + 0.1, df)  # The frequency vector (+ 0.1 to add f_max to frequencies)

with program() as resonator_spec:
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

            # Update the frequency of the digital oscillator linked to the resonator element
            update_frequency("resonator", f)
            # Measure the resonator (send a readout pulse and demodulate the signals to get the 'I' & 'Q' quadratures)
            # measure(
            #     "readout",
            #     "resonator",
            #     None,
            #     demod.full('cos', I, 'out1'),
            #     demod.full('sin', Q, 'out2')
            # )

            measure("readout", "resonator", None,
                    ("cos", "out1", II), ("sin", "out1", IQ),
                    ("cos", "out2", QI), ("sin", "out2", QQ))

            # measure(
            #     "readout",
            #     "resonator",
            #     None,
            #     dual_demod.full('cos', 'out1', 'sin', 'out2', I),
            #     dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q)
            # )

            # Wait for the resonator to deplete
            wait(depletion_time * u.ns, "resonator")
        # Save the 'I' & 'Q' quadratures to their respective streams
        #     save(I, I_st)
        #     save(Q, Q_st)

            save(II, stream_II)
            save(IQ, stream_IQ)
            save(QI, stream_QI)
            save(QQ, stream_QQ)
        # Save the averaging iteration to get the progress bar
        # save(n, n_st)

    with stream_processing():
        # Cast the data into a 1D vector, average the 1D vectors together and store the results on the OPX processor
        # I_st.buffer(len(frequencies)).average().save("I_st")
        # Q_st.buffer(len(frequencies)).average().save("Q_st")
        # n_st.save("iteration")
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
    job = qmm.simulate(config, resonator_spec, simulation_config)
    # Plot the simulated samples
    job.get_simulated_samples().con1.plot()

else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(resonator_spec)
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
    #
    I_m = II + QQ
    Q_m = IQ - QI

    # print(results)

    # I_m = results.I_st.fetch_all()
    # Q_m = results.Q_st.fetch_all()

    # print(I_m)
    # Convert results into Volts

    s = I_m + 1j * Q_m

    S = u.demod2volts(s, 1000)
    R = np.abs(S)  # Amplitude
    phase = np.angle(S)  # Phase
    # Progress bar
    # progress_counter(iteration, n_avg, start_time=results.get_start_time())
    # Plot results
    plt.title("I and Q as function of freq")
    plt.plot(frequencies / u.MHz, I_m, ".", label="I")
    plt.plot(frequencies / u.MHz, Q_m, ".", label="Q")
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Amplitude [V]")
    plt.legend()
    plt.show()
    plt.title("I and Q quadratures")
    plt.plot(I_m, Q_m, '-')
    plt.xlabel("I [V]")
    plt.ylabel("Q [V]")
    plt.show()
    plt.suptitle(f"Resonator spectroscopy - LO = {resonator_LO / u.GHz} GHz")
    ax1 = plt.subplot(211)
    plt.cla()
    plt.plot(frequencies / u.MHz, R, ".")
    plt.ylim([0,max(R)*1.2])

    plt.ylabel(r"$R=\sqrt{I^2 + Q^2}$ [V]")
    plt.subplot(212, sharex=ax1)
    plt.cla()
    plt.plot(frequencies / u.MHz, signal.detrend(np.unwrap(phase)), )
    plt.ylim([-np.pi/2, np.pi/2])
    plt.xlabel("Intermediate frequency [MHz]")
    plt.ylabel("Phase [rad]")
    plt.pause(0.1)
    plt.tight_layout()
    # Fit the results to extract the resonance frequency
    # try:
    #     from qualang_tools.plot.fitting import Fit
    #
    #     fit = Fit()
    #     plt.figure()
    #     res_spec_fit = fit.reflection_resonator_spectroscopy(frequencies / u.MHz, R, plot=True)
    #     plt.title(f"Resonator spectroscopy - LO = {resonator_LO / u.GHz} GHz")
    #     plt.xlabel("Intermediate frequency [MHz]")
    #     plt.ylabel(r"R=$\sqrt{I^2 + Q^2}$ [V]")
    #     print(f"Resonator resonance frequency to update in the config: resonator_IF = {res_spec_fit['f'][0]:.6f} MHz")
    # except (Exception,):
    #     pass
