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

from experiment_utils.change_args import modify_json
from experiment_utils.configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
import matplotlib.pyplot as plt
from scipy import signal

###################
# The QUA program #
###################
n_avg = 1000  # The number of averages
span = 10 * u.MHz
f_min = resonator_freq - span / 2
f_max = resonator_freq + span / 2
df = 100 * u.kHz
long_pulse = False
simulate = False

frequencies = resonator_LO - np.arange(f_min, f_max + 0.1, df)

with program() as resonator_spec:
    n = declare(int)  # QUA variable for the averaging loop
    f = declare(int)  # QUA variable for the readout frequency
    I1 = declare(fixed)  # QUA variable for the measured 'I' quadrature
    Q1 = declare(fixed)  # QUA variable for the measured 'Q' quadrature\
    I2 = declare(fixed)  # QUA variable for the measured 'I' quadrature
    Q2 = declare(fixed)  # QUA variable for the measured 'Q' quadrature
    I_st1 = declare_stream()  # Stream for the 'I' quadrature
    Q_st1 = declare_stream()  # Stream for the 'Q' quadrature
    I_st2 = declare_stream()  # Stream for the 'I' quadrature
    Q_st2 = declare_stream()  # Stream for the 'Q' quadrature
    n_st = declare_stream()  # Stream for the averaging iteration 'n'

    with for_(n, 0, n < n_avg, n + 1):  # QUA for_ loop for averaging
        with for_(*from_array(f, frequencies)):  # QUA for_ loop for sweeping the frequency
            update_frequency("resonator", f)
            measure(
                "readout",
                "resonator",
                None,
                dual_demod.full('cos', 'out1', 'sin', 'out2', I1),
                dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q1)
            )
            wait(thermalization_time // 4, "resonator")
            if long_pulse:
                play("saturation", "qubit")
                play("saturation", "qubit2")

            else:
                play("x180", "qubit")
            wait(100, "qubit")
            align("qubit", "resonator")
            measure(
                "readout",
                "resonator",
                None,
                dual_demod.full('cos', 'out1', 'sin', 'out2', I2),
                dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q2)
            )
            save(I1, I_st1)
            save(Q1, Q_st1)
            save(I2, I_st2)
            save(Q2, Q_st2)
            wait(thermalization_time // 4, "resonator")

        save(n, n_st)

    with stream_processing():
        I_st1.buffer(len(frequencies)).buffer(n_avg).save("I1")
        Q_st1.buffer(len(frequencies)).buffer(n_avg).save("Q1")
        I_st2.buffer(len(frequencies)).buffer(n_avg).save("I2")
        Q_st2.buffer(len(frequencies)).buffer(n_avg).save("Q2")
        n_st.save("iteration")

#####################################
#  Open Communication with the QOP  #
#####################################

qmm = QuantumMachinesManager(host=qm_host, port=qm_port)

#######################
# Simulate or execute #
#######################

if simulate:
    simulation_config = SimulationConfig(duration=10_00)  # In clock cycles = 4ns
    job = qmm.simulate(config, resonator_spec, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    qm = qmm.open_qm(config)
    job = qm.execute(resonator_spec)
    results = fetching_tool(job, data_list=["I1", "Q1", "I2", "Q2", "iteration"], mode="live")

    while results.is_processing():
        I1, Q1, I2, Q2, iteration = results.fetch_all()

        S1 = u.demod2volts(I1 + 1j * Q1, readout_len)
        S2 = u.demod2volts(I2 + 1j * Q2, readout_len)
        R1 = np.abs(S1)  # Amplitude
        R2 = np.abs(S2)  # Amplitude

        R1_var = np.var(R1, axis=0)
        R2_var = np.var(R2, axis=0)

        Var = np.sqrt(R1_var / 2 + R2_var / 2)

        R1 = np.mean(R1, axis=0)
        R2 = np.mean(R2, axis=0)
        phase1 = np.angle(S1)  # Phase
        phase2 = np.angle(S2)  # Phase

        phase1 = signal.detrend(np.unwrap(phase1))
        phase2 = signal.detrend(np.unwrap(phase2))
        phase1 = np.mean(phase1, axis=0)
        phase2 = np.mean(phase2, axis=0)
        progress_counter(iteration, n_avg, start_time=results.get_start_time())

    fig = plt.figure()
    diff = np.abs(R1 - R2)/Var
    res_freq = frequencies[np.argmax(diff)]
    print("Resonator  freq is: ", (resonator_LO - res_freq) / 1e6, "MHz")
    ground = R1[np.argmax(diff)]
    excited = R2[np.argmax(diff)]
    print('ground state amplitude: ', ground)
    print('excited state amplitude: ', excited)

    plt.suptitle(
        f"Resonator spectroscopy - LO = {resonator_LO / u.GHz} GHz "
        f"\nRESONATOR : pulse amplitude = {readout_amp} , "
        f" length = {readout_len / 1e3} us"
        f"\n QUBIT : pulse amplitude = {saturation_amp}, length = {saturation_len / 1e3} us"
    )
    plt.subplot(311)
    plt.axvline(x=(resonator_LO - res_freq) / u.MHz, color='r', linestyle='--')
    plt.axvline(x=resonator_freq / u.MHz, color='g', linestyle='--')
    plt.plot((resonator_LO - frequencies) / u.MHz, R1, label='Without Drive')
    plt.plot((resonator_LO - frequencies) / u.MHz, R2, label='With Drive')
    plt.legend()
    #
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
    plt.ylabel("Diff ")
    plt.plot((resonator_LO - frequencies) / u.MHz, diff)
    plt.axvline(x=(resonator_LO - res_freq) / u.MHz, color='r', linestyle='--',
                label=f'new freq = {(resonator_LO - res_freq) / u.MHz} MHz')
    plt.axvline(x=resonator_freq / u.MHz, color='g', linestyle='--',
                label=f'old freq = {(resonator_freq) / u.MHz} MHz')
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

plt.show()
