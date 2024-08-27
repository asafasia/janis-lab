"""
        QUBIT SPECTROSCOPY
This sequence involves sending a saturation pulse to the qubit, placing it in a mixed state,
and then measuring the state of the resonator across various qubit drive intermediate dfs.
In order to facilitate the qubit search, the qubit pulse duration and amplitude can be changed manually in the QUA
program directly without having to modify the configuration.

The data is post-processed to determine the qubit resonance frequency, which can then be used to adjust
the qubit intermediate frequency in the configuration under "qubit_IF".

Note that it can happen that the qubit is excited by the image sideband or LO leakage instead of the desired sideband.
This is why calibrating the qubit mixer is highly recommended.

This step can be repeated using the "x180" operation instead of "saturation" to adjust the pulse parameters (amplitude,
duration, frequency) before performing the next calibration steps.

Prerequisites:
    - Identification of the resonator's resonance frequency when coupled to the qubit in question (referred to as "resonator_spectroscopy").
    - Calibration of the IQ mixer connected to the qubit drive line (whether it's an external mixer or an Octave port).
    - Configuration of the saturation pulse amplitude and duration to transition the qubit into a mixed state.
    - Specification of the expected qubit T1 in the configuration.

Before proceeding to the next node:
    - Update the qubit frequency, labeled as "qubit_IF", in the configuration.
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

###################
# The QUA program #
###################
f_LO = qubit_args["qubit_LO"]
center = qubit_args["qubit_freq"]
saturation_len = qubit_args['saturation_length']
saturation_amp = qubit_args['saturation_amplitude']

n_avg = 1000
span = 20 * u.MHz
df = 200 * u.kHz

frequencies = f_LO - np.arange(center - span / 2, center + span / 2, df)

with program() as qubit_spec:
    n = declare(int)  # QUA variable for the averaging loop
    df = declare(int)  # QUA variable for the qubit frequency
    I = declare(fixed)  # QUA variable for the measured 'I' quadrature
    Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature
    I_st = declare_stream()  # Stream for the 'I' quadrature
    Q_st = declare_stream()  # Stream for the 'Q' quadrature
    n_st = declare_stream()  # Stream for the averaging iteration 'n'

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(df, frequencies)):
            update_frequency("qubit", df)
            play("saturation", "qubit")
            measure(
                "readout",
                "resonator",
                None,
                dual_demod.full('cos', 'out1', 'sin', 'out2', I),
                dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q)
            )
            wait(thermalization_time // 4, "resonator")
            align("qubit", "resonator")
            save(I, I_st)
            save(Q, Q_st)
        save(n, n_st)

    with stream_processing():
        I_st.buffer(len(frequencies)).average().save("I")
        Q_st.buffer(len(frequencies)).average().save("Q")
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
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(qubit_spec)

    results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")
    while results.is_processing():
        I, Q, iteration = results.fetch_all()

        S = u.demod2volts(I + 1j * Q, resonator_args['readout_pulse_length'])
        R = np.abs(S)  # Amplitude
        phase = np.angle(S)  # Phase
        progress_counter(iteration, n_avg, start_time=results.get_start_time())

    plt.suptitle(f"Qubit spectroscopy")

    max_freq = frequencies[np.argmax(R)]

    plt.title(f"drive amplitude = {saturation_amp} - drive duration = {saturation_len / 1e3}us")
    plt.plot((f_LO - frequencies) / u.MHz, R)
    plt.axvline(x=center / u.MHz, color='r', linestyle='--', label='Qubit frequency')
    plt.axvline(x=f_LO / u.MHz, color='g', linestyle='--', label=f'f_LO = {f_LO / 1e6} MHz')
    plt.axvline(x=(f_LO - max_freq) / u.MHz, color='b', linestyle='--',
                label=f'resonance = {(f_LO - max_freq) / 1e6} MHz')
    plt.xlabel("Qubit frequency [MHz]")
    plt.ylabel(r"$R=\sqrt{I^2 + Q^2}$ [V]")
    plt.xlim([(f_LO - frequencies)[0] / 1e6, (f_LO - frequencies)[-1] / 1e6])

    plt.legend()
    # plt.ylim([0.01, 0.035])
    plt.show()

    response = input("Do you want to update qubit freq? (yes/no): ").strip().lower()

    if response == 'y':
        print("Updated the resonator frequency in the configuration file.")
        modify_json(qubit, 'qubit', "qubit_freq", f_LO - max_freq)
    elif response == 'n':
        print("Okay, maybe next time.")
    else:
        print("Invalid response. Please enter 'y' or 'n'.")
