from importlib import reload
import configuration
from saver import Saver

reload(configuration)
from change_args import modify_json
from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from qualang_tools.plot import interrupt_on_close
from change_args import modify_json
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
import matplotlib.pyplot as plt
from macros import readout_macro

###################
# The QUA program #
###################
state_discrimination = True
saturation_amp = qubit_args['saturation_amplitude']
pi_amp = qubit_args['pi_pulse_amplitude']
rabi_freq = saturation_amp / pi_amp / (2 * pi_pulse_length * 1e-9)

print(f"saturation amp = {rabi_freq / 1e6:.4f} MHz")
n_avg = 1000
N = 201
span = 10 * u.MHz
df = span // N
frequencies = qubit_LO - np.arange(qubit_freq - span / 2, qubit_freq + span / 2, df)
detunings = qubit_freq - qubit_LO + frequencies

with program() as qubit_spec:
    n = declare(int)  # QUA variable for the averaging loop
    df = declare(int)  # QUA variable for the qubit frequency
    I = declare(fixed)  # QUA variable for the measured 'I' quadrature
    Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature
    I_st = declare_stream()  # Stream for the 'I' quadrature
    Q_st = declare_stream()  # Stream for the 'Q' quadrature
    n_st = declare_stream()  # Stream for the averaging iteration 'n'
    state = declare(bool)  # QUA variable for state discrimination
    state_st = declare_stream()  # Stream for the qubit state

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(df, frequencies)):
            update_frequency("qubit", df)
            play("saturation", "qubit")
            wait(100, "qubit")
            align("qubit", "resonator")
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
        I_st.buffer(len(frequencies)).average().save("I")
        Q_st.buffer(len(frequencies)).average().save("Q")
        state_st.boolean_to_int().buffer(len(frequencies)).average().save("state")
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
    qm = qmm.open_qm(config)
    job = qm.execute(qubit_spec)
    results = fetching_tool(job, data_list=["I", "Q", "state", "iteration"], mode="live")
    while results.is_processing():
        I, Q, state, iteration = results.fetch_all()
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        S = u.demod2volts(I + 1j * Q, readout_len)
        R = np.abs(S)  # Amplitude
        phase = np.angle(S)  # Phase
        max_freq = frequencies[np.argmax(I)]
        plt.pause(0.1)

    if state_discrimination:
        # y = state
        y = state_measurement_stretch(resonator_args['fidelity_matrix'], state)

    else:
        y = I

    plt.suptitle(f"Qubit spectroscopy")
    plt.title(f"drive amplitude = {saturation_amp}, drive length = {saturation_len / 1e3} us")
    plt.plot(detunings / u.MHz, y)
    plt.axvline(x=0, color='r', linestyle='--', label='Qubit frequency')
    # plt.axvline(x=qubit_LO / u.MHz - qubit_freq/u.MHz, color='g', linestyle='--', label=f'f_LO = {qubit_LO / 1e6} MHz')
    plt.axvline(x=qubit_freq / u.MHz - (qubit_LO - max_freq) / u.MHz, color='b', linestyle='--',
                label=f'resonance = {(qubit_LO - max_freq) / 1e6:.1f} MHz')

    plt.axvline(x=1 / qubit_args['T2'] * 1e3 / 2, color='k', linestyle='--',
                label=f'T2 = {1 / qubit_args["T2"] * 1e3:.2f} MHz')
    plt.axvline(x=-1 / qubit_args['T2'] * 1e3 / 2, color='k', linestyle='--')

    plt.xlabel("Detuning [MHz]")
    plt.ylabel(r"$R=\sqrt{I^2 + Q^2}$ [V]")
    # plt.ylim([(qubit_LO - frequencies)[0] / 1e6, (qubit_LO - frequencies)[-1] / 1e6])
    # plt.axvline(x=(qubit_LO - qubit_freq) / u.MHz, color='g', linestyle='--', label=f'LO = {qubit_LO}')
    plt.ylim([-0.1, 1.1])
    plt.legend()
    t1 = qubit_args['T1'] / 1e9
    t2 = qubit_args['T2'] / 1e9


    def Torrey_solution(detuning, t1, t2, rabi_freq):
        return 0.5 * (-(1 + (detuning * t2) ** 2)) / (1 + (detuning * t2) ** 2 + t1 * t2 * rabi_freq ** 2) + 1 / 2 + 0.1


    # plt.plot(detunings / 1e6, Torrey_solution(detunings, t1, t2, rabi_freq),
    #          'r-', label='Torrey solution')
    plt.show()

    # def lorentzian(x, a, b, c, d):
    #     return 0.5 * (-(1 + (x * t2) ** 2)) / (1 + (x * t2) ** 2 + t1 * t2 * c ** 2) + d
    #
    #
    # try:
    #     from scipy.optimize import curve_fit
    #
    #     args = curve_fit(lorentzian, detunings, y, p0=[1, 0, 0.1, 0])[0]
    #     plt.plot(detunings / u.MHz, lorentzian(detunings, *args), 'g-', label=f'fit f = {args[1]:.5f} MHz')
    # except:
    #     print("Failed to fit the data.")

    response = input("Do you want to update qubit freq? (yes/no): ").strip().lower()

    saver = Saver()
    x = (qubit_LO - max_freq).tolist()
    data = {'x': x, 'I': I.tolist(), 'Q': Q.tolist(), 'state': state.tolist()}
    metadata = {'n_avg': n_avg}

    saver.save("qubit_spectroscopy", data, metadata, args)

    if response == 'y':
        print("Updated the qubit frequency in the configuration file.")
        modify_json(qubit, 'qubit', "qubit_freq", qubit_LO - max_freq)
    elif response == 'n':
        print("Okay, maybe next time.")
    else:
        print("Invalid response. Please enter 'y' or 'n'.")
