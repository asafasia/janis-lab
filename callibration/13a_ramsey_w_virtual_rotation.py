from importlib import reload
import configuration

reload(configuration)
from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from scipy.optimize import curve_fit
from experiment_utils.change_args import modify_json
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
import matplotlib.pyplot as plt

###################
# The QUA program #
###################
n_avg = 5000
tau_min = 4
tau_max = 30_000 // 4
N = 200
d_tau = tau_max // N // 4 * 4

taus = np.arange(tau_min, tau_max + 0.01, d_tau)  # + 0.1 to add tau_max to taus
detuning = 0.25 * u.MHz  # in Hz
state_discrimination = True

print("qubit_freq", qubit_freq / 1e6, "MHz")

with program() as ramsey:
    n = declare(int)  # QUA variable for the averaging loop
    tau = declare(int)  # QUA variable for the idle time
    phase = declare(fixed)  # QUA variable for dephasing the second pi/2 pulse (virtual Z-rotation)
    I = declare(fixed)  # QUA variable for the measured 'I' quadrature
    Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature
    state = declare(bool)  # QUA variable for the qubit state
    I_st = declare_stream()  # Stream for the 'I' quadrature
    Q_st = declare_stream()  # Stream for the 'Q' quadrature
    state_st = declare_stream()  # Stream for the qubit state
    n_st = declare_stream()  # Stream for the averaging iteration 'n'

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(tau, taus)):
            assign(phase, Cast.mul_fixed_by_int(detuning * 1e-9, 4 * tau))
            play("x90", "qubit")
            wait(tau, "qubit")
            frame_rotation_2pi(phase, "qubit")
            play("x90", "qubit")
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
            save(I, I_st)
            save(Q, Q_st)
            save(state, state_st)
            reset_frame("qubit")
        save(n, n_st)

    with stream_processing():
        I_st.buffer(len(taus)).average().save("I")
        Q_st.buffer(len(taus)).average().save("Q")
        state_st.boolean_to_int().buffer(len(taus)).average().save("state")
        n_st.save("iteration")

qmm = QuantumMachinesManager(host=qm_host, port=qm_port)
simulate = False

if simulate:
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, ramsey, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    qm = qmm.open_qm(config)
    job = qm.execute(ramsey)
    results = fetching_tool(job, data_list=["I", "Q", "state", "iteration"], mode="live")
    while results.is_processing():
        I, Q, state, iteration = results.fetch_all()
        I, Q = u.demod2volts(I, readout_len), u.demod2volts(Q, readout_len)
        progress_counter(iteration, n_avg, start_time=results.get_start_time())

    # %%
    plt.title(f"Ramsey measurement with virtual Z rotations \n(detuning = {detuning / 1e6} MHz)")


    def exp_decay(x, A, T2, f, C):
        return A * np.exp(-x / T2) * np.cos(2 * np.pi * f * x) + C


    y = state_measurement_stretch(fid_matrix, state)

    plt.plot(taus * 4 / 1e3, y, '.', label='I')
    try:
        args = curve_fit(exp_decay, taus * 4, y, p0=[max(y) / 2 - min(y) / 2, 10000, detuning / 1e9, np.mean(I)])[0]
        plt.plot(taus * 4 / 1e3, exp_decay(taus * 4, *args),
                 label=f"T2* = {args[1] / 1e3:.2f} us, f = {args[2] * 1e3:.4} MHz")
        qubit_detuning = args[2] * u.GHz - detuning
    except RuntimeError:
        print("Fit failed.")
        qubit_detuning = 0
    print(f"Qubit detuning to update in the config: qubit_IF += {qubit_detuning / 1e6:0.4f} MHz")
    plt.legend()
    plt.xlabel("Idle time [us]")
    plt.ylabel("I quadrature [V]")
    plt.show()

    response = input("Do you want to update qubit freq? (yes/no): ").strip().lower()

    new_qubit_freq = int(qubit_freq - qubit_detuning)
    if response == 'y':
        print(f"old qubit freq: {qubit_freq} -> new qubit freq: {new_qubit_freq}")
        modify_json(qubit, 'qubit', "qubit_freq", new_qubit_freq)
    elif response == 'n':
        print("Okay, maybe next time.")
    else:
        print("Invalid response. Please enter 'y' or 'n'.")
