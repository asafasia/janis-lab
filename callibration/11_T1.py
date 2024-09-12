from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from scipy.optimize import curve_fit
from experiment_utils.configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array, get_equivalent_log_array
import matplotlib.pyplot as plt
import experiment_utils.labber_util as lu

###################
# The QUA program #
###################
state_discrimination = True
n_avg = 2000
N = 100
tau_min = 4
tau_max = 150e3
d_tau = tau_max // N // 4 * 4
taus = np.arange(tau_min, tau_max + 0.1, d_tau)  # Linear sweep

with program() as T1:
    n = declare(int)  # QUA variable for the averaging loop
    t = declare(int)  # QUA variable for the wait time
    I = declare(fixed)  # QUA variable for the measured 'I' quadrature
    Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature
    I_st = declare_stream()  # Stream for the 'I' quadrature
    Q_st = declare_stream()  # Stream for the 'Q' quadrature
    n_st = declare_stream()  # Stream for the averaging iteration 'n'
    state = declare(bool)  # QUA variable for state discrimination
    state_st = declare_stream()  # Stream for the qubit state

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(t, taus // 4)):
            play("x180", "qubit")
            wait(t, "qubit")
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
        if np.isclose(np.std(taus[1:] / taus[:-1]), 0, atol=1e-3):
            taus = get_equivalent_log_array(taus)
            I_st.buffer(len(taus)).average().save("I")
            Q_st.buffer(len(taus)).average().save("Q")
        else:
            I_st.buffer(len(taus)).average().save("I")
            Q_st.buffer(len(taus)).average().save("Q")
            state_st.boolean_to_int().buffer(len(taus)).average().save("state")
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
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, T1, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    qm = qmm.open_qm(config)
    job = qm.execute(T1)
    results = fetching_tool(job, data_list=["I", "Q", "state", "iteration"], mode="live")
    while results.is_processing():
        I, Q, state, iteration = results.fetch_all()

        I, Q = u.demod2volts(I, readout_len), u.demod2volts(Q, readout_len)
        progress_counter(iteration, n_avg, start_time=results.get_start_time())

        S = u.demod2volts(I + 1j * Q, readout_len)
        R = np.abs(S)

    # %%
    y = state_measurement_stretch(fid_matrix, state)
    plt.plot(taus / 1e3, y, "o", label='data')


    def exp_decay(x, A, T1, C):
        return A * np.exp(-x / T1) + C


    args = curve_fit(exp_decay, taus, y, p0=[max(y) - min(y), 40e3, np.mean(I)])[0]
    plt.plot(taus / 1e3, exp_decay(taus, *args), 'r-', label=f"T1 = {args[1] / 1e3:.1f} us")
    qubit_T1 = args[1]
    print(f"Qubit decay time to update in the config: qubit_T1 = {qubit_T1 / 1e3:.1f} us")
    plt.title("T1 measurement")
    plt.legend()
    plt.xlabel("Delay [us]")
    plt.ylabel("I quadrature [V]")
    plt.ylabel("I quadrature [V]")
    plt.show()

    # %%
    import experiment_utils.labber_util as lu

    # add tags and user
    meta_data = {}

    meta_data["user"] = "Asaf"
    meta_data["n_avg"] = n_avg
    meta_data["args"] = args

    measured_data = dict(states=y)
    sweep_parameters = dict(delay=taus/1e9)
    units = dict(delay="s")

    exp_result = dict(measured_data=measured_data, sweep_parameters=sweep_parameters, units=units, meta_data=meta_data)
    lu.create_logfile("T1", **exp_result, loop_type="1d")
