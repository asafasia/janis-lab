from importlib import reload

import configuration

reload(configuration)

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
import matplotlib.pyplot as plt

###################
# The QUA program #
###################

n_avg = 5200  # The number of averages
state_discrimination = True
n_pis = 20  # Number of pi pulses in the sequence

with program() as power_rabi:
    n = declare(int)  # QUA variable for the averaging loop
    i = declare(int)  # QUA variable for the number of pi pulses
    I = declare(fixed)  # QUA variable for the measured 'I' quadrature
    Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature
    I_st = declare_stream()  # Stream for the 'I' quadrature
    Q_st = declare_stream()  # Stream for the 'Q' quadrature
    n_st = declare_stream()  # Stream for the averaging iteration 'n'
    state = declare(bool)  # QUA variable for the qubit state
    state_st = declare_stream()  # Stream for the qubit state
    t = declare(int)  # QUA variable for the number of pi pulses
    with for_(n, 0, n < n_avg, n + 1):  # QUA for_ loop for averaging
        with for_(i, 0, i < n_pis, i + 1):  # QUA for_ loop for sweeping the number of pi pulses

            with for_(t, 0, t < i, t + 1):  # QUA for_ loop for sweeping the number of pi pulses
                align("qubit", "resonator")
                play("x90", "qubit")
                # wait(40, "resonator")

            measure(
                "readout",
                "resonator",
                None,
                dual_demod.full('cos', 'out1', 'sin', 'out2', I),
                dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q)
            )
            wait(thermalization_time // 4, "resonator")
            save(I, I_st)
            save(Q, Q_st)
            assign(state, I > ge_threshold)
            save(state, state_st)

        save(n, n_st)

    with stream_processing():
        I_st.buffer(n_pis).average().save("I")
        Q_st.buffer(n_pis).average().save("Q")
        state_st.boolean_to_int().buffer(n_pis).average().save("state")
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
    job = qmm.simulate(config, power_rabi, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    qm = qmm.open_qm(config)
    job = qm.execute(power_rabi)
    results = fetching_tool(job, data_list=["I", "Q", "state", "iteration"], mode="live")
    while results.is_processing():
        I, Q, state, iteration = results.fetch_all()
        I, Q = u.demod2volts(I, readout_len), u.demod2volts(Q, readout_len)
        S = u.demod2volts(I + 1j * Q, readout_len)
        R = np.abs(S)  # Amplitude
        phase = np.angle(S)  # Phase
        progress_counter(iteration, n_avg, start_time=results.get_start_time())

    # %%
    print('s')
    plt.title("X Pulse rep experiment")
    y = state
    plt.plot(range(n_pis), y, '-')
    plt.plot(range(n_pis), y, 'r.', label=f'pi pulse = {qubit_args["pi_pulse_amplitude"]}')
    plt.plot(range(1, n_pis - 1, 2), y[1:-2:2], 'g-')
    plt.xlabel("Number of pi pulses")
    plt.ylabel("state")
    plt.legend()

    plt.show()
