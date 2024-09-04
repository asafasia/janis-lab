from matplotlib import pyplot as plt
from qm.qua import *
from qm import QuantumMachinesManager

from configuration import *
from qualang_tools.results import fetching_tool

N = 9
n_avg = 1000
state_discrimination = True

with program() as resonator_spec_2D:
    n = declare(int)  # QUA variable for the averaging loop
    i = declare(int)  # QUA variable for the iteration loop
    df = declare(int)  # QUA variable for the readout frequency
    I = declare(fixed)  # QUA variable for the measured 'I' quadrature
    Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature
    n_st = declare_stream()  # Stream for the averaging iteration 'n'
    state = declare(bool)  # QUA variable for state discrimination
    state_st = declare_stream()
    I_st = declare_stream()
    Q_st = declare_stream()

    with for_(n, 0, n < n_avg, n + 1):  # QUA for_ loop for averaging
        with for_(i, 0, i < N, i + 1):
            # play("x180", "qubit")
            measure(
                "readout",
                "resonator",
                None,
                dual_demod.full('cos', 'out1', 'sin', 'out2', I),
                dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q)
            )
            assign(state, I > threshold)
            save(state, state_st)
            save(I, I_st)
    with stream_processing():
        state_st.buffer(n_avg).buffer(N).save("state")
        I_st.buffer(n_avg).buffer(N).save("I")

qmm = QuantumMachinesManager(host=qm_host, port=qm_port)
qm = qmm.open_qm(config)
job = qm.execute(resonator_spec_2D)
results = fetching_tool(job, data_list=["state", "I"], mode="live")

states, I = results.fetch_all()

states = np.mean(states, axis=1)

plt.plot(states)
plt.ylim([0, 1])
plt.show()
