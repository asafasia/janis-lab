from importlib import reload

import configuration

reload(configuration)
from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
import matplotlib.pyplot as plt
from macros import readout_macro

state_discrimination = True
saturation_amp = qubit_args['saturation_amplitude']
pi_amp = qubit_args['pi_pulse_amplitude']
rabi_freq = saturation_amp / pi_amp / (2 * pi_pulse_length * 1e-9)

print(f"saturation amp = {rabi_freq / 1e6:.4f} MHz")
n_avg = 1
N = 2
max_time = 3000 // 4
dt = max_time // N // 4 * 4 + 1
print(dt)

taus = np.arange(4, max_time, dt)

basis = 'z'
basis_vec = ['x', 'y', 'z']
print(taus)
with program() as qubit_spec:
    n = declare(int)  # QUA variable for the averaging loop
    df = declare(int)  # QUA variable for the qubit frequency
    I = declare(fixed)  # Q0UA variable for the measured 'I' quadrature
    Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature
    I_st = declare_stream()  # Stream for the 'I' quadrature
    Q_st = declare_stream()  # Stream for the 'Q' quadrature
    n_st = declare_stream()  # Stream for the averaging iteration 'n'
    statex = declare(bool)  # QUA variable for state discrimination
    statey = declare(bool)  # QUA variable for state discrimination
    statez = declare(bool)  # QUA variable for state discrimination

    state_stx = declare_stream()  # Stream for the qubit state
    state_sty = declare_stream()  # Stream for the qubit state
    state_stz = declare_stream()  # Stream for the qubit state

    tau = declare(int)  # QUA variable for the qubit frequency
    c = declare(int)  # QUA variable for switching between projections

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(tau, taus)):
            for b in basis_vec:
                play("-y90", "qubit2")
                wait(100, "qubit2")
                align("qubit", "qubit2")
                # the experiment:
                with if_(tau < saturation_len // 4):
                    play("saturation", "qubit", duration=tau)
                with else_():
                    play("saturation", "qubit")

                wait(100, "qubit")
                wait(tau, "resonator")
                if basis == 'x':
                    play("-y90", "qubit2")
                    align("qubit", "resonator")
                    statex, _, _ = readout_macro(threshold=ge_threshold, state=statex)
                elif basis == 'y':
                    play("x90", "qubit2")
                    align("qubit", "resonator")
                    statey, _, _ = readout_macro(threshold=ge_threshold, state=statex)
                else:
                    wait(10, "qubit")
                    align("qubit", "resonator")
                    statez, _, _ = readout_macro(threshold=ge_threshold, state=statex)

                wait(thermalization_time // 4, "resonator")
                save(statex, state_stx)

        save(n, n_st)

    with stream_processing():
        state_stx.boolean_to_int().buffer(len(taus)).average().save("statex")
        state_sty.boolean_to_int().buffer(len(taus)).average().save("statey")
        state_stz.boolean_to_int().buffer(len(taus)).average().save("statez")

        n_st.save("iteration")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qm_host, port=qm_port)

###########################
# Run or Simulate Program #
###########################
simulate = True

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, qubit_spec, simulation_config)
    job.get_simulated_samples().con1.plot()
    plt.show()

else:
    qm = qmm.open_qm(config)
    job = qm.execute(qubit_spec)
    results = fetching_tool(job, data_list=["statex", "iteration"], mode="live")
    while results.is_processing():
        statex, iteration = results.fetch_all()
        progress_counter(iteration, n_avg, start_time=results.get_start_time())

    y = state_measurement_stretch(resonator_args['fidelity_matrix'], statex)
    plt.plot(taus * 4, y)
    plt.xlabel("scan time [ns]")
    plt.ylabel("state")
    plt.axvline(x=25)
    plt.axvline(x=25 + saturation_len)
    plt.show()
