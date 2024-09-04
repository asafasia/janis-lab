"""
        IQ BLOBS
This sequence involves measuring the state of the resonator 'N' times, first after thermalization (with the qubit
in the |g> state) and then after applying a pi pulse to the qubit (bringing the qubit to the |e> state) successively.
The resulting IQ blobs are displayed, and the data is processed to determine:
    - The rotation angle required for the integration weights, ensuring that the separation between |g> and |e> states
      aligns with the 'I' quadrature.
    - The threshold along the 'I' quadrature for effective qubit state discrimination.
    - The readout fidelity matrix, which is also influenced by the pi pulse fidelity.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the config.

Next steps before going to the next node:
    - Update the rotation angle (rotation_angle) in the configuration.
    - Update the g -> e threshold (ge_threshold) in the configuration.
"""
from matplotlib import pyplot as plt
from qm.qua import *
from qm import SimulationConfig
from qm import QuantumMachinesManager
from qualang_tools.results import fetching_tool, progress_counter

from change_args import modify_json
from configuration import *
from qualang_tools.analysis.discriminator import two_state_discriminator

###################
# The QUA program #
###################

n_runs = 20000  # Number of runs

with program() as IQ_blobs:
    n = declare(int)
    I_g = declare(fixed)
    Q_g = declare(fixed)
    I_g_st = declare_stream()
    Q_g_st = declare_stream()
    I_e = declare(fixed)
    Q_e = declare(fixed)
    I_e_st = declare_stream()
    Q_e_st = declare_stream()
    n_st = declare_stream()  # Stream for the averaging iteration 'n'

    with for_(n, 0, n < n_runs, n + 1):
        measure(
            "readout",
            "resonator",
            None,
            dual_demod.full('cos', 'out1', 'sin', 'out2', I_g),
            dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q_g)
        )

        wait(thermalization_time // 4, "resonator")

        save(I_e, I_e_st)
        save(Q_e, Q_e_st)

        align()  # global align
        play("x180", "qubit")
        # play("saturation", "qubit")

        align("qubit", "resonator")

        measure(
            "readout",
            "resonator",
            None,
            dual_demod.full('cos', 'out1', 'sin', 'out2', I_e),
            dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q_e)
        )
        wait(thermalization_time // 4, "resonator")

        save(I_g, I_g_st)
        save(Q_g, Q_g_st)

        save(n, n_st)

    with stream_processing():
        I_g_st.save_all("I_g")
        Q_g_st.save_all("Q_g")
        I_e_st.save_all("I_e")
        Q_e_st.save_all("Q_e")
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
    job = qmm.simulate(config, IQ_blobs, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    qm = qmm.open_qm(config)
    job = qm.execute(IQ_blobs)
    results = fetching_tool(job, data_list=["I_g", "Q_g", "I_e", "Q_e", "iteration"], mode="live")
    while results.is_processing():
        I_g, Q_g, I_e, Q_e, iteration = results.fetch_all()
        progress_counter(iteration, n_runs, start_time=results.get_start_time())

    plt.plot(I_g, Q_g, '.', label='ground', color='C00', alpha=0.5)
    plt.plot(I_e, Q_e, '.', label='excited', color='C03', alpha=0.2)
    plt.xlabel('I')
    plt.ylabel('Q')
    plt.legend()
    plt.title('IQ Blobs')
    angle, threshold, fidelity, gg, ge, eg, ee = two_state_discriminator(I_g, Q_g, I_e, Q_e, b_print=True, b_plot=True)
    modify_json(qubit, 'resonator', "rotation_angle", resonator_args['rotation_angle'] + angle)
    modify_json(qubit, 'resonator', "threshold", threshold)

plt.show()
