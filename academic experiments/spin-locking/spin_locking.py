# %% imports
import sys
# sys.path.append('C:\\Users\\owner\\miniconda3\\envs\\janis-lab-env\\lib\\site-packages\\Labber\\_include39')
# sys.path.append('C:\\Users\\owner\\miniconda3\\envs\\janis-lab-env\\lib\\site-packages\\Labber\\_include39\\labber\\config\\')
# sys.path.append('C:\\Users\\owner\\miniconda3\\envs\\janis-lab-env\\lib\\site-packages')

from qm.qua import *
from experiment_utils.configuration import *
import matplotlib.pyplot as plt
from experiments_objects.qubit_spectroscopy import Qubit_Spec
from qm import QuantumMachinesManager
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
import matplotlib.pyplot as plt
# %%
if __name__ == "__main__":
    n_avg = 1000  # The number of averages
    N = 10
    # Pulse duration sweep (in clock cycles = 4ns) - must be larger than 4 clock cycles
    t_min = 0 // 4
    t_max = 300 // 4
    dt = (t_max - t_min) // N // 4
    durations = np.arange(t_min, t_max, dt)
    amplitude = 0.03
    
    
    qmm = QuantumMachinesManager(host=qm_host, port=qm_port)  # creates a manager instance
    qm = qmm.open_qm(config)  # opens a quantum machine with the specified configuration
    
    with program() as spin_lock:
        n = declare(int)  # QUA variable for the averaging loop
        I = declare(fixed)  # QUA variable for the measured 'I' quadrature
        Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature
        t = declare(int) # QUA variable for the pulse time
        state = declare(bool)  # QUA variable for state discrimination
        state_st = declare_stream()  # Stream for the qubit state
        I_st = declare_stream()  # Stream for the 'I' quadrature
        Q_st = declare_stream()  # Stream for the 'Q' quadrature
        n_st = declare_stream()  # Stream for the averaging iteration 'n'
        t_St = declare_stream()
        
        with for_(n, 0, n < n_avg, n + 1):
            with for_(*from_array(t,durations)):
                align("qubit", "resonator")
                play('x90', "qubit", duration=t)
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
              I_st.buffer(len(durations)).average().save("I")
              Q_st.buffer(len(durations)).average().save("Q")
              state_st.boolean_to_int().buffer(len(durations)).average().save("state")
              n_st.save("iteration")


    job = qm.execute(spin_lock)
    results = fetching_tool(job, data_list=["I", "Q", "state", "iteration"], mode="live")
    while results.is_processing():
        I, Q, state, iteration = results.fetch_all()
        I, Q = u.demod2volts(I, readout_len), u.demod2volts(Q, readout_len)
        S = u.demod2volts(I + 1j * Q, readout_len)
        R = np.abs(S)  # Amplitude
        phase = np.angle(S)  # Phase
        progress_counter(iteration,n_avg, start_time=results.get_start_time())
        
    
    plt.plot(4 * durations, state)

    
        
    
    