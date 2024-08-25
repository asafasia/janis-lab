from qm import QuantumMachinesManager
from qm.qua import *
from configuration import *
from get_traces_spectrum import plot_traces
from instruments_py27.spectrum_analyzer import N9010A_SA
from pprint import pprint

LOFreq = 6000-0.1

qop_ip = None
qmm = QuantumMachinesManager("192.168.43.137", 9510)

qm = qmm.open_qm(config)

pprint(config)

with program() as prog:
    with infinite_loop_():
        play("readout", "resonator")


pending_job = qm.queue.add_to_start(prog)


sa = N9010A_SA("TCPIP0::192.168.43.100::inst0::INSTR", False)

# sa.setup_spectrum_analyzer(center_freq=6e3, span=6e6, BW=1e6, points=5000)
sa.setup_spectrum_analyzer(center_freq=LOFreq, span=0.5e6, BW=0.1e6, points=15)
plot_traces(LOFreq, 500e6, 1e5, 5000, True)

# sa.setup_averaging(True,10)

# qm.set_output_dc_offset_by_element("resonator", "I", 0.1)

