from qm import QuantumMachinesManager
from qm.qua import *
from configuration import *
from instruments_py27.spectrum_analyzer import N9010A_SA

qop_ip = None
qmm = QuantumMachinesManager(qm_host, 9510)

qm = qmm.open_qm(config)

with program() as prog:
    with infinite_loop_():
        play("readout", "resonator")


pending_job = qm.queue.add_to_start(prog)


sa = N9010A_SA(sa_address, False)

sa.setup_spectrum_analyzer(center_freq=6e3, span=100e6, BW=1e6, points=150)


