from qm import QuantumMachinesManager
from qm.qua import *
from configuration import *


def open_qm():
    qm = QuantumMachinesManager(qm_host, 9510)

    return qm.open_qm({
        "version": 1,
        "controllers": {
            "con1": {
                "type": "opx1",
                "analog_outputs": {
                    8: {"offset": 0.1},
                }
            }
        },
        "elements": {
            "RR1": {
                "singleInput": {
                    "port": ("con1", 8)
                },
                "intermediate_frequency": 0.0,
                "operations": {
                    "pulse": "my_pulse"
                }
            },

        },
        "pulses": {
            "my_pulse": {
                "operation": "control",
                "length": 2000,
                "waveforms": {
                    "single": "zero_wave"
                }
            }
        },
        "waveforms": {
            "zero_wave": {
                "type": "constant",
                "sample": 0.0
            }
        }
    })


with program() as prog:
    with infinite_loop_():
        play("pulse", "RR1")

qm = open_qm()
job = qm.execute(prog)
