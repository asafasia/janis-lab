#Send constant pulse and read demodulated data
#Constant readout and XY frequency, scan delay after drive
#Written by Naftali 2/21,3/22

import device_parameters
import Labber
import importlib
importlib.reload(device_parameters)
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig, LoopbackInterface
from qm.qua import *
import OPX.config_generator as config_generator
import numpy as np
from matplotlib import pyplot as plt
# import instruments_py27.anritsu as Antitsu_MG
# import instruments_py27.E8241A as E8241A_MG
from os.path import join
import os
plt.ion()

params = device_parameters.device_parameters

def set_zero(qdac_lib, qdac, QDAC_channel):
    # set DC to 0 to prevent hysteresis etc.
    qdac_lib.set_and_show_QDAC(qdac, QDAC_channel, 0.0)

save_to_Labber = True
labber_file_name = "IQM 2qubit Coherent_T1_qubit2"

labber_comment = "Coherent_T1.py. IQM 2 qubit chip. "
tags = ["IQM"]
project_name = "IQM"
user_name = "Naftali Kirsh"


#-----------Parameters-------------

debug = False #set True to measure a constant frequency pulse
simulate = False

#sweep parameters
num_delay_steps = 100#800
first_delay = 16 #ns, multiples of 4, at least 16
delay_step = 5120 #1000#ns, multiple of 4
delay_times = np.linspace(first_delay,first_delay+(num_delay_steps-1)*delay_step,num_delay_steps)


if save_to_Labber:
    lStep = [dict(name="Delay",unit="ns",values=delay_times)]
    lLog = [dict(name="Amplitude", unit="AU", vector=False), dict(name="Unwrapped Phase", unit="radian", vector=False)]
    f = Labber.createLogFile_ForData(labber_file_name, lLog, lStep)

repetitions =1000


#Local oscillator
#mg_address_readout = "GPIB0::5::INSTR"
#mg_address_drive = "GPIB0::28::INSTR"
mg_address_readout = params["mg_address_readout"]
mg_address_drive = params["mg_address_drive"]


#MG_ro = Antitsu_MG.Anritsu_MG
#MG_drive = E8241A_MG.E8241A_MG
MG_ro = params["MG_ro"]
MG_drive = params["MG_drive"]


f_r = params["f_r"] #Hz
lo_freq_readout = params["lo_freq_readout"] #Hz
if_freq_readout = lo_freq_readout-f_r #Hz, SBM frequency is lo_freq-if_freq
lo_power_readout = params["lo_power_readout"] #dBm
f_d = params["f_d"]#Hz
lo_freq_drive = params["lo_freq_drive"] #Hz
if_freq_drive = lo_freq_drive-f_d #Hz, SBM frequency is lo_freq-if_freq
lo_power_drive = params["lo_power_drive"] #dBm

#DC bias
use_QDAC = True
QDAC_port = params["QDAC_port"]
QDAC_channel = params["QDAC_channel"]
dc_bias = params["dc_bias"]



#OPX

wait_time = params["wait_time"]

#input=readout channels
I_input_channel = params["I_input_channel"]
Q_input_channel = params["Q_input_channel"]
I_input_offset = params["I_input_offset"]
Q_input_offset = params["Q_input_offset"]
#output channels
I_channel_ro = params["I_channel_ro"]
Q_channel_ro = params["Q_channel_ro"]
I_output_offset_ro = params["I_output_offset_ro"]
Q_output_offset_ro = params["Q_output_offset_ro"]
I_channel_drive = params["I_channel_drive"]
Q_channel_drive = params["Q_channel_drive"]
I_output_offset_drive = params["I_output_offset_drive"]
Q_output_offset_drive = params["Q_output_offset_drive"]
I_channel_ro_monitor = params["I_channel_ro_monitor"]
Q_channel_ro_monitor = params["Q_channel_ro_monitor"]
I_channel_drive_monitor = params["I_channel_drive_monitor"]
Q_channel_drive_monitor = params["Q_channel_drive_monitor"]

#Pulse - drive
ampl_drive = params["ampl_drive"]
pulse_length_drive = params["pulse_length_drive"] #ns
delay_step_4 = delay_step/4
first_delay_4 = first_delay/4

#Pulse - readout
pulse_length_ro = params["pulse_length_ro"]  # ns
ampl_ro = params["ampl_ro"]


running_time = repetitions*1e-9*(num_delay_steps*(4*wait_time+pulse_length_ro+pulse_length_drive)+sum(delay_times))
print("Estimated running time is %g minutes. Press Ctrl-C to stop." % ((running_time) / 60.0))


# Readout
trigger_delay = 0
trigger_length = 10
time_of_flight = params["time_of_flight"]  # ns. must be at least 28

# OPX config
cg = config_generator.ConfigGenerator(
    output_offsets={I_channel_ro: I_output_offset_ro, Q_channel_ro: Q_output_offset_ro,
                    I_channel_drive: I_output_offset_drive, Q_channel_drive: Q_output_offset_drive,
                    I_channel_ro_monitor: I_output_offset_ro,
                    Q_channel_ro_monitor: Q_output_offset_ro,
                    I_channel_drive_monitor: I_output_offset_drive,
                    Q_channel_drive_monitor: Q_output_offset_drive
                    },
    input_offsets={I_input_channel: I_input_offset, Q_input_channel: Q_input_offset})
cg.add_mixer("mixer_ro", {(lo_freq_readout, if_freq_readout): [1.0, 0.0, 0.0, 1.0]})
cg.add_mixer("mixer_drive", {(lo_freq_drive, if_freq_drive): [1.0, 0.0, 0.0, 1.0]})
cg.add_mixed_readout_element("readout", lo_freq_readout + if_freq_readout, lo_freq_readout, I_channel_ro, Q_channel_ro,
                             {"out_I": I_input_channel, "out_Q": Q_input_channel}, "mixer_ro", time_of_flight)
cg.add_mixed_input_element("readout_mon", lo_freq_readout + if_freq_readout, lo_freq_readout, I_channel_ro_monitor,
                           Q_channel_ro_monitor, "mixer_ro")
cg.add_mixed_input_element("drive", lo_freq_drive + if_freq_drive, lo_freq_drive, I_channel_drive, Q_channel_drive,
                           "mixer_drive")
cg.add_mixed_input_element("drive_mon", lo_freq_drive + if_freq_drive, lo_freq_drive, I_channel_drive_monitor,
                           Q_channel_drive_monitor, "mixer_drive")

# Output / readout
cg.add_constant_waveform("const_ro", ampl_ro)
# cg.add_arbitrary_waveform("arb_ro", arb_ro_samples)
cg.add_constant_waveform("const_drive", ampl_drive)
cg.add_constant_waveform("const_zero", 0.0)
cg.add_integration_weight("simple_cos", [1.0] * (pulse_length_ro // 4), [0.0] * (pulse_length_ro // 4))
cg.add_integration_weight("simple_sin", [0.0] * (pulse_length_ro // 4), [1.0] * (pulse_length_ro // 4))
cg.add_mixed_measurement_pulse("const_readout", pulse_length_ro, ["const_ro", "const_ro"],
                               {"simple_cos": "simple_cos", "simple_sin": "simple_sin"},
                               cg.TriggerType.RISING_TRIGGER, trigger_delay, trigger_length)
cg.add_operation("readout", "readout", "const_readout")
# cg.add_mixed_measurement_pulse("arb_readout", total_length_ro, ["arb_ro", "arb_ro"],
#                                {"simple_cos": "simple_cos", "simple_sin": "simple_sin"},
#                                cg.TriggerType.RISING_TRIGGER, trigger_delay, trigger_length)
# cg.add_operation("readout", "readout", "arb_readout")
cg.add_mixed_control_pulse("const_drive", pulse_length_drive, ["const_drive", "const_drive"])
cg.add_mixed_control_pulse("const_zero", 16, ["const_zero", "const_zero"])
cg.add_operation("drive", "drive", "const_drive")
cg.add_operation("drive", "drive_zero", "const_zero")
cg.add_operation("drive_mon", "drive", "const_drive")
cg.add_operation("drive_mon", "drive_zero", "const_zero")
cg.add_mixed_control_pulse("const_readout_mon", pulse_length_ro, ["const_ro", "const_ro"])
cg.add_operation("readout", "zero_readout", "const_zero")
cg.add_operation("readout_mon", "readout_mon", "const_readout_mon")
cg.add_operation("readout_mon", "zero_readout", "const_zero")



# OPX measurement program
with program() as prog:
    stream_II = declare_stream()
    stream_IQ = declare_stream()
    stream_QI = declare_stream()
    stream_QQ = declare_stream()
    delay_idx = declare(int)
    rep = declare(int)
    II = declare(fixed)
    IQ = declare(fixed)
    QI = declare(fixed)
    QQ = declare(fixed)

    with for_(rep, 0, rep < repetitions, rep + 1):
        with for_(delay_idx, 0, delay_idx<num_delay_steps, delay_idx+1):
            play("drive", "drive")
            play("drive", "drive_mon")
            wait(first_delay_4+delay_idx*delay_step_4)
            align("readout","drive")
            align("readout_mon", "drive_mon")
            measure("readout", "readout", None,
                    ("simple_cos", "out_I", II), ("simple_sin", "out_I", IQ),
                    ("simple_cos", "out_Q", QI), ("simple_sin", "out_Q", QQ))
            play("readout_mon", "readout_mon")
            save(II, stream_II)
            save(IQ, stream_IQ)
            save(QI, stream_QI)
            save(QQ, stream_QQ)

            if wait_time > 0:
                wait(wait_time)

    with stream_processing():
        stream_II.buffer(num_delay_steps).average().save("II")
        stream_IQ.buffer(num_delay_steps).average().save("IQ")
        stream_QI.buffer(num_delay_steps).average().save("QI")
        stream_QQ.buffer(num_delay_steps).average().save("QQ")

# ----------------main program---------------
# run
# DC bias
if not simulate:
    if use_QDAC:
        import qdac as qdac_lib

        with qdac_lib.qdac(QDAC_port) as qdac:
            # Setup QDAC
            qdac.setVoltageRange(QDAC_channel, 10)
            qdac.setCurrentRange(QDAC_channel, 1e-4)
            qdac_lib.set_and_show_QDAC(qdac, QDAC_channel, dc_bias)
    # MG
    mg_ro = MG_ro(mg_address_readout)
    mg_ro.setup_MG(lo_freq_readout / 1e6, lo_power_readout)
    mg_drive = MG_drive(mg_address_drive)
    mg_drive.setup_MG(lo_freq_drive / 1e6, lo_power_drive)

qmManager = QuantumMachinesManager()
config = cg.get_config()
# add trigger - TODO: using config_generator
config["elements"]["readout_mon"]["digitalInputs"] = {
    "trigger1":
        {
            "port": ("con1", 1),
            "delay": 144,
            "buffer": 10
        }
}
config["pulses"]["const_readout_mon"]["digital_marker"] = "trigger"
config["digital_waveforms"]["trigger"] = {"samples": [(1, 0)]}

config["controllers"]["con1"]["digital_outputs"] = {}
config["controllers"]["con1"]["digital_outputs"][1] = {}

qm = qmManager.open_qm(config)
if simulate:
    job = qmManager.simulate(config, prog,
                      SimulationConfig(int(running_time*1e9*1.5) // 4, LoopbackInterface([("con1", 1, "con1", 1), ("con1", 2, "con1", 2)])))#, include_analog_waveforms=True))
    samples = job.get_simulated_samples()
    plt.figure()
    samples.con1.plot()
    # plt.subplot(211)
    # samples.con1.plot(analog_ports=["1","2"],digital_ports=["1"])
    # plt.subplot(212)
    # samples.con1.plot(analog_ports=["3", "4"])
    plt.show()
    result_handles = job.result_handles
    result_handles.wait_for_all_values()
else:
    pending_job = qm.queue.add_to_start(prog, duration_limit=0, data_limit=0)
    # get data
    job = pending_job.wait_for_execution()
    result_handles = job.result_handles
    result_handles.wait_for_all_values()
    print("Got results")
    mg_ro.set_on(False)
    mg_drive.set_on(False)
    if use_QDAC:
        with qdac_lib.qdac(QDAC_port) as qdac:
            qdac_lib.set_and_show_QDAC(qdac, QDAC_channel, 0.0)

# analyze
II = result_handles.II.fetch_all()
QI = result_handles.QI.fetch_all()
IQ = result_handles.IQ.fetch_all()
QQ = result_handles.QQ.fetch_all()
# I_m,Q_m = negative detuning from lo_freq by if_freq
I_m = II + QQ
Q_m = IQ - QI
s = I_m + 1j * Q_m
# I_p = II - QQ
# Q_p = -QI - IQ
# s = I_p+1j*Q_p
amp_all = np.abs(s)
phase_all = np.angle(s)

if save_to_Labber:
    phase = np.unwrap(phase_all)

    data_add = {"Amplitude": np.abs(s), "Unwrapped Phase": phase}
    f.addEntry(data_add)

    labber_comment = labber_comment + str(params) + ("\n repetitions = %d" % repetitions)
    labber_comment = labber_comment + "\n" + (use_QDAC * "with ") + ((not use_QDAC) * "without ") + "DC bias."
    f.setProject(project_name)
    f.setComment(labber_comment)
    f.setTags(tags)
    f.setUser(user_name)

plt.figure(9999)
plt.plot(delay_times, amp_all/pulse_length_ro, '*-', label=("ro pulse length=%g ns, readout ampl.=%g, drive ampl.=%g, repetitions=%d" % (pulse_length_ro,ampl_ro,ampl_drive,repetitions)))
plt.legend()
plt.xlabel("Delay_times (ns)")
plt.ylabel("Mean amplitude/(readout pulse length)")
plt.figure(9998)
plt.plot(delay_times, np.unwrap(phase_all)/2/np.pi, '*-',label=("ro pulse length=%g ns, readout ampl.=%g, drive ampl..=%g, repetitions=%d" % (pulse_length_ro,ampl_ro,ampl_drive,repetitions)))
plt.xlabel("Delay_times [V]")
plt.ylabel("Mean phase/2\pi (unwrapped)")
plt.legend()
plt.figure(10000)
plt.scatter(I_m/pulse_length_ro, Q_m/pulse_length_ro,c=delay_times)
plt.colorbar(label="Delay times (ns)")
plt.xlabel("<I>/(readout pulse length)")
plt.ylabel("<Q>/(readout pulse length)")

