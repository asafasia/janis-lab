import numpy as np
import labber_util as lu

if __name__ == "__main__":
    result_X = np.array([1, 1, 1])
    delay_vec = np.array([1, 2, 3])
    meta_data = {}
    meta_data["tags"] = ["Nadav-Lab", "spin-locking", "overnight"]
    meta_data["user"] = "Guy"
    measured_data = dict(X=result_X)
    sweep_parameters = dict(hold_time=delay_vec)
    units = dict(hold_time="s", detuning="Hz")
    exp_result = dict(measured_data=measured_data, sweep_parameters=sweep_parameters, units=units, meta_data=meta_data)

    lu.create_logfile("spin_locking", **exp_result, loop_type="1d")

    lu.get_log_name()