import pybamm
from pybamm.m26_study.run_intercycle_extrap import run_one_trial
from pybamm.m26_study.m26_params import get_parameter_values
import copy
import concurrent.futures


def run_parametric_sweep(param_dict):

    lengths = [len(v) for v in param_dict.values()]
    assert len(set(lengths)) == 1, "All parameter lists must be the same length"

    param = pybamm.ParameterValues(get_parameter_values())

    trial_names = param_dict["trial_name"]
    del param_dict["trial_name"]

    param_list = [copy.deepcopy(param) for _ in range(len(trial_names))]

    for param_name, values_list in param_dict.items():
        for param, val in zip(param_list, values_list):
            param.update({param_name: val})
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(run_one_trial, trial_names, param_list)


if __name__ == "__main__":
    param_dict = {
        "trial_name": [
            "trial009_100%dod_kLix0p01",
            "trial009_100%dod_kLix0p001",
        ],
        "Negative electrode porosity": [0.18, 0.18],
        "Lithium plating kinetic rate constant [m.s-1]": [1e-09 * 0.01, 1e-09 * 0.001],
        # "Outer SEI solvent diffusivity [m2.s-1]": [2.5e-22 * 2, 2.5e-22 * 5],
        # "Negative particle radius [m]": [
        #     5.86e-06 * 0.6,
        #     5.86e-06 * 0.7,
        #     5.86e-06 * 0.8,
        #     5.86e-06 * 0.9,
        # ],
        # "Lithium plating transfer coefficient": [0.6, 0.5, 0.4]
    }
    run_parametric_sweep(param_dict)
