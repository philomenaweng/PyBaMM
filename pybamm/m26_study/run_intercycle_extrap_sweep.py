import argparse
import pybamm
from pybamm.m26_study.run_intercycle_extrap import run_one_trial
from pybamm.m26_study.m26_params import get_parameter_values
from pybamm.m26_study.pybamm_sim import data_dir
import os
import shutil
import copy
import concurrent.futures


def run_parametric_sweep(param_dict, config_fname):

    lengths = [len(v) for v in param_dict.values()]
    assert len(set(lengths)) == 1, "All parameter lists must be the same length"

    param = pybamm.ParameterValues(get_parameter_values())

    trial_names = param_dict["trial_name"]
    del param_dict["trial_name"]

    for trial_name in trial_names:
        os.makedirs(f"{data_dir}/{trial_name}", exist_ok=True)
        shutil.copyfile(f"{data_dir}/{config_fname}", f"{data_dir}/{trial_name}/config.json")

    param_list = [copy.deepcopy(param) for _ in range(len(trial_names))]

    for param_name, values_list in param_dict.items():
        for param, val in zip(param_list, values_list):
            param.update({param_name: val})
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(run_one_trial, trial_names, param_list)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("config_fname", type=str, help="config file name")
    args = parser.parse_args()

    config_fname = args.config_fname

    param_dict = {
        "trial_name": [
            "trial013_80%dod_nofAx_kSEIx0p15",
            # "trial013_100%dod_nofAx_kSEIx0p2",
            # "trial013_100%dod_nofAx_kSEIx0p25",
            # "trial013_100%dod_nofAx_kSEIx0p3",
        ],
        # "Negative electrode porosity": [0.18, 0.185, 0.19, 0.195],
        # "Negative electrode critical porosity": [0.1, 0.15, 0.2, 0.25],
        # "SEI kinetic rate constant [m.s-1]": [1e-12 * 1, 1e-12 * 0.5, 1e-12 * 0.1, 1e-12 * 0.05],
        "SEI reaction exchange current density [A.m-2]": [
            1.5e-07 * 0.15,
            # 1.5e-07 * 0.2,
            # 1.5e-07 * 0.25,
            # 1.5e-07 * 0.3,
        ],
        "Electrode height [m]": [0.065],  # , 0.065, 0.065, 0.065],
        "Electrode width [m]": [1.58 * 0.49],  # , 1.58 * 0.49, 1.58 * 0.49, 1.58 * 0.49],
    }
    run_parametric_sweep(param_dict, config_fname)
