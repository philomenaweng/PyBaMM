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
        shutil.copyfile(
            f"{data_dir}/{config_fname}", f"{data_dir}/{trial_name}/config.json"
        )

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
    trial_v = "016"
    dod_case = 80
    param_dict = {
        "trial_name": [
            f"trial{trial_v}_{dod_case}%dod_Axmin0p5_LiMcrit40",
            f"trial{trial_v}_{dod_case}%dod_Axmin0p5_LiMcrit50",
            f"trial{trial_v}_{dod_case}%dod_Axmin0p5_LiMcrit60",
        ],
        # "Negative electrode porosity": [0.3, 0.35],
        # "Negative electrode critical porosity": [0.15, 0.15, 0.15],
        # "Negative electrode minimum fraction": [0.5, 0.5, 0.5],
        "Ax minimum fraction": [0.5, 0.5, 0.5],
        "Critical plated lithium concentration [mol.m-3]": [40, 50, 60],
        # "SEI reaction exchange current density [A.m-2]": [1.5e-07 for i in range(3)],
        # "Outer SEI solvent diffusivity [m2.s-1]": [
        #     2.5e-22 * 10,
        #     2.5e-22 * 10 * 0.5,
        #     2.5e-22 * 10 * 0.1,
        # ],
        # "SEI kinetic rate constant [m.s-1]": [
        #     7.5e-17 * 0.8,
        #     7.5e-17 * 0.6,
        #     7.5e-17 * 0.4,
        # ],
        # "SEI open-circuit potential [V]": [0.39, 0.38, 0.37],
        # "Lithium plating kinetic rate constant [m.s-1]": [1e-12 * 0.1, 1e-12 * 0.01],
    }

    # n_trials = len(param_dict["trial_name"])
    # param_dict.update(
    #     {
    #         "Electrode height [m]": [0.065 for _ in range(n_trials)],
    #         "Electrode width [m]": [1.58 * 0.49 for _ in range(n_trials)],
    #         "Negative electrode active material volume fraction": [0.75 for _ in range(n_trials)],
    #     }
    # )
    run_parametric_sweep(param_dict, config_fname)
