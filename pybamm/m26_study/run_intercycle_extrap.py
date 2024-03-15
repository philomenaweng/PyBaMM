import pybamm
import argparse
import json

from pybamm.m26_study.m26_params import get_parameter_values
from pybamm.m26_study.pybamm_sim import pybamm_sim, data_dir


def run_one_trial(trial_name, param):
    print(f"Running trial {trial_name}")
    with open(f"{data_dir}/{trial_name}/config.json", "r") as f:
        config = json.load(f)

    config["trial_name"] = trial_name
    c_rate = config["c_rate"]
    toc_v = config["toc_v"]
    bod_v = config["bod_v"]
    max_cycles = config["max_cycles"]

    with open(f"{data_dir}/{trial_name}/config.json", "w") as f:
        json.dump(config, f, indent=4)

    exp = pybamm.Experiment(
        [
            (
                f"Discharge at {c_rate} until {bod_v} V",
                "Rest for 1 hour",
            )
        ]
        + [
            (
                f"Charge at {c_rate} until {toc_v} V",
                f"Hold at {toc_v} V until C/10",
                "Rest for 1 hour",
                f"Discharge at {c_rate} until {bod_v} V",
                "Rest for 1 hour",
            )
        ]
        * 1
    )

    options = {
        "SEI": "solvent-diffusion limited",
        "SEI porosity change": "true",
        "lithium plating": "irreversible",
        "lithium plating porosity change": "true",
        # "particle mechanics": ("swelling and cracking", "swelling only"),
        # "SEI on cracks": "true",
        # "loss of active material": "stress-driven",
        # "calculate discharge energy": "true",  # for compatibility with older PyBaMM versions
    }

    sim = pybamm_sim(options, param, exp, version=trial_name)
    sim.run(3.6)
    run_flag = True
    for i in range(10):
        try:
            sim.extrapolate_states(n_delta=5)
            sim.run(3.6)
        except Exception as e:
            print(e)
            run_flag = False
            break
    sim.save_data()

    while run_flag and sim.n_total_cycles[-1] <= max_cycles:
        try:
            sim.extrapolate_states(n_delta=500)
            sim.run(3.6)
        except Exception as e:
            print(e)
            run_flag = False
            break

        if len(sim.n_total_cycles) % 10 == 0:
            sim.save_data()

    sim.save_data()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trial_name", type=str, help="Trial name")
    args = parser.parse_args()

    trial_name = args.trial_name

    param = pybamm.ParameterValues(get_parameter_values())
    param.update(
        {
            "Negative electrode porosity": 0.18,
            "Outer SEI solvent diffusivity [m2.s-1]": 2.5e-22,
            "Negative particle radius [m]": 5.86e-06,
        }
    )

    run_one_trial(trial_name, param)
