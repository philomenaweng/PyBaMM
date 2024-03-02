import pybamm

from pybamm.m26_study.m26_params import get_parameter_values
from pybamm.m26_study.pybamm_sim import pybamm_sim

if __name__ == "__main__":
    trial_name = "003_100%dod"
    c_rate = "C/4"
    toc_v = 4.2
    bod_v = 3

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
                f"Hold at {toc_v} V until C/100",
                "Rest for 1 hour",
                f"Discharge at {c_rate} until {bod_v} V",
                "Rest for 1 hour",
            )
        ]
        * 1
    )

    param = pybamm.ParameterValues(get_parameter_values())

    param.update(
        {
            # "Initial outer SEI thickness [m]": 5e-9,
            "Negative electrode porosity": 0.18,
            # "Lithium plating kinetic rate constant [m.s-1]": 1e-9,
        }
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

    while run_flag and sim.n_total_cycles[-1] <= 9999:
        try:
            sim.extrapolate_states(n_delta=500)
            sim.run(3.6)
        except Exception as e:
            print(e)
            run_flag = False
            break

    sim.save_data()
