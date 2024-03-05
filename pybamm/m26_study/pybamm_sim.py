import os
import pickle
import numpy as np
import pandas as pd
import pybamm

data_dir = "/local/data/philomenaweng/projects/degradation/pybamm_modeling/solutions"


class pybamm_sim:
    """
    Class representing a PyBaMM simulation.

    Args:
        options (dict): Options for the simulation.
        parameters (dict): Parameters for the simulation.
        experiment (pybamm.Experiment): Experiment to simulate.
        version (str, optional): Version of the simulation. Defaults to "000".

    Attributes:
        model (pybamm.BaseModel): PyBaMM model used for the simulation.
        parameter_values (pybamm.ParameterValues): Parameter values for the simulation.
        experiment (pybamm.Experiment): Experiment to simulate.
        solution (list): List of solutions obtained from the simulation.
        sim_dfs (list): List of pandas DataFrames containing simulation data.
        parameters_list (list): List of parameter values used in each simulation.
        n_total_cycles (list): List of total cycle numbers.
        n_extrapolated (int): Number of cycles extrapolated.
        sods (list): List of state of discharge [total moles of lithium, _, _] values.
        fname (str): File name for saving simulation data.

    Methods:
        solve(start_voltage: float = 3.2): Solve the simulation.
        create_output(): Create output data from the simulation.
        agg_steps(): Aggregate simulation steps.
        extract_sod(): Extract state of discharge values.
        extrapolate_states(n_delta: int = 10): Extrapolate states for the simulation.
        _find_max_ndelta_from_state(var_name: str, yvar_max: np.array, yvar_min: np.array): Find
            the maximum number of cycles to extrapolate based on a state variable.

    """

    def __init__(
        self,
        options,
        parameters,
        experiment,
        var_pts={"x_n": 100, "x_s": 10, "x_p": 10, "r_n": 10, "r_p": 10},
        version="000",
        save=True,
    ):
        self.model = pybamm.lithium_ion.DFN(options=options)
        self.parameter_values = parameters.copy()
        self.experiment = experiment
        self.var_pts = var_pts
        self.sim_dfs = []
        self.sim_steps_df = []
        self.parameters_list = []
        self.n_total_cycles = [0]
        self.n_extrapolated = 0
        self.state_variables = []
        self.fname = f"{data_dir}/trial{version}"
        self.save_flag = save
        if self.save_flag:
            os.makedirs(self.fname, exist_ok=False)
            with open(f"{self.fname}/options.pkl", "wb") as f:
                pickle.dump(options, f)
            with open(f"{self.fname}/experiment.pkl", "wb") as f:
                pickle.dump(experiment, f)
            with open(f"{self.fname}/parameters.pkl", "wb") as f:
                pickle.dump(parameters, f)
            with open(f"{self.fname}/var_pts.pkl", "wb") as f:
                pickle.dump(var_pts, f)

    def solve(self, start_voltage: float = 3.6):
        self.sim = pybamm.Simulation(
            self.model,
            parameter_values=self.parameter_values,
            experiment=self.experiment,
            solver=pybamm.CasadiSolver(return_solution_if_failed_early=False, dt_max=0.1),
            var_pts=self.var_pts,
        )
        self.parameters_list.append(self.parameter_values.copy())
        n_cycle = int(self.n_total_cycles[-1] + self.n_extrapolated)
        solution = self.sim.solve(
            showprogress=False, initial_soc=f"{start_voltage} V", calc_esoh=False
        )
        self.solution = solution
        self.n_total_cycles.append(int(n_cycle))

    def save(self):
        n_cycle = self.n_total_cycles[-1]
        # self.sim.save(f"{self.fname}/cycle{str(n_cycle).zfill(4)}_sim.pkl")

        with open(f"{self.fname}/cycle{str(n_cycle).zfill(4)}_params.pkl", "wb") as f:
            pickle.dump(self.parameter_values, f)

        # sim_obj = pybamm.Simulation(
        #     model=self.model,
        #     parameter_values=self.parameter_values,
        #     var_pts={"x_n": 50, "x_s": 10, "x_p": 10, "r_n": 10, "r_p": 10},
        # )
        # sim_obj.build()
        # sim_obj.save_model(
        #     f"{self.fname}/cycle{str(n_cycle).zfill(4)}_model", mesh=True, variables=True
        # )

        # with open(f"{self.fname}/cycle{str(n_cycle).zfill(4)}_sol.pkl", "wb") as f:
        #     pickle.dump(solution, f)

    def create_output(self):
        sim_df = pd.DataFrame()
        soln = self.solution
        sim_df["total_time_s"] = soln["Time [s]"].entries
        sim_df["voltage"] = soln["Terminal voltage [V]"].entries
        sim_df["current"] = soln["Current [A]"].entries
        sim_df["capacity_Ah"] = -soln["Discharge capacity [A.h]"].entries

        sim_df["step_type"] = sim_df["current"].apply(
            lambda x: "charge" if x < 0 else "discharge" if x > 0 else "rest"
        )
        sim_df["current_sign"] = sim_df["step_type"].apply(
            lambda x: -1 if x == "charge" else 1 if x == "discharge" else 0
        )

        sim_df["step_change"] = sim_df["step_type"].ne(sim_df["step_type"].shift(1))
        sim_df["step_number"] = sim_df["step_change"].cumsum()
        sim_df["cycle_number"] = self.n_total_cycles[-1]
        self.sim_dfs.append(sim_df)

    def agg_steps(self):
        sim_df = self.sim_dfs[-1]
        sim_steps_df_first = sim_df.groupby("step_number").first()
        sim_steps_df_last = sim_df.groupby("step_number").last()
        sim_steps_df = sim_steps_df_first.join(sim_steps_df_last, lsuffix="_first", rsuffix="_last")
        sim_steps_df["step_time_s"] = (
            sim_steps_df["total_time_s_last"] - sim_steps_df["total_time_s_first"]
        )
        sim_steps_df["step_capacity_Ah"] = (
            sim_steps_df["capacity_Ah_last"] - sim_steps_df["capacity_Ah_first"]
        )
        sim_steps_df["cycle_number"] = self.n_total_cycles[-1]
        self.sim_steps_df.append(sim_steps_df)

    def check_if_completed(self):
        """
        Check if the simulation is completed.

        Returns:
            bool: True if the simulation is completed, False otherwise.

        """
        steps_df = self.sim_steps_df[-1]
        exp_steps = self.experiment.operating_conditions_steps

        n_steps = len(list(filter(lambda x: not x.description.startswith("Hold"), exp_steps)))
        assert len(steps_df) == n_steps, "Simulation did not complete all steps."

        if steps_df["step_type_first"].iloc[-1] == "rest":
            assert (
                steps_df["step_time_s"].iloc[-1] >= exp_steps[-1].duration - 11
            ), "Simulation did not complete last step."
        else:
            raise ValueError("Simulation did not end with a rest step.")

    def track_states(self):
        """
        Track battery state at beginning of cycle
        """
        soln = self.solution
        n_lithium = soln["Total lithium in particles [mol]"].entries[0]
        am_neg = soln["Negative electrode active material volume fraction"].entries[0].mean()
        am_pos = soln["Positive electrode active material volume fraction"].entries[0].mean()
        min_porosity_neg = soln["Negative electrode porosity"].entries[0, :].min()
        sei_neg = soln["Negative total SEI thickness [m]"].entries[0, :].mean()
        li_neg = (
            soln["X-averaged negative lithium plating concentration [mol.m-3]"].entries[0].mean()
        )

        self.state_variables.append(
            [
                am_pos,
                am_neg,
                n_lithium,
                min_porosity_neg,
                sei_neg,
                li_neg,
            ]
        )

    def extrapolate_states(self, n_delta: int = 100):
        # change in negative electrode porosity, positive/negative electrode concentration,
        # positive/negative active material fraction

        extrap_vars = {
            "Average negative particle concentration [mol.m-3]": (
                0,
                self.parameter_values["Maximum concentration in negative electrode [mol.m-3]"],
                "Initial concentration in negative electrode [mol.m-3]",
                0,
            ),
            "Average positive particle concentration [mol.m-3]": (
                0,
                self.parameter_values["Maximum concentration in positive electrode [mol.m-3]"],
                "Initial concentration in positive electrode [mol.m-3]",
                0,
            ),
            "Negative electrode porosity": (
                0,
                self.parameter_values["Negative electrode porosity"],
                "Negative electrode porosity",
                1,
            ),
            "Negative outer SEI thickness [m]": (0, 1e8, "Initial outer SEI thickness [m]", 0),
            "X-averaged negative lithium plating concentration [mol.m-3]": (
                0,
                1e8,
                "Initial plated lithium concentration [mol.m-3]",
                0,
            ),
        }
        soln = self.solution
        for var_name, (yvar_min, yvar_max, param_name, ndim) in extrap_vars.items():
            yvar_start, yvar_delta, n_delta_max = self._find_max_ndelta_from_state(
                var_name, yvar_max, yvar_min
            )

            n_delta = int(min(n_delta, n_delta_max / 2))
            extrap_vars[var_name] = (yvar_start, yvar_delta, param_name, ndim, n_delta_max)

        for var_name, (yvar_start, yvar_delta, param_name, ndim, _) in extrap_vars.items():
            yvar_new = yvar_start + n_delta * yvar_delta

            if ndim == 0:
                self.parameter_values.update({param_name: np.average(yvar_new.reshape(-1))})
            elif ndim == 1:

                if "negative" in var_name.lower():
                    xarr = soln["x_n [m]"].entries[:, 0]
                elif "positive" in var_name.lower():
                    xarr = soln["x_p [m]"].entries[:, 0]
                else:
                    raise ValueError("variable not implemented")

                self.parameter_values.update(
                    {
                        param_name: pybamm.Interpolant(
                            xarr, yvar_new, pybamm.standard_spatial_vars.x_n, extrapolate=False
                        )
                    }
                )

            else:
                raise ValueError("variable dimension not implemented")
        if n_delta <= 0:
            n_delta = 1
        smallest_var = min(extrap_vars, key=lambda x: extrap_vars[x][-1])
        print(f"Extrapolated {n_delta} cycles. Smallest n_delta from {smallest_var}.")

        self.n_extrapolated = n_delta
        return extrap_vars

    def _find_max_ndelta_from_state(self, var_name: str, yvar_max: np.array, yvar_min: np.array):
        """
        Find the maximum number of cycles to extrapolate based on a state variable.

        Args:
            var_name (str): Name of the state variable.
            yvar_max (np.array): Maximum value of the state variable.
            yvar_min (np.array): Minimum value of the state variable.

        Returns:
            tuple: Tuple containing the start value, delta value, and maximum number of cycles.

        """
        soln = self.solution
        yvar = soln[var_name].entries

        steps_df = self.sim_steps_df[-1]
        time_start = steps_df.loc[steps_df["step_type_first"] == "rest", "total_time_s_last"].iloc[
            0
        ]
        idx_start = np.where(soln["Time [s]"].entries == time_start)[0][0]

        if yvar.ndim == 1:
            yvar_start = yvar[idx_start]
            yvar_end = yvar[-1]

        elif yvar.ndim == 2:
            yvar_start = yvar[:, idx_start]
            yvar_end = yvar[:, -1]

        else:
            raise ValueError("variable dimension not implemented")

        yvar_delta = yvar_end - yvar_start
        if np.all(yvar_delta > 0):
            n_delta_max = np.floor((yvar_max - yvar_start) / yvar_delta).min()
        else:
            n_delta_max = np.floor((yvar_min - yvar_start) / yvar_delta).min()

        return yvar_start, yvar_delta, n_delta_max

    def run(self, start_voltage: float = 3.6):
        """
        Run the simulation.

        Args:
            start_voltage (float, optional): Starting voltage for the simulation. Defaults to 3.2.

        """
        self.solve(start_voltage=start_voltage)
        self.create_output()
        self.agg_steps()
        self.check_if_completed()
        self.track_states()
        if self.save_flag:
            self.save()

    def save_data(self):
        """
        Save simulation data to files.

        """
        sim_df = pd.concat(self.sim_dfs)
        steps_df = pd.concat(self.sim_steps_df)
        sod_df = pd.DataFrame(
            self.state_variables,
            columns=[
                "am_pos",
                "am_neg",
                "n_lithium_mol",
                "min_porosity_neg",
                "sei_neg_m",
                "li_neg_M",
            ],
        )
        sod_df["cycle_number"] = self.n_total_cycles[1:]
        save_df = {
            "sim_data": sim_df,
            "steps_data": steps_df,
            "sod_data": sod_df,
        }
        with open(f"{self.fname}/sim_data.pkl", "wb") as f:
            pickle.dump(save_df, f)
