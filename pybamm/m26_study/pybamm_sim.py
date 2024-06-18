import pickle
import numpy as np
import pandas as pd
import pybamm
from sklearn.metrics import root_mean_squared_error as rmse
from scipy.optimize import minimize
import concurrent.futures
from tqdm import tqdm

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
        experiment=None,
        var_pts={"x_n": 20, "x_s": 10, "x_p": 10, "r_n": 10, "r_p": 10},
        version="trial000",
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
        self.fname = f"{data_dir}/{version}"
        self.save_flag = save
        if self.save_flag:
            with open(f"{self.fname}/options.pkl", "wb") as f:
                pickle.dump(options, f)
            with open(f"{self.fname}/experiment.pkl", "wb") as f:
                pickle.dump(experiment, f)
            with open(f"{self.fname}/parameters.pkl", "wb") as f:
                pickle.dump(parameters, f)
            with open(f"{self.fname}/var_pts.pkl", "wb") as f:
                pickle.dump(var_pts, f)

    def solve(self, start_voltage: float = 3.6, new_params=None):

        submesh_types = self.model.default_submesh_types
        submesh_types["negative electrode"] = pybamm.MeshGenerator(
            pybamm.Exponential1DSubMesh, submesh_params={"side": "right"}
        )

        if new_params is not None:
            param_values = new_params
        else:
            param_values = self.parameter_values

        self.sim = pybamm.Simulation(
            self.model,
            parameter_values=param_values,
            experiment=self.experiment,
            solver=pybamm.CasadiSolver(
                return_solution_if_failed_early=False, dt_max=0.1
            ),
            submesh_types=submesh_types,
            var_pts=self.var_pts,
        )
        self.parameters_list.append(self.parameter_values.copy())
        n_cycle = int(self.n_total_cycles[-1] + self.n_extrapolated)
        solution = self.sim.solve(
            showprogress=False, initial_soc=f"{start_voltage} V", calc_esoh=False
        )
        self.solution = solution
        self.n_total_cycles.append(int(n_cycle))
        return solution

    def save(self):
        n_cycle = self.n_total_cycles[-1]
        # self.sim.save(f"{self.fname}/cycle{str(n_cycle).zfill(4)}_sim.pkl")

        with open(f"{self.fname}/cycle{str(n_cycle).zfill(4)}_params.pkl", "wb") as f:
            pickle.dump(self.parameter_values, f)

    def create_output(self, soln=None):
        sim_df = pd.DataFrame()
        if soln is None:
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
        return sim_df

    def agg_steps(self):
        sim_df = self.sim_dfs[-1]
        sim_steps_df_first = sim_df.groupby("step_number").first()
        sim_steps_df_last = sim_df.groupby("step_number").last()
        sim_steps_df = sim_steps_df_first.join(
            sim_steps_df_last, lsuffix="_first", rsuffix="_last"
        )
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

        n_steps = len(
            list(filter(lambda x: not x.description.startswith("Hold"), exp_steps))
        )
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
        am_neg = (
            soln["Negative electrode active material volume fraction"]
            .entries[:, 0]
            .mean()
        )
        am_pos = (
            soln["Positive electrode active material volume fraction"]
            .entries[:, 0]
            .mean()
        )
        min_porosity_neg = soln["Negative electrode porosity"].entries[:, 0].min()
        sei_neg = soln["Negative total SEI thickness [m]"].entries[:, 0].mean()
        li_neg = soln[
            "X-averaged negative lithium plating concentration [mol.m-3]"
        ].entries[0]
        Ax = (
            soln["Current [A]"].entries[0]
            / soln["Total current density [A.m-2]"].entries[0]
        )

        self.state_variables.append(
            [am_pos, am_neg, n_lithium, min_porosity_neg, sei_neg, li_neg, Ax]
        )

    def extrapolate_states(self, n_delta: int = 100):
        # change in negative electrode porosity, positive/negative electrode concentration,
        # positive/negative active material fraction

        extrap_vars = {
            "Average negative particle concentration [mol.m-3]": (
                0,
                self.parameter_values[
                    "Maximum concentration in negative electrode [mol.m-3]"
                ],
                "Initial concentration in negative electrode [mol.m-3]",
                0,
            ),
            "Average positive particle concentration [mol.m-3]": (
                0,
                self.parameter_values[
                    "Maximum concentration in positive electrode [mol.m-3]"
                ],
                "Initial concentration in positive electrode [mol.m-3]",
                0,
            ),
            "Negative electrode porosity": (0, 0.25, "Negative electrode porosity", 1,),
            "Negative outer SEI thickness [m]": (
                0,
                1e8,
                "Initial outer SEI thickness [m]",
                0,
            ),
            "X-averaged negative dead lithium concentration [mol.m-3]": (
                0,
                20,
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
            extrap_vars[var_name] = (
                yvar_start,
                yvar_delta,
                param_name,
                ndim,
                n_delta_max,
            )

        if n_delta < 1:
            n_delta = 1
        smallest_var = min(extrap_vars, key=lambda x: extrap_vars[x][-1])
        print(f"Extrapolated {n_delta} cycles. Smallest n_delta from {smallest_var}.")
        self.n_extrapolated = n_delta

        for (
            var_name,
            (yvar_start, yvar_delta, param_name, ndim, _),
        ) in extrap_vars.items():
            yvar_new = yvar_start + n_delta * yvar_delta

            if ndim == 0:
                self.parameter_values.update(
                    {param_name: np.average(yvar_new.reshape(-1))}
                )
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
                            xarr,
                            yvar_new,
                            pybamm.standard_spatial_vars.x_n,
                            extrapolate=False,
                        )
                    }
                )

            else:
                raise ValueError("variable dimension not implemented")

            if var_name == "Negative electrode porosity":
                self.parameter_values.update(
                    {"Negative electrode minimum porosity": yvar_new.min(),}
                )

            if var_name == "X-averaged negative dead lithium concentration [mol.m-3]":
                self.parameter_values.update(
                    {
                        "Initial X-averaged plated lithium concentration [mol.m-3]": yvar_new.mean(),
                    }
                )

        return extrap_vars

    def _find_max_ndelta_from_state(
        self, var_name: str, yvar_max: np.array, yvar_min: np.array
    ):
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
        time_start = steps_df.loc[
            steps_df["step_type_first"] == "rest", "total_time_s_last"
        ].iloc[0]
        idx_start = np.where(soln["Time [s]"].entries == time_start)[0][0]

        if yvar.ndim == 1:
            yvar0 = np.around(yvar[0], 16)
            yvar_start = np.around(yvar[idx_start], 16)
            yvar_end = yvar[-1]

        elif yvar.ndim == 2:
            yvar0 = np.around(yvar[:, 0], 16)
            yvar_start = np.around(yvar[:, idx_start], 16)
            yvar_end = yvar[:, -1]

        else:
            raise ValueError("variable dimension not implemented")

        yvar_delta = np.around(yvar_end - yvar_start, 12)
        if np.all(yvar_delta > 0):
            n_delta_max = np.floor((yvar_max - yvar_start) / yvar_delta).min()
        elif np.all(yvar_delta < 0):
            n_delta_max = np.floor((yvar_min - yvar_start) / yvar_delta).min()
        else:
            n_delta_max = 1e5

        if np.abs(yvar_delta).min() > 0:
            n_delta_max_lim = np.floor(
                ((yvar_max - yvar_min) * 0.05 / np.abs(yvar_delta)).min()
            )
        else:
            n_delta_max_lim = 1e5

        if (var_name == "Negative dead lithium concentration [mol.m-3]") | (
            var_name == "X-averaged negative dead lithium concentration [mol.m-3]"
        ):
            yvar0 += np.around(
                soln["Negative lithium plating concentration [mol.m-3]"].entries[:, 0],
                12,
            )
        return yvar0, yvar_delta, min(n_delta_max, n_delta_max_lim)

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
                "Ax_m2",
            ],
        )
        try:
            sod_df["cycle_number"] = self.n_total_cycles[1:]
        except Exception as e:
            print(e)
            sod_df["cycle_number"] = self.n_total_cycles[1:-1]

        sod_df["LLI_%"] = (
            1 - sod_df["n_lithium_mol"] / sod_df["n_lithium_mol"].iloc[0]
        ) * 100
        sod_df["f_Ax"] = sod_df["Ax_m2"] / sod_df["Ax_m2"].iloc[0]
        sod_df["LAM_pe_%"] = (1 - sod_df["f_Ax"]) * 100
        sod_df["LAM_ne_%"] = (
            1 - sod_df["am_neg"] / sod_df["am_neg"].iloc[0] * sod_df["f_Ax"]
        ) * 100

        save_df = {
            "sim_data": sim_df,
            "steps_data": steps_df,
            "sod_data": sod_df,
        }
        with open(f"{self.fname}/sim_data.pkl", "wb") as f:
            pickle.dump(save_df, f)

    def parametric_sweep(
        self, param_name: str, param_values: list, start_voltage: float = 3.6
    ):
        """
        Perform a parametric sweep over a parameter.

        Args:
            param_name (str): Name of the parameter to sweep.
            param_values (list): List of values for the parameter.

        """
        self.param_results = {}

        def _run_one_case(
            param_value, param_name=param_name, start_voltage=start_voltage
        ):
            param_values = self.parameter_values.copy()
            param_values[param_name] = param_value
            print(f"Running simulation for {param_name} = {param_value}")
            soln = self.solve(start_voltage=start_voltage, new_params=param_values)
            sim_df = self.create_output(soln)
            self.param_results[param_value] = sim_df

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            tqdm(executor.map(_run_one_case, param_values), total=len(param_values))

    def parametric_sweep_multiple(
        self, param_sweep_dict: dict[str, list], start_voltage: float = 3.6
    ):
        """
        Perform a parametric sweep over multiple parameters.

        Args:
            param_dict (dict[str, list]): Dictionary of parameter names and values to sweep.
            start_voltage (float, optional): Starting voltage for the simulation. Defaults to 3.6.

        """
        self.param_results = {}
        params_init = self.parameter_values.copy()

        params_list = []
        for case_num, params in enumerate(zip(*param_sweep_dict.values())):
            params_updated = params_init.copy()
            for param_name, param_value in zip(param_sweep_dict.keys(), params):
                params_updated[param_name] = param_value
            params_list.append((case_num, params_updated))

        def _run_one_case(num_params_updated, start_voltage=start_voltage):
            print(f"Running simulation for case {num_params_updated[0]}")
            soln = self.solve(
                start_voltage=start_voltage, new_params=num_params_updated[1]
            )
            sim_df = self.create_output(soln)
            self.param_results[num_params_updated[0]] = sim_df

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            tqdm(
                executor.map(_run_one_case, params_list), total=len(params_list),
            )

    def optimize_params(
        self,
        params_range: dict[str, tuple[float, float, bool]],
        opt_input_dict,
        start_voltage: float,
        method: str = "Nelder-Mead",
    ):
        """
        Optimize parameters using an experiment DataFrame.

        Args:
            params_range (dict[str, tuple[float, float]]): Dictionary of parameters optimize and
                their min, max, whether it is log-scaled.
            opt_input_df (pd.DataFrame): DataFrame containing experimental data with columns:
                ["time", "current", "voltage"].
            start_voltage (float): Starting voltage for the simulation.
            method (str, optional): Optimization method. Defaults to "Bayesian". Options are
                ["Bayesian", "Powell", "Nelder-Mead"].

        """
        current_interp = pybamm.Interpolant(
            opt_input_dict["time"], opt_input_dict["current"], pybamm.t,
        )
        params0 = self.parameter_values.copy()
        params0.update({"Current function [A]": current_interp})

        def loss_func(x, param_names, sim_v0, sim_params):
            try:
                results_df = run_sim(x, param_names, sim_v0, sim_params)
                results_df_dch = results_df[results_df["step_type"] == "discharge"]
                voltage_loss_dch = rmse(
                    results_df_dch["voltage_real"], results_df_dch["voltage_sim"]
                )
                voltage_loss_all = rmse(
                    results_df["voltage_real"], results_df["voltage_sim"]
                )
                voltage_loss = voltage_loss_dch * 9 + voltage_loss_all
                print(f"{x} -> {voltage_loss}")
                return voltage_loss
            except Exception as e:
                print(e)
                return 1e3

        def run_sim(x, params_dict, sim_v0, sim_params):
            for param_name, param_value in zip(params_dict.keys(), x):
                if params_dict[param_name][2]:
                    param_value = 10 ** param_value
                sim_params[param_name] = param_value
            sim_df = self.create_output(
                soln=self.solve(start_voltage=sim_v0, new_params=sim_params)
            )
            sim_df.rename(
                columns={"total_time_s": "time", "voltage": "voltage_sim"}, inplace=True
            )
            if sim_df["time"].max() == opt_input_dict["time"].max():
                results_df = sim_df.copy()
                results_df["voltage_real"] = opt_input_dict["voltage"]
                results_df["current_real"] = opt_input_dict["current"]
                results_df["time_real"] = opt_input_dict["time"]

            else:
                results_df = pd.merge(
                    sim_df,
                    pd.DataFrame(opt_input_dict),
                    how="right",
                    on="time",
                    suffixes=("_sim", "_real"),
                )

            results_df["step_type"] = results_df["current_real"].apply(
                lambda x: "charge" if x < 0 else "discharge" if x > 0 else "rest"
            )
            results_df.replace(np.nan, 0, inplace=True)
            return results_df

        opt_results = minimize(
            loss_func,
            x0=[
                params0[v] if not params_range[v][2] else np.log10(params0[v])
                for v in params_range.keys()
            ],
            args=(params_range, start_voltage, params0),
            bounds=[params_range[v][0:2] for v in params_range.keys()],
            method=method,
            options={"xatol": 0.01, "fatol": 1e-4},
        )

        opt_sim_df = run_sim(opt_results.x, params_range, start_voltage, params0)

        return opt_sim_df, opt_results
