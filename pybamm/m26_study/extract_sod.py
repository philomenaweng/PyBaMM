import logging
import os
import re
import pickle
from glob import glob
from functools import reduce
from datetime import datetime
import time
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
from plotly.express.colors import sample_colorscale
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, minimize
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import pygad

import concurrent.futures

import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px

import warnings

warnings.filterwarnings("ignore")


colors = px.colors.qualitative.Plotly

pio.templates.default = "plotly_white"

folder_dir = "/local/data/philomenaweng/espeon_copy/projects/degradation/2022_m26/LongTermStudyRPTs/processed"
case_names = {
    "26043": "100% DOD",
    "26044": "100% DOD",
    "26045": "80% DOD",
    "26046": "80% DOD",
    "26047": "80%/50% DOD",
    "26048": "80%/50% DOD",
    "26049": "50% DOD",
    "26050": "50% DOD",
    "26051": "50%/30% DOD",
    "26052": "50%/30% DOD",
    "26053": "30% DOD",
    "26054": "30% DOD",
    "26055": "80% DOD w/ FC",
    "26056": "80% DOD w/ FC",
    "26057": "30% DOD w/ T21",
    "26058": "30% DOD w/ T21",
}
processed_files = [x for x in os.listdir(folder_dir) if x.endswith(".csv")]
processed_cells = sorted(
    list(set([re.findall(r"(\d+)", x)[0] for x in processed_files]))
)


def load_cycle_data(cell_id: str) -> pd.DataFrame:
    cycle_data_all = pd.read_csv(f"{folder_dir}/{cell_id}_cycle_data.csv")
    cycle_data_all["timestamp"] = pd.to_datetime(cycle_data_all["timestamp"])

    sign_map = {"charge": 1, "discharge": -1, "rest": 0}
    cycle_data_all["sign"] = cycle_data_all["state"].apply(lambda x: sign_map[x])
    cycle_data_all["step_capacity_Ah"] = (
        cycle_data_all["cellCapacity"] * cycle_data_all["sign"]
    )

    cycle_data_all["step_type_change"] = (
        (cycle_data_all["state"] != cycle_data_all["state"].shift(1))
        | (cycle_data_all["nodeID"] != cycle_data_all["nodeID"].shift(1))
    ).astype(int)
    cycle_data_all["step_number"] = np.cumsum(cycle_data_all["step_type_change"])

    step_df = []
    for node_id in cycle_data_all["nodeID"].unique():
        cycle_data = cycle_data_all[cycle_data_all["nodeID"] == node_id]
        step_cap_df = np.cumsum(
            cycle_data.groupby(by="step_number").last()["step_capacity_Ah"]
        )
        step_cap_df -= step_cap_df.min()
        step_df.append(step_cap_df)

    step_df = pd.concat(step_df)
    cycle_data_all["step_Q_cum"] = cycle_data_all["step_number"].apply(
        lambda x: step_df[max(x - 1, 1)]
    )
    cycle_data_all["cellCapacity"] = (
        cycle_data_all["step_capacity_Ah"] + cycle_data_all["step_Q_cum"]
    )
    return cycle_data_all


def find_slope(x_arr, y_arr):
    def lin_func(x, a, b):
        return a * x + b

    popt, _ = curve_fit(lin_func, xdata=x_arr, ydata=y_arr)
    return popt[0]


def filter_out_slow_charge(tmp_df: pd.DataFrame,
                           shift_dqdv_v: bool=False,
                           upper_v: Optional[float] = np.nan,
                           lower_v: Optional[float] = np.nan,
                           ):
    assert tmp_df["nodeID"].nunique() == 1

    steps_df_start = (
        tmp_df[["step_number", "voltage", "c_rate", "state"]]
        .groupby(by=["step_number"])
        .first()
    )
    steps_df_end = (
        tmp_df[["step_number", "voltage", "c_rate", "step_time_s"]]
        .groupby(by=["step_number"])
        .last()
    )
    steps_df = pd.merge(
        steps_df_start, steps_df_end, on=["step_number"], suffixes=("_start", "_end")
    ).reset_index()

    slow_chg_steps = steps_df[
        (steps_df["state"] == "charge")
        & (steps_df["c_rate_start"].abs() < 0.05)
        & (steps_df["voltage_start"] < 3.5)
        & (steps_df["voltage_end"] > 4)
    ]
    chg_fit_df = tmp_df[tmp_df["step_number"].isin(set(slow_chg_steps["step_number"]))]
    if not np.isnan(upper_v):
        chg_fit_df = chg_fit_df[chg_fit_df["voltage"] <= upper_v].copy()
    if not np.isnan(lower_v):
        chg_fit_df = chg_fit_df[chg_fit_df["voltage"] >= lower_v].copy()

    if shift_dqdv_v:
        rac_df = steps_df[
            steps_df["step_number"] == chg_fit_df["step_number"].max() + 1
        ]
        assert len(rac_df) == 1
        del_dqdv_v = rac_df["voltage_start"].values[0] - rac_df["voltage_end"].values[0]
    else:
        del_dqdv_v = 0

    # rest steps
    tmp_df["prev_state"] = tmp_df["state"].shift(1)
    rest_df = tmp_df[tmp_df["state"] == "rest"].copy()

    def calculate_dVdt(sub_df, last_x_s):
        # Note this will be affected by sampling time!
        last_x_df = sub_df[
            sub_df["step_time_s"] >= sub_df["step_time_s"].max() - last_x_s
        ]
        return find_slope(last_x_df.step_time_s, last_x_df.voltage) * 1e3
        # def lin_func(x, a, b):
        #     return a*x + b
        # popt, _ = curve_fit(lin_func, xdata=last_x_df.step_time_s, ydata=last_x_df.voltage)
        # return popt[0]

    dVdt_df = pd.DataFrame(
        rest_df.groupby(by="step_number").apply(lambda x: calculate_dVdt(x, 600)),
        columns=["dVdt_mV/s"],
    ).reset_index()
    prev_state_df = (
        rest_df[["step_number", "state", "prev_state"]]
        .groupby(by="step_number")
        .first()
        .reset_index()
    )
    last_q_v_df = (
        rest_df[["step_number", "step_time_s", "voltage", "cellCapacity"]]
        .groupby(by="step_number")
        .last()
        .reset_index()
    )

    rest_fit_df = reduce(
        lambda df1, df2: pd.merge(df1, df2, on="step_number"),
        [dVdt_df, prev_state_df, last_q_v_df],
    )

    real_dict = {
        "fc_v_arr": np.array(chg_fit_df.voltage),
        "fc_q_arr": np.array(chg_fit_df.cellCapacity),
    }

    real_dqdv_dict = find_dqdv(
        df_q=real_dict["fc_q_arr"], df_v=real_dict["fc_v_arr"], del_dqdv_v=del_dqdv_v
    )
    real_dict.update(real_dqdv_dict)

    return real_dict, rest_fit_df.reset_index(drop=True)


def find_dqdv(df_v, df_q, dv=0.005, dq=0.03, del_dqdv_v=0):

    tmp_df = pd.DataFrame()
    tmp_df["v"] = df_v
    tmp_df["q"] = df_q

    def find_dqdv_q(q_target, data_df, q_range=0.005):
        df_vicinity = data_df[np.abs(data_df["q"] - q_target) <= q_range]
        scale = 2
        while len(df_vicinity) < 3:
            q_range_new = q_range * scale
            scale += 1
            df_vicinity = data_df[np.abs(data_df["q"] - q_target) <= q_range_new]
        # return find_slope(df_vicinity.v, df_vicinity.q)
        return np.polyfit(df_vicinity.v, df_vicinity.q, 1)[0]

    def find_dqdv_v(v_target, data_df, v_range=0.012):
        df_vicinity = data_df[np.abs(data_df["v"] - v_target) <= v_range]
        scale = 2
        while len(df_vicinity) < 3:
            v_range_new = v_range * scale
            scale += 1
            df_vicinity = data_df[np.abs(data_df["v"] - v_target) <= v_range_new]
        # return find_slope(df_vicinity.v, df_vicinity.q)
        return np.polyfit(df_vicinity.v, df_vicinity.q, 1)[0]

    q_arr = np.arange(df_q.min(), df_q.max(), dq)
    dqdv_q_arr = list(map(lambda x: find_dqdv_q(x, tmp_df, dq), q_arr))
    v_arr = np.arange(df_v.min(), df_v.max(), dv)
    dqdv_v_arr = list(map(lambda x: find_dqdv_v(x, tmp_df, dv), v_arr))

    v_arr -= del_dqdv_v
    return {"dqdv_q": (q_arr, dqdv_q_arr), "dqdv_v": (v_arr, dqdv_v_arr)}


def extract_peic(
    real_data: Tuple[Dict[str, np.ndarray], pd.DataFrame],
    pe_ocv_low: float = 3.95,
    pe_ocv_high: float = 4.0,
    v_shifted: bool = False,
) -> float:
    real_dict, rest_data_df = real_data

    if v_shifted == True:
        delV = 0
    else:
        rev = rest_data_df[rest_data_df["prev_state"] == "charge"]["voltage"].values[0]
        delV = real_dict["fc_v_arr"].max() - rev

    dqdv_filt = np.where(
        (real_dict["dqdv_v"][0] >= (pe_ocv_low + delV))
        & (real_dict["dqdv_v"][0] <= (pe_ocv_high + delV)),
        real_dict["dqdv_v"][1],
        np.nan,
    )
    return np.nanmean(dqdv_filt)


def simulate_ocv(
    sod,
    fc_params,
    half_cell_names=("M26_2006_half_cell_pe_Co48_ch", "M26_2001_half_cell_ne_Co48_dch"),
):
    # SOD = (LAM_pe, LAM_ne, LLI)
    qpe0 = fc_params["qpe_load"] / 1e3
    qne0 = fc_params["qne_load"] / 1e3
    ofs0 = fc_params["ofs"]

    half_cell_path = "/local/data/philomenaweng/repos/bi-sox/resources/sod/"
    pe_fname = f"{half_cell_path}{half_cell_names[0]}.pkl"
    ne_fname = f"{half_cell_path}{half_cell_names[1]}.pkl"
    with open(pe_fname, "rb") as f:
        pe_list = pickle.load(f)
    with open(ne_fname, "rb") as f:
        ne_list = pickle.load(f)

    half_cell_ocvs = {
        "pe_soc": pe_list[-1],
        "pe_ocv": pe_list[0],
        "ne_soc": ne_list[-1],
        "ne_ocv": ne_list[0],
    }

    pe_q_arr0 = qpe0 * (half_cell_ocvs["pe_soc"] - ofs0)
    ne_q_arr0 = qne0 * (half_cell_ocvs["ne_soc"])
    # fc_q0 = min(pe_q_arr0.max(), ne_q_arr0.max()) - max(pe_q_arr0.min(), ne_q_arr0.min())
    # pe_q_arr = (pe_q_arr0 - fc_q0 * sod[2]) * (1 - sod[0])
    pe_q_arr = (pe_q_arr0 - pe_q_arr0.max() * sod[2]) * (1 - sod[0])
    ne_q_arr = ne_q_arr0 * (1 - sod[1])

    pe_v_arr = half_cell_ocvs["pe_ocv"]
    ne_v_arr = half_cell_ocvs["ne_ocv"]

    fc_q_arr = np.arange(
        max(pe_q_arr.min(), ne_q_arr.min()), min(pe_q_arr.max(), ne_q_arr.max()), 1e-3
    )
    f_pe = interp1d(x=pe_q_arr, y=pe_v_arr)
    f_ne = interp1d(x=ne_q_arr, y=ne_v_arr)
    fc_v_arr = f_pe(fc_q_arr) - f_ne(fc_q_arr)

    if len(fc_q_arr) <= 1:
        print(f"SOD NOT FEASIBLE! {sod}")
        fc_q_arr = np.linspace(0, max(qpe0, qne0))
        fc_v_arr = np.zeros(100)
        sim_dict = {
            "pe_q_arr": pe_q_arr,
            "pe_v_arr": pe_v_arr,
            "ne_q_arr": ne_q_arr,
            "ne_v_arr": ne_v_arr,
            "fc_q_arr": fc_q_arr,
            "fc_v_arr": fc_v_arr,
            "dqdv_q": (np.linspace(0, max(qpe0, qne0)), np.zeros(50)),
            "dqdv_v": (np.linspace(0, 4), np.zeros(50)),
        }

    else:
        # print("SOD FEASIBLE!")
        sim_dict = {
            "pe_q_arr": pe_q_arr,
            "pe_v_arr": pe_v_arr,
            "ne_q_arr": ne_q_arr,
            "ne_v_arr": ne_v_arr,
            "fc_q_arr": fc_q_arr,
            "fc_v_arr": fc_v_arr,
        }

        sim_dqdv_dict = find_dqdv(df_q=fc_q_arr, df_v=fc_v_arr)
        sim_dict.update(sim_dqdv_dict)

    return sim_dict


def opt_func(sod, data_dict, rest_df, fc_params, hc_names):
    sim_dict = simulate_ocv(sod, fc_params, hc_names)
    # dQdV - V
    f_sim_dqdv_v = interp1d(
        x=sim_dict["dqdv_v"][0],
        y=sim_dict["dqdv_v"][1],
        bounds_error=False,
        fill_value=0,
    )
    mse = (
        np.sqrt(
            mean_squared_error(
                data_dict["dqdv_v"][1], f_sim_dqdv_v(data_dict["dqdv_v"][0])
            )
        )
        * 1e3
    )  # ~300

    return mse


def opt_func_batch(sods, data_dict, rest_df, fc_params, hc_names):
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        mse_arr = list(
            executor.map(
                opt_func,
                sods,
                [data_dict] * len(sods),
                [rest_df] * len(sods),
                [fc_params] * len(sods),
                [hc_names] * len(sods),
            )
        )
    return mse_arr


def optimize_sod_with_ga(
    real_data,
    sim_params,
    num_generations=300,  # 300 #####################################
    num_parents_mating=4,
    fitness_batch_size=8,
    sol_per_pop=8,
    num_genes=3,
    init_range_low=-0.5,
    init_range_high=0.5,
    parent_selection_type="tournament",
    crossover_type="single_point",
    mutation_type="random",
    mutation_percent_genes=50,
    opt_lims=((-0.2, 0.5), (-0.2, 0.5), (-0.3, 0.3))
):
    def generate_fitness_function(real_data, sim_params):
        chg_dict, rest_df = real_data
        fc_params, hc_names = sim_params

        def fitness_func(ga_instance, sod, solution_idx):
            if np.shape(sod) == (3,):
                mse = opt_func(sod, chg_dict, rest_df, fc_params, hc_names)
                return 1 / mse
            else:
                mse_arr = opt_func_batch(sod, chg_dict, rest_df, fc_params, hc_names)
                return 1 / np.array(mse_arr)

        return fitness_func

    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        fitness_batch_size=fitness_batch_size,
        fitness_func=generate_fitness_function(real_data, sim_params),
        sol_per_pop=sol_per_pop,
        num_genes=num_genes,
        init_range_low=init_range_low,
        init_range_high=init_range_high,
        parent_selection_type=parent_selection_type,
        crossover_type=crossover_type,
        mutation_type=mutation_type,
        mutation_percent_genes=mutation_percent_genes,
        gene_space=[
            {"low": opt_lims[0][0], "high": opt_lims[0][1]},
            {"low": opt_lims[1][0], "high": opt_lims[1][1]},
            {"low": opt_lims[2][0], "high": opt_lims[2][1]},
        ],
    )
    ga_instance.run()
    return ga_instance


def extract_sod_with_ga(
        real_data,
        fc_params_init,
        hc_names,
        opt_lims = ((-0.2, 0.5), (-0.2, 0.5), (-0.3, 0.3))
        ):

    sim_params = (fc_params_init, hc_names)
    ga_opt = optimize_sod_with_ga(real_data, sim_params)

    local_opt = minimize(
        opt_func,
        x0=ga_opt.best_solution()[0],
        bounds=opt_lims,
        args=(*real_data, *sim_params),
        method="Powell",
        # options={
        #     "maxiter":10 ######################################
        #         }
    )

    opt_sod = local_opt.x
    print(f"Initialized SOD: {opt_sod}")
    fc_params = {
        "ofs": np.around(opt_sod[2], 4),
        "qpe_load": np.around(fc_params_init["qpe_load"] * (1 - opt_sod[0]), 2),
        "qne_load": np.around(fc_params_init["qne_load"] * (1 - opt_sod[1]), 2),
    }

    return fc_params, local_opt, ga_opt


def extract_sod_evolution(
        cell_id: str,
        shift_dqdv: bool,
        savepath: str,
        overwrite: bool=False,
        use_peic: bool=False,
        upper_v_cutoff: Optional[float]=np.nan,
):
    start_time = time.time()
    if os.path.exists(savepath) and not overwrite:
        raise ValueError("File already exists. Set overwrite=True to overwrite.")
    else:
        os.makedirs(savepath, exist_ok=True)

    cycle_data_all = load_cycle_data(cell_id)
    rpt_nums = sorted(
        cycle_data_all["nodeID"].unique()
    )  # [:3] ##################################

    # Find first set of SOD using GA
    cycle_data = cycle_data_all[cycle_data_all["nodeID"] == rpt_nums[0]].copy()
    real_data = filter_out_slow_charge(cycle_data, shift_dqdv)

    if use_peic:
        real_data_cropped = filter_out_slow_charge(cycle_data, shift_dqdv, upper_v=upper_v_cutoff)
        peic0 = extract_peic(real_data, v_shifted=shift_dqdv)
    else:
        real_data_cropped = real_data.copy()

    fc_params_init = {"ofs": 0, "qpe_load": 3200, "qne_load": 3000}
    hc_names = ("M26_2006_half_cell_pe_Co48_ch", "M26_2001_half_cell_ne_Co48_dch")

    fc_params, local_opt, ga_opt = extract_sod_with_ga(
        real_data_cropped, fc_params_init, hc_names
    )
    ga_opt.plot_fitness()

    records = []
    sod = np.array([0, 0, 0])

    for rpt_num in rpt_nums:
        if (rpt_num == 22.02) | (rpt_num == 54.02):
            continue

        try:
            real_data = filter_out_slow_charge(
                cycle_data_all[cycle_data_all["nodeID"] == rpt_num].copy(),
                shift_dqdv_v=shift_dqdv,
            )
            if use_peic:
                real_data_cropped = filter_out_slow_charge(
                    cycle_data_all[cycle_data_all["nodeID"] == rpt_num].copy(),
                    shift_dqdv_v=shift_dqdv,
                    upper_v=upper_v_cutoff,
                )
                peic = extract_peic(real_data, v_shifted=shift_dqdv)
                lam_pe_peic = 1 - peic / peic0
                print(f"[{cell_id}]{rpt_num}: LAM_pe from PEIC {lam_pe_peic}")
                lam_pe_lims = (lam_pe_peic-0.02, lam_pe_peic+0.02)
            else:
                real_data_cropped = real_data.copy()
                lam_pe_lims = (-0.05, 0.5)

            lam_ne_lims = (-0.05, 0.5)
            lli_lims = (-0.05, 0.3)
            sim_params = (fc_params, hc_names)
            local_opt = minimize(
                opt_func,
                x0=sod,
                bounds=(lam_pe_lims, lam_ne_lims, lli_lims),
                args=(*real_data_cropped, *sim_params),
                method="Powell",
            )

            if not local_opt.success:
                print(f"[{cell_id}]Optimization failed")
                results_df = pd.DataFrame(
                    records,
                    columns=["cell_id", "rpt_num", "LAM_pe", "LAM_ne", "LLI", "rmse"],
                )
                results_df.to_csv(f"{savepath}/{cell_id}_ga_FAIL.csv", index=False)
                return results_df
            else:
                if (
                    np.abs(sod - local_opt.x).max()
                    > 0.03  ######################################
                ):
                    print(
                        f"[{cell_id}]Running global opt with GA for {rpt_num}: {local_opt.x}"
                    )
                    _, local_opt, _ = extract_sod_with_ga(
                        real_data_cropped, fc_params, hc_names, (lam_pe_lims, lam_ne_lims, lli_lims)
                    )
                    print(
                        f"[{cell_id}]Time elapsed: {(time.time() - start_time)/60:.0f}min"
                    )
                sod = np.around(local_opt.x, 4)
                print(f"[{cell_id}]{rpt_num}: {sod}; {local_opt.fun}")
                records.append((cell_id, rpt_num, *sod, local_opt.fun))

        except Exception as e:
            print(f"[{cell_id}]Error on {rpt_num}: {e}")

    results_df = pd.DataFrame(
        records, columns=["cell_id", "rpt_num", "LAM_pe", "LAM_ne", "LLI", "rmse",],
    )
    for k in list(fc_params):
        results_df[k] = fc_params[k]
    results_df.to_csv(f"{savepath}/{cell_id}_ga.csv", index=False)
    return results_df, ga_opt


if __name__ == "__main__":

    # for handler in logging.root.handlers[:]:
    #     logging.root.removeHandler(handler)

    savepath = "/local/data/philomenaweng/projects/degradation/m26/extract_sod/v3"
    overwrite = True
    shift_dqdv = True
    results = []
    start_time = time.time()
    use_peic = True
    upper_v_cutoff = 3.8

    def extract_sod_concurr(cell_id):
        return cell_id, extract_sod_evolution(cell_id, shift_dqdv, savepath, overwrite, use_peic, upper_v_cutoff)

    cell_id_list = processed_cells  # [:6]  ############################
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        tqdm(executor.map(extract_sod_concurr, cell_id_list), total=len(cell_id_list))
